import math
from typing import Tuple, Union, Optional

import torch
from torch import nn
from torch.nn import functional as F
from torch import Tensor
from transformers.models.speech_to_text.modeling_speech_to_text import (
    shift_tokens_right,
    Speech2TextDecoder,
)
from transformers.models.llama.modeling_llama import (
    LlamaConfig,
    LlamaRotaryEmbedding,
)

from leap.model import BaseModel


def gem(x, p=3, eps=1e-6):
    return F.avg_pool2d(x.clamp(min=eps).pow(p), (x.size(-2), x.size(-1))).pow(1.0 / p)


class GeM(nn.Module):
    def __init__(self, p=3, eps=1e-6, p_trainable=False):
        super(GeM, self).__init__()
        if p_trainable:
            self.p = nn.Parameter(torch.ones(1) * p)
        else:
            self.p = p
        self.eps = eps

    def forward(self, x):
        ret = gem(x, p=self.p, eps=self.eps)
        return ret

    def __repr__(self):
        return (self.__class__.__name__  + f"(p={self.p.data.tolist()[0]:.4f},eps={self.eps})")


class Decoder(nn.Module):
    def __init__(self, decoder_config):
        super(Decoder, self).__init__()

        self.config = decoder_config
        self.decoder = Speech2TextDecoder(decoder_config)
        self.lm_head = nn.Linear(decoder_config.d_model, decoder_config.vocab_size, bias=False)

        self.decoder_start_token_id = decoder_config.decoder_start_token_id
        self.decoder_pad_token_id = decoder_config.pad_token_id #used for early stopping
        self.decoder_end_token_id= decoder_config.eos_token_id

    def forward(self, x, labels=None, attention_mask = None, encoder_attention_mask = None):

        if labels is not None:
            decoder_input_ids = shift_tokens_right(labels, self.config.pad_token_id, self.config.decoder_start_token_id)

        decoder_outputs = self.decoder(input_ids=decoder_input_ids,
                                       encoder_hidden_states=x,
                                       attention_mask = attention_mask,
                                       encoder_attention_mask = encoder_attention_mask)
        lm_logits = self.lm_head(decoder_outputs.last_hidden_state)
        return lm_logits

    def generate(self, x, max_new_tokens=33, encoder_attention_mask=None):
        decoder_input_ids = torch.ones((x.shape[0], 1), device=x.device, dtype=torch.long).fill_(self.decoder_start_token_id)
        for i in range(max_new_tokens-1):
            decoder_outputs = self.decoder(input_ids=decoder_input_ids,encoder_hidden_states=x, encoder_attention_mask=encoder_attention_mask)
            logits = self.lm_head(decoder_outputs.last_hidden_state)
            decoder_input_ids = torch.cat([decoder_input_ids,logits.argmax(2)[:,-1:]],dim=1)

            if torch.all((decoder_input_ids==self.decoder_end_token_id).sum(-1) > 0):
                break

        return decoder_input_ids


class Swish(nn.Module):
    def __init__(self) -> None:
        super(Swish, self).__init__()

    def forward(self, inputs: Tensor) -> Tensor:
        return inputs * inputs.sigmoid()


class GLU(nn.Module):
    def __init__(self, dim: int) -> None:
        super(GLU, self).__init__()
        self.dim = dim

    def forward(self, inputs: Tensor) -> Tensor:
        outputs, gate = inputs.chunk(2, dim=self.dim)
        return outputs * gate.sigmoid()


class FeedForwardModule(nn.Module):
    def __init__(
        self,
        encoder_dim: int = 512,
        expansion_factor: int = 4,
        dropout_p: float = 0.1,
    ) -> None:
        super(FeedForwardModule, self).__init__()
        self.ffn1 = nn.Linear(encoder_dim, encoder_dim * expansion_factor, bias=True)
        self.act = Swish()
        self.do1 = nn.Dropout(p=dropout_p)
        self.ffn2 = nn.Linear(encoder_dim * expansion_factor, encoder_dim, bias=True)
        self.do2 = nn.Dropout(p=dropout_p)

    def forward(self, x):
        x = self.ffn1(x)
        x = self.act(x)
        x = self.do1(x)
        x = self.ffn2(x)
        x = self.do2(x)
        return x


class RelPositionalEncoding(nn.Module):
    """
    Relative positional encoding module.
    Args:
        d_model: Embedding dimension.
        max_len: Maximum input length.
    """

    def __init__(self, d_model: int = 512, max_len: int = 60) -> None:
        super(RelPositionalEncoding, self).__init__()
        self.d_model = d_model
        self.pe = None
        self.extend_pe(torch.tensor(0.0).expand(1, max_len))

    def extend_pe(self, x):
        if self.pe is not None:
            if self.pe.size(1) >= x.size(1) * 2 - 1:
                if self.pe.dtype != x.dtype or self.pe.device != x.device:
                    self.pe = self.pe.to(dtype=x.dtype, device=x.device)
                return

        pe_positive = torch.zeros(x.size(1), self.d_model)
        pe_negative = torch.zeros(x.size(1), self.d_model)
        position = torch.arange(0, x.size(1), dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, self.d_model, 2, dtype=torch.float32) * -(math.log(10000.0) / self.d_model)
        )
        pe_positive[:, 0::2] = torch.sin(position * div_term)
        pe_positive[:, 1::2] = torch.cos(position * div_term)
        pe_negative[:, 0::2] = torch.sin(-1 * position * div_term)
        pe_negative[:, 1::2] = torch.cos(-1 * position * div_term)

        pe_positive = torch.flip(pe_positive, [0]).unsqueeze(0)
        pe_negative = pe_negative[1:].unsqueeze(0)
        pe = torch.cat([pe_positive, pe_negative], dim=1)
        self.pe = pe.to(device=x.device, dtype=x.dtype)

    def forward(self, x: torch.Tensor):
        self.extend_pe(x)
        pos_emb = self.pe[
            :,
            self.pe.size(1) // 2 - x.size(1) + 1 : self.pe.size(1) // 2 + x.size(1),
        ]
        return pos_emb


class DepthwiseConv1d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int,
                 stride: int = 1, padding: int = 0, bias: bool = False) -> None:
        super(DepthwiseConv1d, self).__init__()
        assert out_channels % in_channels == 0, "out_channels should be constant multiple of in_channels"
        self.conv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            groups=in_channels,
            stride=stride,
            padding=padding,
            bias=bias,
        )

    def forward(self, inputs: Tensor) -> Tensor:
        return self.conv(inputs)


class DepthwiseConv2d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: Union[int, Tuple], stride: int = 2, padding: int = 0) -> None:
        super(DepthwiseConv2d, self).__init__()
        assert out_channels % in_channels == 0, "out_channels should be constant multiple of in_channels"
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=in_channels,
        )

    def forward(self, inputs: Tensor) -> Tensor:
        return self.conv(inputs)


class PointwiseConv1d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1, padding: int = 0, bias: bool = True) -> None:
        super(PointwiseConv1d, self).__init__()
        self.conv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1,
            stride=stride,
            padding=padding,
            bias=bias,
        )

    def forward(self, inputs: Tensor) -> Tensor:
        return self.conv(inputs)


class ConvModule(nn.Module):
    def __init__(self, in_channels: int, kernel_size: int = 3, expansion_factor: int = 2, dropout_p: float = 0.0) -> None:
        super(ConvModule, self).__init__()
        assert (kernel_size - 1) % 2 == 0, "kernel_size should be a odd number for 'SAME' padding"
        assert expansion_factor == 2, "Currently, Only Supports expansion_factor 2"

        self.pw_conv_1 = PointwiseConv1d(in_channels, in_channels * expansion_factor, stride=1, padding=0, bias=True)
        self.act1 = GLU(dim=1)
        self.dw_conv = DepthwiseConv1d(in_channels, in_channels, kernel_size, stride=1, padding=(kernel_size - 1) // 2)
        self.bn = nn.BatchNorm1d(in_channels)
        self.act2 = Swish()
        self.pw_conv_2 = PointwiseConv1d(in_channels, in_channels, stride=1, padding=0, bias=True)
        self.do = nn.Dropout(p=dropout_p)

    def forward(self, x, mask_pad):
        # mask batch padding
        x = x.transpose(1, 2)
        if mask_pad.size(2) > 0:  # time > 0
            x = x.masked_fill(~mask_pad, 0.0)
        x = self.pw_conv_1(x)
        x = self.act1(x)
        x = self.dw_conv(x)
        # torch.Size([4, 128, 384])
        x_bn = x.permute(0,2,1).reshape(-1, x.shape[1])
        mask_bn = mask_pad.view(-1)
        x_bn[mask_bn] = self.bn(x_bn[mask_bn])
        x = x_bn.view(x.permute(0,2,1).shape).permute(0,2,1)
        '''
        x = self.bn(x)
        '''
        x = self.act2(x)
        x = self.pw_conv_2(x)
        x = self.do(x)
        # mask batch padding
        if mask_pad.size(2) > 0:  # time > 0
            x = x.masked_fill(~mask_pad, 0.0)
        x = x.transpose(1, 2)
        return x


def make_scale(encoder_dim):
    scale = nn.Parameter(torch.tensor([1.] * encoder_dim)[None,None,:])
    bias = nn.Parameter(torch.tensor([0.] * encoder_dim)[None,None,:])
    return scale, bias


class SqueezeformerBlock(nn.Module):
    def __init__(self, encoder_dim: int = 512, num_attention_heads: int = 8, feed_forward_expansion_factor: int = 4,
                 conv_expansion_factor: int = 2, feed_forward_dropout_p: float = 0.1, attention_dropout_p: float = 0.1,
                 conv_dropout_p: float = 0.1, conv_kernel_size: int = 3):
        super(SqueezeformerBlock, self).__init__()
        self.scale_mhsa, self.bias_mhsa = make_scale(encoder_dim)
        self.scale_ff_mhsa, self.bias_ff_mhsa = make_scale(encoder_dim)
        self.scale_conv, self.bias_conv = make_scale(encoder_dim)
        self.scale_ff_conv, self.bias_ff_conv = make_scale(encoder_dim)

        self.mhsa_llama = LlamaAttention(
            LlamaConfig(
                hidden_size=encoder_dim,
                num_attention_heads=num_attention_heads,
                max_position_embeddings=384,
            )
        )
        self.ln_mhsa = nn.LayerNorm(encoder_dim)

        self.ff_mhsa = FeedForwardModule(
            encoder_dim=encoder_dim,
            expansion_factor=feed_forward_expansion_factor,
            dropout_p=feed_forward_dropout_p,
        )
        # Attention_mask = (bsz, self.num_heads, q_len, kv_seq_len)
        self.ln_ff_mhsa = nn.LayerNorm(encoder_dim)
        self.conv = ConvModule(
            in_channels=encoder_dim,
            kernel_size=conv_kernel_size,
            expansion_factor=conv_expansion_factor,
            dropout_p=conv_dropout_p,
        )
        self.ln_conv = nn.LayerNorm(encoder_dim)
        self.ff_conv = FeedForwardModule(
            encoder_dim=encoder_dim,
            expansion_factor=feed_forward_expansion_factor,
            dropout_p=feed_forward_dropout_p,
        )
        self.ln_ff_conv = nn.LayerNorm(encoder_dim)

    def forward(self, x, cos, sin, mask):
        mask_pad = ( mask).long().bool().unsqueeze(1)
        mask_pad = ~( mask_pad.permute(0, 2,1) * mask_pad)
        mask_flat = mask.reshape(-1).bool()
        bs, slen, nfeats = x.shape

        residual = x
        x = x * self.scale_mhsa.to(x.dtype) + self.bias_mhsa.to(x.dtype)
        x = residual + self.mhsa_llama(x, cos, sin, attention_mask = mask_pad.unsqueeze(1) )[0]
        # Skip pad #1
        x_skip = x.reshape(-1, x.shape[-1])
        x = x_skip[mask_flat].unsqueeze(0)
        x = self.ln_mhsa(x) #casts to fp32
        residual = x
        #32
        x = x * self.scale_ff_mhsa.to(x.dtype) + self.bias_ff_mhsa.to(x.dtype)
        #32
        x = residual + self.ff_mhsa(x)
        #32
        x = self.ln_ff_mhsa(x) #casts to fp32
        # Unskip pad #1
#         print(x_skip[mask_flat].dtype, x[0].dtype)
        x_skip[mask_flat] = x[0].to(x_skip.dtype)
        x = x_skip.reshape(bs, slen, nfeats)
        residual = x
        # torch.Size([16, 384, 128])
        x = x * self.scale_conv.to(x.dtype) + self.bias_conv.to(x.dtype)
        x = residual + self.conv(x, mask_pad = mask.bool().unsqueeze(1))
        # Skip pad #2
        x_skip = x.reshape(-1, x.shape[-1])
        x = x_skip[mask_flat].unsqueeze(0)

        x = self.ln_conv(x)

        residual = x
        x = x * self.scale_ff_conv.to(x.dtype) + self.bias_ff_conv.to(x.dtype)
        x = residual + self.ff_conv(x)
        x = self.ln_ff_conv(x)
        # Unskip pad #2
        x_skip[mask_flat] = x[0].to(x_skip.dtype)
        x = x_skip.view(bs, slen, nfeats)

        return x


class SqueezeformerEncoder(nn.Module):
    """
    Squeezeformer encoder first processes the input with a convolution subsampling layer and then
    with a number of squeezeformer blocks.

    Args:
        input_dim (int, optional): Dimension of input vector
        encoder_dim (int, optional): Dimension of squeezeformer encoder
        num_layers (int, optional): Number of squeezeformer blocks
        reduce_layer_index (int, optional): The layer index to reduce sequence length
        recover_layer_index (int, optional): The layer index to recover sequence length
        num_attention_heads (int, optional): Number of attention heads
        feed_forward_expansion_factor (int, optional): Expansion factor of feed forward module
        conv_expansion_factor (int, optional): Expansion factor of squeezeformer convolution module
        feed_forward_dropout_p (float, optional): Probability of feed forward module dropout
        attention_dropout_p (float, optional): Probability of attention module dropout
        conv_dropout_p (float, optional): Probability of squeezeformer convolution module dropout
        conv_kernel_size (int or tuple, optional): Size of the convolving kernel
        half_step_residual (bool): Flag indication whether to use half step residual or not
    Inputs: inputs, input_lengths
        - **inputs** (batch, time, dim): Tensor containing input vector
        - **input_lengths** (batch): list of sequence input lengths
    Returns: outputs, output_lengths
        - **outputs** (batch, out_channels, time): Tensor produces by squeezeformer encoder.
        - **output_lengths** (batch): list of sequence output lengths
    """

    def __init__(
        self,
        input_dim: int = 80,
        encoder_dim: int = 512,
        num_layers: int = 16,
        reduce_layer_indices: list = [],
        recover_layer_index: int = 15,
        num_attention_heads: int = 8,
        feed_forward_expansion_factor: int = 4,
        conv_expansion_factor: int = 2,
        input_dropout_p: float = 0.1,
        feed_forward_dropout_p: float = 0.1,
        attention_dropout_p: float = 0.1,
        conv_dropout_p: float = 0.1,
        conv_kernel_size: int = 31,
    ):
        super(SqueezeformerEncoder, self).__init__()
        self.num_layers = num_layers
        self.recover_tensor = None

        self.blocks = nn.ModuleList()
        for _ in range(num_layers):
            self.blocks.append(
                SqueezeformerBlock(
                    encoder_dim=encoder_dim,
                    num_attention_heads=num_attention_heads,
                    feed_forward_expansion_factor=feed_forward_expansion_factor,
                    conv_expansion_factor=conv_expansion_factor,
                    feed_forward_dropout_p=feed_forward_dropout_p,
                    attention_dropout_p=attention_dropout_p,
                    conv_dropout_p=conv_dropout_p,
                    conv_kernel_size=conv_kernel_size,
                )
            )

    def forward(self, x: Tensor, cos: Tensor, sin: Tensor, mask: Tensor):
        outputs = x
        output_lengths = mask.sum(-1)
        for layer in self.blocks:
            # Rebuild mask
            seq = torch.arange(outputs.shape[1], device=x.device)
            outputs_mask = (seq < output_lengths.unsqueeze(-1)).long()
            outputs = layer(outputs, cos, sin, outputs_mask)
        return outputs, outputs_mask


class LlamaAttention(nn.Module):
    def __init__(self, config: LlamaConfig):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.max_position_embeddings = config.max_position_embeddings

        if (self.head_dim * self.num_heads) != self.hidden_size:
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
                f" and `num_heads`: {self.num_heads})."
            )
        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)
        #self.rotary_emb = LlamaRotaryEmbedding(self.head_dim, max_position_embeddings=self.max_position_embeddings)

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def forward(
        self,
        hidden_states: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,

    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states).view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)

        kv_seq_len = key_states.shape[-2]
        if past_key_value is not None:
            kv_seq_len += past_key_value[0].shape[-2]
        '''
        cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)
        '''
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos[:, :, :kv_seq_len], sin[:, :, :kv_seq_len])
        # [bsz, nh, t, hd]

        if past_key_value is not None:
            # reuse k, v, self_attention
            key_states = torch.cat([past_key_value[0], key_states], dim=2)
            value_states = torch.cat([past_key_value[1], value_states], dim=2)

        past_key_value = (key_states, value_states) if use_cache else None

        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

        if attn_weights.size() != (bsz, self.num_heads, q_len, kv_seq_len):
            raise ValueError(
                f"Attention weights should be of size {(bsz, self.num_heads, q_len, kv_seq_len)}, but is"
                f" {attn_weights.size()}"
            )

        if attention_mask is not None:
            if attention_mask.size() != (bsz, 1, q_len, kv_seq_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, q_len, kv_seq_len)}, but is {attention_mask.size()}"
                )
            attn_weights.masked_fill_(attention_mask, torch.finfo(attn_weights.dtype).min)

        # upcast attention to fp32
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_output = torch.matmul(attn_weights, value_states)

        if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.transpose(1, 2)
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)
        attn_output = self.o_proj(attn_output)
        if not output_attentions:
            attn_weights = None
        return attn_output, attn_weights, past_key_value


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin):
    q = q.unsqueeze(1)
    k = k.unsqueeze(1)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    q_embed = q_embed.squeeze(1)
    k_embed = k_embed.squeeze(1)
    return q_embed, k_embed


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=3):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=4):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc1 = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)


class cbam_block(nn.Module):
    def __init__(self, channel, ratio=4, kernel_size=3):
        super(cbam_block, self).__init__()
        self.channelattention = ChannelAttention(channel, ratio=ratio)
        self.spatialattention = SpatialAttention(kernel_size=kernel_size)

    def forward(self, x):
        x = x * self.channelattention(x)
        x = x * self.spatialattention(x)
        return x


class CNN(nn.Module):
    def __init__(self, input_size: int, F1: int, D: int = 2, final_dropout=0.25):
        super(CNN, self).__init__()
        self.drop_out = final_dropout
        self.att = cbam_block(D * F1)
        self.block_1 = nn.Sequential(
            # nn.ZeroPad2d((input_size//2, input_size//2, 0, 0)),
            nn.Conv2d(
                in_channels=1,
                out_channels=F1,
                kernel_size=(1, input_size),
                stride=(1, 1),
                bias=False,
                padding="same",
            ),
            nn.BatchNorm2d(F1),
            nn.ReLU(inplace=True),
            nn.AvgPool2d((1, 1)),
        )
        self.block_2 = nn.Sequential(
            # nn.ZeroPad2d((input_size//2, input_size//2, 0, 0)),
            nn.Conv2d(
                in_channels=F1,
                out_channels=D * F1,
                kernel_size=(1, 1),
                stride=(1, 1),
                bias=False,
                padding="same",
            ),
            Swish(),#GLU(dim=1),)
            nn.Conv2d(
                in_channels= D * F1,
                out_channels=D * F1,
                kernel_size=(1, input_size),
                stride=(1, 1),
                bias=False,
                groups=D * F1,
                padding="same",
            ),
            nn.BatchNorm2d(D * F1),
            Swish(),
            nn.Conv2d(
                in_channels=D * F1,
                out_channels=D * F1,
                kernel_size=(1, 1),
                stride=(1, 1),
                bias=False,
                padding="same",
            ),
            nn.BatchNorm2d(D * F1),
        )
        self.block_3 = nn.Sequential(
            nn.Conv2d(
                in_channels=D * F1,
                out_channels=D * D * F1,
                kernel_size=(1, 1),
                stride=(1, 1),
                groups=4,
                bias=False,
                padding="same",
            ),
            Swish(),#GLU(dim=1),)
            nn.Conv2d(
                in_channels=D * D * F1,
                out_channels=D * D * F1,
                kernel_size=(3, 1),
                stride=(1, 1),
                groups=D * D * F1,
                padding="same",
                bias=False,
            ),
            nn.BatchNorm2d(D * D * F1),
            Swish(),
            nn.Conv2d(
                in_channels=D * D * F1,
                out_channels=D * F1, # D * D * F1,
                kernel_size=(1, 1),
                stride=(1, 1),
                groups=4,
                bias=False,
                padding="same",
            ),
            nn.BatchNorm2d(D * F1), # D * D * F1
            # ChannelShuffle(D * D * F1, 4), # D * D * F1
        )
        self.block_4 = nn.Sequential(
            # nn.ZeroPad2d((input_size//4, input_size//4, 0, 0)),
            nn.Conv2d(
                in_channels= D * F1,
                out_channels=D * D * F1,
                kernel_size=(1, 1),
                stride=(1, 1),
                bias=False,
                padding="same",
            ),
            Swish(),#GLU(dim=1),)
            nn.Conv2d(
                in_channels= D * D * F1,
                out_channels=D * D * F1,
                kernel_size=(1, input_size),
                stride=(1, 1),
                bias=False,
                groups=D * D * F1,
                padding="same",
            ),
            nn.BatchNorm2d(D * D * F1),
            Swish(),
            nn.Conv2d(
                in_channels=D * D * F1,
                out_channels=D * F1,
                kernel_size=(1, 1),
                stride=(1, 1),
                bias=False,
                groups=4,
                padding="same",
            ),
            nn.BatchNorm2d(D * F1),
            nn.AvgPool2d((1, 1)),
            # nn.ReLU(inplace=True),
            # ChannelShuffle(D * D * F1, 4), # D * D * F1
        )

    def forward(self, x):
        # print(x.shape)
        x = self.block_1(x)
        # print(x.shape)
        x = self.block_2(x)
        # print(x.shape)
        x = self.att(x)
        # print(x.shape)
        x = self.block_3(x)
        # print(x.shape)
        x = self.block_4(x)
        # print(x.shape)
        # exit()
        return x


class ConvSqueezeFormerModel(BaseModel):
    def __init__(self, input_size, output_size, input_dim=128, D=2, encoder_dim=128, num_attention_heads=4,
                 num_layers=4, max_len=60, feed_forward_expansion_factor=2, conv_expansion_factor=2,
                 input_dropout_p=0.1, feed_forward_dropout_p=0.2, attention_dropout_p=0.2, conv_dropout_p=0.2,
                 conv_kernel_size=3, pool="gem", gem_p_trainable=True, reduce_layer_indices=[],
                 ignore_mask=None, delta=1.0):
        super(ConvSqueezeFormerModel, self).__init__(ignore_mask=ignore_mask, delta=delta)
        self.n_classes = 368
        in_channels = 25

        D = 2
        F1 = input_dim // 2
        self.cnn = CNN(in_channels, F1=F1, D=D, final_dropout=0.25)

        rotary_emb = LlamaRotaryEmbedding(encoder_dim//num_attention_heads, max_position_embeddings=max_len)
        self.cos = nn.Parameter(rotary_emb.cos_cached, requires_grad=False)#[:, :, :seq_len, ...]#.to(dtype=x.dtype)
        self.sin = nn.Parameter(rotary_emb.sin_cached, requires_grad=False)#[:, :, :seq_len, ...]#.to(dtype=x.dtype)

        while len(self.cos.shape)<4: self.cos = nn.Parameter(self.cos.unsqueeze(0))
        while len(self.sin.shape)<4: self.sin = nn.Parameter(self.sin.unsqueeze(0))

        self.encoder = SqueezeformerEncoder(
            input_dim=input_dim,
            reduce_layer_indices=reduce_layer_indices,
            encoder_dim=encoder_dim,
            num_layers=num_layers,
            num_attention_heads=num_attention_heads,
            feed_forward_expansion_factor=feed_forward_expansion_factor,
            conv_expansion_factor= conv_expansion_factor,
            input_dropout_p=input_dropout_p,
            feed_forward_dropout_p= feed_forward_dropout_p,
            attention_dropout_p=attention_dropout_p,
            conv_dropout_p=conv_dropout_p,
            conv_kernel_size=conv_kernel_size,
        )

        if pool == "gem":
            self.global_pool = GeM(p_trainable=gem_p_trainable)
        elif pool == "identity":
            self.global_pool = torch.nn.Identity()
        elif pool == "avg":
            self.global_pool = nn.AdaptiveAvgPool2d(1)

        self.head = torch.nn.Linear(encoder_dim, self.n_classes)

    def forward(self, batch):
        v = batch["x_vector"]  # (bs, num_features, seq_len)
        s = batch["x_scalar"].unsqueeze(dim=-1).repeat(1, 1, v.size(-1))  # (bs, num_features, seq_len)
        x = torch.cat([v, s], dim=1)
        x = x.permute(0, 2, 1).unsqueeze(-1)  # (bs, seq_len, num_features, 1)
        mask = torch.ones_like(x[:, :, 0, 0], dtype=torch.long)  # (bs, seq_len)

        x = x.permute(0, 3, 2, 1)  # (bs, 1, num_features, seq_len)
        x = self.cnn(x)

        x = x.permute(0, 1, 3, 2)  # (bs, ch_size, seq_len, num_features)
        bs, slen, hf, wf = x.shape
        # print(x.shape)
        x = x.reshape(bs, slen * hf, wf)
        # print(x.shape)
        x = x.unsqueeze(-1)

        x = self.global_pool(x)[:,:,0,0]
        # print(x.shape)
        x = x.reshape(bs, slen, hf)
        x = x.permute(0, 2, 1)
        # print(x.shape)

        # Input torch.Size([8, 625, 128])
        x, mask = self.encoder(x, self.cos, self.sin, mask)

        # print(x.shape)
        x = x.unsqueeze(1).permute(0, 3, 1, 2)
        # print(x.shape)
        x = self.global_pool(x)
        # print(x.shape)
        # exit()
        x = x[:, :, 0, 0]
        logits = self.head(x)
        return {
            "logits": logits
        }

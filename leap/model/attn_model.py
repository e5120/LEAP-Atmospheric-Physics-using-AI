import torch
import torch.nn as nn
import torch.nn.functional as F

from leap.model import BaseModel
from leap.model.modules import MLPBlock, PositionalEncoding
# from leap.model.light_cnn_model_v2 import Conv1dBlock


class AttnModel(BaseModel):
    def __init__(self, input_size, output_size, num_scalar_feats, num_vector_feats,
                 d_model=512, nhead=8, dim_feedforward=2048, num_layers=6, hidden_sizes=[],
                 p=0.1, use_positional_embedding=True, pos_type="sinusoid", ignore_mask=None, use_aux=False):
        super().__init__(ignore_mask=ignore_mask, use_aux=use_aux)
        self.embedding = MLPBlock(num_scalar_feats+num_vector_feats, d_model)
        self.use_positional_embedding = use_positional_embedding
        if self.use_positional_embedding:
            self.positional_embedding = PositionalEncoding(d_model, max_len=60, dropout=p, pos_type=pos_type)
        encoder = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=p,
            batch_first=True,
            norm_first=True,
            activation=nn.GELU(),
        )
        self.encoder = nn.TransformerEncoder(encoder, num_layers=num_layers)
        # self.conv_blocks = nn.Sequential(*[
        #     Conv1dBlock(d_model, d_model, activation="swish")
        #     for _ in range(3)
        # ])
        self.decoder = MLPBlock(d_model, 14, hidden_sizes=hidden_sizes)
        # self.decoder = nn.Conv1d(d_model, 14, kernel_size=1, padding="same")

    def forward(self, batch):
        bs = batch["x_scalar"].size(0)
        v = batch["x_vector"]                                             # (bs,  9, 60)
        s = batch["x_scalar"].unsqueeze(dim=-1).repeat(1, 1, v.size(-1))  # (bs, 16, 60)
        x = torch.cat([v, s], dim=1)                                      # (bs, 25, 60)
        x = x.permute(0, 2, 1)
        emb = self.embedding(x)
        if self.use_positional_embedding:
            emb = self.positional_embedding(emb)
        out = self.encoder(emb)
        out = self.decoder(out)
        out = out.permute(0, 2, 1)
        # out = self.conv_blocks(out)
        vector_out = out[:, :6].reshape(bs, -1)
        scalar_out = F.avg_pool1d(out[:, 6:], kernel_size=out.size(2)).squeeze(-1)
        logits = torch.cat([scalar_out, vector_out], dim=1)
        return {
            "logits": logits,
        }

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from leap.model import BaseModel
from leap.model.modules import MLPBlock


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=60, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


class AttnModel(BaseModel):
    def __init__(self, input_size, output_size, num_scalar_feats, num_vector_feats, d_model=512, nhead=8, dim_feedforward=2048, num_layers=6, hidden_sizes=[256,512,512,256], p=0.1, pos_type="embedding", ignore_mask=None):
        super().__init__(ignore_mask=ignore_mask)
        assert pos_type in ["embedding", "sinusoid"]
        self.scalar_embedding = MLPBlock(num_scalar_feats, d_model, hidden_sizes=[], p=p)
        self.vector_embedding = MLPBlock(num_vector_feats, d_model, hidden_sizes=[], p=p)
        self.pos_type = pos_type
        if pos_type == "embedding":
            self.positional_embedding = nn.Embedding(60, d_model)
        else:
            self.positional_embedding = PositionalEncoding(d_model)
        encoder = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            batch_first=True,
            norm_first=True,
            activation=nn.GELU(),
        )
        self.encoder = nn.TransformerEncoder(encoder, num_layers=num_layers)
        self.decoder = MLPBlock(d_model, 14, hidden_sizes=hidden_sizes)

    def forward(self, batch):
        bs = batch["x_scalar"].size(0)
        scalar_emb = self.scalar_embedding(batch["x_scalar"]).unsqueeze(dim=1)  # (bs, 1, d_model)
        vector_emb = self.vector_embedding(batch["x_vector"].permute(0, 2, 1))  # (bs, 60, d_model)
        if self.pos_type == "embedding":
            pos_emb = self.positional_embedding(torch.arange(60).to(batch["x_scalar"].device)).unsqueeze(dim=0)   # (1, 60, d_model)
            emb = scalar_emb + vector_emb + pos_emb
        else:
            emb = self.positional_embedding(scalar_emb + vector_emb)
        out = self.encoder(emb)
        out = self.decoder(out)
        out = out.permute(0, 2, 1)
        vector_out = out[:, :6].reshape(bs, -1)
        scalar_out = F.avg_pool1d(out[:, 6:], kernel_size=out.size(2)).squeeze(-1)
        logits = torch.cat([scalar_out, vector_out], dim=1)
        return {
            "logits": logits,
        }

import torch
import torch.nn as nn
import torch.nn.functional as F

from leap.model import BaseModel
from leap.model.modules import MLPBlock


class AttnModel(BaseModel):
    def __init__(self, input_size, output_size, num_scalar_feats, num_vector_feats, d_model=512, nhead=8, dim_feedforward=2048, num_layers=6, hidden_sizes=[256,512,512,256], p=0.1, ignore_mask=None):
        super().__init__(ignore_mask=ignore_mask)
        self.scalar_embedding = MLPBlock(num_scalar_feats, d_model, hidden_sizes=[], p=p)
        self.vector_embedding = MLPBlock(num_vector_feats, d_model, hidden_sizes=[], p=p)
        self.positional_embedding = nn.Embedding(60, d_model)
        encoder = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder, num_layers=num_layers)
        self.decoder = MLPBlock(d_model, 7, hidden_sizes=hidden_sizes)
        self.scalar_decoder = MLPBlock(60, 8)

    def forward(self, batch):
        bs = batch["x_scalar"].size(0)
        scalar_emb = self.scalar_embedding(batch["x_scalar"]).unsqueeze(dim=1)  # (bs, 1, d_model)
        vector_emb = self.vector_embedding(batch["x_vector"].permute(0, 2, 1))  # (bs, 60, d_model)
        pos_emb = self.positional_embedding(torch.arange(60).to(batch["x_scalar"].device)).unsqueeze(dim=0)   # (1, 60, d_model)
        emb = scalar_emb + vector_emb + pos_emb
        out = self.encoder(emb)
        out = self.decoder(out)
        vector_out = out[:, :, :-1].permute(0, 2, 1).reshape(bs, -1)
        scalar_out = out[:, :, -1]
        scalar_out = self.scalar_decoder(scalar_out)
        logits = torch.cat([scalar_out, vector_out], dim=1)
        return {
            "logits": logits,
        }

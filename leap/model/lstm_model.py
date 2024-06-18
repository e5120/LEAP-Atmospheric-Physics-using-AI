import torch
import torch.nn as nn
import torch.nn.functional as F

from leap.model import BaseModel


class LSTMModel(BaseModel):
    def __init__(self, input_size, output_size, hidden_dim, num_scalar_feats, num_vector_feats, n_layers=1, bidirectional=True, p=0.0, ignore_mask=None):
        super().__init__(ignore_mask=ignore_mask)
        self.hidden_dim = hidden_dim
        scalar_hidden_dim = 4*n_layers*hidden_dim if bidirectional else 2*n_layers*hidden_dim
        self.scalar_fc = nn.Sequential(
            nn.Linear(num_scalar_feats, scalar_hidden_dim),
            nn.LayerNorm(scalar_hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=p),
        )
        self.vector_fc = nn.Sequential(
            nn.Linear(num_vector_feats, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=p),
        )
        self.lstm_layers = nn.LSTM(
            hidden_dim, hidden_dim,
            num_layers=n_layers,
            bidirectional=bidirectional,
            dropout=p,
            batch_first=True,
        )
        out_features = 2*hidden_dim if bidirectional else hidden_dim
        self.fc = nn.Linear(out_features, output_size)

    def forward(self, batch):
        bs = batch["x_scalar"].size(0)
        scalar = self.scalar_fc(batch["x_scalar"])
        h_0, c_0 = scalar.chunk(2, dim=1)
        h_0 = h_0.view(bs, -1, self.hidden_dim).permute(1, 0, 2).contiguous()
        c_0 = c_0.view(bs, -1, self.hidden_dim).permute(1, 0, 2).contiguous()
        vec = batch["x_vector"].permute(0, 2, 1)
        vec = self.vector_fc(vec)
        out, _ = self.lstm_layers(vec, (h_0, c_0))
        out = out.permute(0, 2, 1)
        out = F.avg_pool1d(out, kernel_size=out.size(2)).squeeze(dim=-1)
        logits = self.fc(out)
        return {
            "logits": logits,
        }

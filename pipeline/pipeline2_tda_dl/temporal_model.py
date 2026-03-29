from __future__ import annotations

import math
from typing import Optional

import torch
from torch import nn


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, seq_len: int):
        super().__init__()
        self.embedding = nn.Embedding(seq_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, d = x.shape
        positions = torch.arange(t, device=x.device).unsqueeze(0).expand(b, t)
        pos = self.embedding(positions)
        return x + pos


class TemporalClassifier(nn.Module):
    def __init__(self, latent_dim: int, num_classes: int, seq_len: int, d_model: int = 128):
        super().__init__()
        self.input_proj = nn.Linear(latent_dim, d_model)
        self.pos_enc = PositionalEncoding(d_model, seq_len)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=4,
            dim_feedforward=256,
            dropout=0.1,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.head = nn.Sequential(
            nn.Linear(d_model, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, num_classes),
        )

    def forward(self, seq: torch.Tensor) -> torch.Tensor:
        # seq: [B, T, latent_dim]
        x = self.input_proj(seq)
        x = self.pos_enc(x)
        x = self.encoder(x)  # [B, T, D]
        x = x.mean(dim=1)
        return self.head(x)

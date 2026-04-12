from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [seq_len, batch_size, embedding_dim]
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)

class TemporalTransformer(nn.Module):
    def __init__(
        self, 
        input_dim: int, 
        num_classes: int = 11,  # 10 acciones + 1 fondo (background)
        d_model: int = 128, 
        nhead: int = 4, 
        num_layers: int = 2, 
        dim_feedforward: int = 512, 
        dropout: float = 0.1
    ):
        super().__init__()
        self.model_type = 'Transformer'
        
        # Proyección de entrada para ajustar dimensiones al d_model del Transformer
        self.input_projection = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        
        encoder_layers = nn.TransformerEncoderLayer(
            d_model, nhead, dim_feedforward, dropout, batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)
        
        self.d_model = d_model
        
        # CAPA MODIFICADA: Ahora la salida es para cada paso temporal
        self.decoder = nn.Linear(d_model, num_classes)

    def forward(self, src: torch.Tensor) -> torch.Tensor:
        """
        Args:
            src: Tensor de entrada de forma [batch_size, seq_len, input_dim]
        Returns:
            Tensor de salida de forma [batch_size, seq_len, num_classes]
        """
        # 1. Proyectar dimensiones de entrada
        src = self.input_projection(src) * math.sqrt(self.d_model)
        
        # 2. Codificación posicional
        # TransformerEncoder con batch_first=True espera [batch_size, seq_len, d_model]
        src = self.pos_encoder(src.transpose(0, 1)).transpose(0, 1)
        
        # 3. Pasar por las capas del Transformer
        output = self.transformer_encoder(src)
        
        # --- CAMBIO CLAVE PARA SEGMENTACIÓN ---
        # Antes: output = output.mean(dim=1) (Global Average Pooling)
        # Ahora: Pasamos cada paso temporal por la capa lineal
        
        logits = self.decoder(output)
        
        # Resultado: [batch_size, seq_len, num_classes]
        return logits
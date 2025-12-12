import torch
import torch.nn as nn
import numpy as np

class transformer(nn.Module):
    def __init__(self, input_dynamic_channels, input_static_channels, output_channels, seq_len, hidden_dim, nhead, num_layers, dropout):
        super().__init__()
        self.embedding = nn.Linear(input_dynamic_channels + input_static_channels, hidden_dim)
        
        # Positional encoding
        pe = torch.zeros(seq_len, hidden_dim)
        position = torch.arange(0, seq_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, hidden_dim, 2).float() * (-np.log(10000.0) / hidden_dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=nhead,
            dim_feedforward=hidden_dim * 2,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(hidden_dim, output_channels)
        
    def forward(self, x_dynamic, x_static):
        x = torch.cat((x_dynamic, x_static), dim=-1)
        x = self.embedding(x)  # (batch, seq_len, hidden_dim)
        x = x + self.pe  # add positional encoding
        x = self.transformer(x)
        x = self.fc(x)
        return x
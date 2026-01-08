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
            dim_feedforward=hidden_dim * 4,  
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(hidden_dim, output_channels)
        
    def forward(self, x_dynamic, x_static):
        x = torch.cat((x_dynamic, x_static), dim=-1)
        x = self.embedding(x)  
        x = x + self.pe  
        x = self.transformer(x)
        x = self.fc(x)
        return x
    
class patch_transformer(nn.Module):
    def __init__(self, input_dynamic_channels: int, input_static_channels: int, output_channels: int,
        seq_len: int, pred_len: int, patch_len: int, stride: int, hidden_dim: int = 128,
        nhead: int = 4, num_layers: int = 3, dropout: float = 0.1, dyn_pool: str = "mean"):
        super().__init__()
        
        if dyn_pool not in ("mean",):
            raise ValueError(f"dyn_pool currently supports only 'mean', got {dyn_pool}")

        self.seq_len = int(seq_len)
        self.pred_len = int(pred_len)
        self.patch_len = int(patch_len)
        self.stride = int(stride)
        self.in_dyn = int(input_dynamic_channels)
        self.in_stat = int(input_static_channels)
        self.out_ch = int(output_channels)
        self.hidden_dim = int(hidden_dim)
        self.dyn_pool = dyn_pool
        self.nhead = nhead
        
        if self.seq_len < self.patch_len:
            raise ValueError(f"seq_len={seq_len} must be >= patch_len={patch_len}")

        # number of patches along time
        self.num_patches = int((self.seq_len - self.patch_len) / self.stride) + 1
        if self.num_patches <= 0:
            raise ValueError("num_patches computed <= 0; check seq_len/patch_len/stride")

        # patch embedding per channel (patch_len -> hidden_dim)
        self.patch_embedding = nn.Linear(self.patch_len, self.hidden_dim)

        # learnable positional embedding (1, num_patches, hidden_dim)
        self.pos_embed = nn.Parameter(torch.randn(1, self.num_patches, self.hidden_dim) * 0.02)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=self.hidden_dim,  
            nhead=nhead,
            dim_feedforward=self.hidden_dim * 4,  
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)

        # Static encoder
        self.static_encoder = nn.Sequential(
            nn.Linear(self.in_stat, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # x_enc: (B, C, num_patches, hidden_dim)
        # pool patches -> (B, C, hidden_dim) then flatten -> (B, C*hidden_dim)
        head_in = (self.in_dyn * self.hidden_dim) + self.hidden_dim

        self.head = nn.Sequential(
            nn.LayerNorm(head_in),
            nn.Linear(head_in, self.hidden_dim * 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(self.hidden_dim * 4, self.pred_len * self.out_ch),
        )

    def forward(self, x_dynamic: torch.Tensor, x_static: torch.Tensor) -> torch.Tensor:
        """
        x_dynamic: (B, seq_len, in_dyn)
        x_static : (B, seq_len, in_stat) or (B, in_stat)
        return   : (B, pred_len, out_ch)  (scaled space)
        """
        if x_dynamic.dim() != 3:
            raise ValueError(f"x_dynamic must be 3D (B,T,C), got shape {tuple(x_dynamic.shape)}")

        B, T, C = x_dynamic.shape
        if T != self.seq_len:
            raise ValueError(f"Expected seq_len={self.seq_len}, got T={T}")
        if C != self.in_dyn:
            raise ValueError(f"Expected in_dyn={self.in_dyn}, got C={C}")

        # patching: (B, T, C) -> (B, C, T)
        x = x_dynamic.permute(0, 2, 1)  # (B, C, T)

        # unfold into patches: (B, C, num_patches, patch_len)
        x_p = x.unfold(dimension=-1, size=self.patch_len, step=self.stride)

        # patch embedding: (B, C, num_patches, hidden_dim)
        x_e = self.patch_embedding(x_p)
        x_e = x_e + self.pos_embed.unsqueeze(1)  # (1,1,num_patches,hidden_dim) broadcast

        # channel-independent encoding: reshape to (B*C, num_patches, hidden_dim)
        x_in = x_e.reshape(B * C, self.num_patches, self.hidden_dim)
        x_enc = self.encoder(x_in)  # (B*C, num_patches, hidden_dim)
        x_enc = x_enc.reshape(B, C, self.num_patches, self.hidden_dim)

        # pool over patches -> (B, C, hidden_dim)
        x_chan = x_enc.mean(dim=2)

        # flatten channels -> (B, C*hidden_dim)
        x_dyn_flat = x_chan.flatten(start_dim=1)

        # static -> (B, hidden_dim)
        x_stat_emb = self.static_encoder(x_static[:, 0, :])
        
        # head
        z = torch.cat([x_dyn_flat, x_stat_emb], dim=1)  # (B, head_in)
        out = self.head(z)                              # (B, pred_len*out_ch)
        out = out.view(B, self.pred_len, self.out_ch)
        return out

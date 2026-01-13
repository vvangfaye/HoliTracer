# vlr_cls.py

import torch
import torch.nn as nn
import math


class PositionalEncoding(nn.Module):
    def __init__(self, embed_dim, max_len=1024):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, embed_dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, embed_dim, 2).float() * (-math.log(10000.0) / embed_dim)
        )
        pe[:, 0::2] = torch.sin(position * div_term)  # Even dimensions
        pe[:, 1::2] = torch.cos(position * div_term)  # Odd dimensions
        pe = pe.unsqueeze(0)  # (1, max_len, embed_dim)
        self.register_buffer("pe", pe)

    def forward(self, x):
        """
        x: (B, N, E)
        """
        x = x + self.pe[:, : x.size(1), :].to(x.device)
        return x


class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, dim_feedforward=512, dropout=0.1):
        super(TransformerBlock, self).__init__()
        self.self_attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout)
        self.linear1 = nn.Linear(embed_dim, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, embed_dim)

        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = nn.ReLU(inplace=True)

    def forward(self, x):
        """
        x: (B, N, E)
        """
        B, N, E = x.size()

        # Convert to (N, B, E) to adapt to nn.MultiheadAttention
        x_transposed = x.permute(1, 0, 2)  # (N, B, E)

        # Self-attention
        attn_output, _ = self.self_attn(
            x_transposed, x_transposed, x_transposed
        )  # (N, B, E)
        attn_output = attn_output.permute(1, 0, 2)  # (B, N, E)

        # Residual connection + Normalization
        x = self.norm1(x + self.dropout1(attn_output))  # (B, N, E)

        # Feedforward network
        ff_output = self.linear2(
            self.dropout(self.activation(self.linear1(x)))
        )  # (B, N, E)

        # Residual connection + Normalization
        x = self.norm2(x + self.dropout2(ff_output))  # (B, N, E)

        return x  # (B, N, E)


class VLRCls(nn.Module):
    def __init__(
        self,
        state_dim,
        feature_dim,
        num_layers=3,
        num_heads=8,
        dim_feedforward=512,
        dropout=0.1,
        num_points=64,
    ):
        super(VLRCls, self).__init__()
        self.embed_dim = state_dim
        self.num_points = num_points
        # embedding layer
        self.input_linear = nn.Linear(feature_dim, self.embed_dim)  # Input feature dimension is 3
        # Positional encoding
        self.positional_encoding = PositionalEncoding(
            self.embed_dim, max_len=num_points
        )

        # Custom Transformer encoder block
        self.transformer_blocks = nn.ModuleList(
            [
                TransformerBlock(self.embed_dim, num_heads, dim_feedforward, dropout)
                for _ in range(num_layers)
            ]
        )

        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(self.embed_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, 1),  # Output a logit representing binary classification
        )

    def forward(self, x, valid_mask=None):
        """
        x: (B, E, N)
        valid_mask: (B, N) Optional, 1 indicates valid, 0 indicates invalid
        """
        x = x.permute(0, 2, 1)  # (B, N, E)
        B, N, E = x.size()
        # Input header
        x = self.input_linear(x)
        # Add positional encoding
        x = self.positional_encoding(x)  # (B, N, E)

        # Pass through all custom Transformer encoder blocks
        for block in self.transformer_blocks:
            x = block(x)  # (B, N, E)

        # Classification head
        logits = self.classifier(x).squeeze(-1)  # (B, N)

        return logits  # Binary classification logits for each point
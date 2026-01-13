import torch
import torch.nn as nn
import torch.nn.functional as F
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
        x: (B, NUM_POINTS, EMBED_DIM)
        """
        x = x + self.pe[:, : x.size(1), :]
        return x


class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, dim_feedforward=2048, dropout=0.1):
        super(TransformerBlock, self).__init__()
        self.self_attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout)
        self.linear1 = nn.Linear(embed_dim, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, embed_dim)

        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = F.relu

    def forward(self, x):
        """
        x: (B, NUM_POINTS, EMBED_DIM)
        """
        # Convert to (NUM_POINTS, B, EMBED_DIM) to adapt to nn.MultiheadAttention
        x = x.permute(1, 0, 2)
        # Self-attention
        attn_output, _ = self.self_attn(x, x, x)
        attn_output = attn_output.permute(1, 0, 2)  # (B, NUM_POINTS, EMBED_DIM)
        # Residual connection + Normalization
        x = self.norm1(x.permute(1, 0, 2) + self.dropout1(attn_output))
        # Feedforward network
        ff_output = self.linear2(self.dropout(self.activation(self.linear1(x))))
        # Residual connection + Normalization
        x = self.norm2(x + self.dropout2(ff_output))
        return x  # (B, NUM_POINTS, EMBED_DIM)


class FusionLayer(nn.Module):
    def __init__(self, embed_dim, fusion_dim):
        super(FusionLayer, self).__init__()
        self.fusion_linear = nn.Linear(embed_dim * 2, fusion_dim)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, states):
        """
        states: list of tensors, each of shape (B, NUM_POINTS, EMBED_DIM)
        """
        # Concatenate all states
        state = torch.cat(states, dim=2)  # (B, NUM_POINTS, EMBED_DIM * num_layers)
        # Global max pooling to extract global features
        global_feat, _ = torch.max(
            state, dim=1, keepdim=True
        )  # (B, 1, EMBED_DIM * num_layers)
        global_feat = global_feat.expand(
            -1, state.size(1), -1
        )  # (B, NUM_POINTS, EMBED_DIM * num_layers)
        # Concatenate global features and local features
        fused = torch.cat(
            [state, global_feat], dim=2
        )  # (B, NUM_POINTS, EMBED_DIM * num_layers * 2)
        # Fuse features through linear layer
        fused = self.fusion_linear(fused)  # (B, NUM_POINTS, fusion_dim)
        fused = self.relu(fused)
        return fused  # (B, NUM_POINTS, fusion_dim)


class PredictionLayer(nn.Module):
    def __init__(self, input_dim, hidden_dim=256, output_dim=2):
        super(PredictionLayer, self).__init__()
        self.prediction = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, output_dim),
        )

    def forward(self, x):
        """
        x: (B, NUM_POINTS, input_dim)
        """
        x = self.prediction(x)  # (B, NUM_POINTS, output_dim)
        return x


class VLROffset(nn.Module):
    def __init__(
        self,
        state_dim,
        feature_dim,
        num_layers=7,
        num_heads=8,
        fusion_dim=256,
        dim_feedforward=2048,
        dropout=0.1,
        num_points=64,
    ):
        super(VLROffset, self).__init__()
        self.embed_dim = state_dim
        self.num_layers = num_layers

        # Input header: convert input features to state dimension
        self.input_linear = nn.Linear(feature_dim, self.embed_dim)
        self.positional_encoding = PositionalEncoding(
            self.embed_dim, max_len=num_points
        )

        # Transformer blocks
        self.transformer_blocks = nn.ModuleList(
            [
                TransformerBlock(self.embed_dim, num_heads, dim_feedforward, dropout)
                for _ in range(num_layers)
            ]
        )

        # Fusion layer
        self.fusion = FusionLayer(self.embed_dim * (num_layers + 1), fusion_dim)

        # Prediction layer
        self.prediction = PredictionLayer(fusion_dim, hidden_dim=256, output_dim=2)

    def forward(self, x):
        """
        x: (B, NUM_FEATURES, NUM_POINTS)
        """
        # Convert to (B, NUM_POINTS, NUM_FEATURES)
        x = x.permute(0, 2, 1)  # (B, NUM_POINTS, NUM_FEATURES)
        # Input header
        x = self.input_linear(x)  # (B, NUM_POINTS, EMBED_DIM)
        x = self.positional_encoding(x)  # Add positional encoding

        states = [x]

        # Pass through all Transformer blocks
        for block in self.transformer_blocks:
            x = block(x)  # (B, NUM_POINTS, EMBED_DIM)
            states.append(x)

        # Fusion layer
        fused = self.fusion(states)  # (B, NUM_POINTS, fusion_dim)

        # Prediction layer
        prediction = self.prediction(fused)  # (B, NUM_POINTS, 2)

        return prediction.permute(0, 2, 1)  # (B, 2, NUM_POINTS)


if __name__ == "__main__":
    # Example input
    batch_size = 8
    num_features = 128  # Input feature dimension
    num_points = 64
    feature_dim = num_features
    state_dim = 256  # Same as embed_dim
    num_layers = 7
    num_heads = 8
    fusion_dim = 256
    dim_feedforward = 2048
    dropout = 0.1
    max_points = 1024

    # Randomly generate input data
    x = torch.randn(
        batch_size, num_features, num_points
    )  # (B, NUM_FEATURES, NUM_POINTS)

    # Initialize model
    model = VLROffset(
        state_dim=state_dim,
        feature_dim=feature_dim,
        num_layers=num_layers,
        num_heads=num_heads,
        fusion_dim=fusion_dim,
        dim_feedforward=dim_feedforward,
        dropout=dropout,
        num_points=max_points,
    )

    # Forward propagation
    output = model(x)  # (B, NUM_POINTS, 2)

    print("Output shape:", output.shape)  # Should be (B, NUM_POINTS, 2)
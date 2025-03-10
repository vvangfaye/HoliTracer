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
        pe[:, 0::2] = torch.sin(position * div_term)  # 偶数维度
        pe[:, 1::2] = torch.cos(position * div_term)  # 奇数维度
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
        # 转换为 (NUM_POINTS, B, EMBED_DIM) 以适配 nn.MultiheadAttention
        x = x.permute(1, 0, 2)
        # 自注意力
        attn_output, _ = self.self_attn(x, x, x)
        attn_output = attn_output.permute(1, 0, 2)  # (B, NUM_POINTS, EMBED_DIM)
        # 残差连接 + 归一化
        x = self.norm1(x.permute(1, 0, 2) + self.dropout1(attn_output))
        # 前馈网络
        ff_output = self.linear2(self.dropout(self.activation(self.linear1(x))))
        # 残差连接 + 归一化
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
        # 拼接所有状态
        state = torch.cat(states, dim=2)  # (B, NUM_POINTS, EMBED_DIM * num_layers)
        # 全局最大池化提取全局特征
        global_feat, _ = torch.max(
            state, dim=1, keepdim=True
        )  # (B, 1, EMBED_DIM * num_layers)
        global_feat = global_feat.expand(
            -1, state.size(1), -1
        )  # (B, NUM_POINTS, EMBED_DIM * num_layers)
        # 拼接全局特征和局部特征
        fused = torch.cat(
            [state, global_feat], dim=2
        )  # (B, NUM_POINTS, EMBED_DIM * num_layers * 2)
        # 通过线性层融合特征
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

        # 输入头部：将输入特征转换为状态维度
        self.input_linear = nn.Linear(feature_dim, self.embed_dim)
        self.positional_encoding = PositionalEncoding(
            self.embed_dim, max_len=num_points
        )

        # Transformer 块
        self.transformer_blocks = nn.ModuleList(
            [
                TransformerBlock(self.embed_dim, num_heads, dim_feedforward, dropout)
                for _ in range(num_layers)
            ]
        )

        # 融合层
        self.fusion = FusionLayer(self.embed_dim * (num_layers + 1), fusion_dim)

        # 预测层
        self.prediction = PredictionLayer(fusion_dim, hidden_dim=256, output_dim=2)

    def forward(self, x):
        """
        x: (B, NUM_FEATURES, NUM_POINTS)
        """
        # 转换为 (B, NUM_POINTS, NUM_FEATURES)
        x = x.permute(0, 2, 1)  # (B, NUM_POINTS, NUM_FEATURES)
        # 输入头部
        x = self.input_linear(x)  # (B, NUM_POINTS, EMBED_DIM)
        x = self.positional_encoding(x)  # 添加位置编码

        states = [x]

        # 通过所有 Transformer 块
        for block in self.transformer_blocks:
            x = block(x)  # (B, NUM_POINTS, EMBED_DIM)
            states.append(x)

        # 融合层
        fused = self.fusion(states)  # (B, NUM_POINTS, fusion_dim)

        # 预测层
        prediction = self.prediction(fused)  # (B, NUM_POINTS, 2)

        return prediction.permute(0, 2, 1)  # (B, 2, NUM_POINTS)


if __name__ == "__main__":
    # 示例输入
    batch_size = 8
    num_features = 128  # 输入特征维度
    num_points = 64
    feature_dim = num_features
    state_dim = 256  # 与 embed_dim 相同
    num_layers = 7
    num_heads = 8
    fusion_dim = 256
    dim_feedforward = 2048
    dropout = 0.1
    max_points = 1024

    # 随机生成输入数据
    x = torch.randn(
        batch_size, num_features, num_points
    )  # (B, NUM_FEATURES, NUM_POINTS)

    # 初始化模型
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

    # 前向传播
    output = model(x)  # (B, NUM_POINTS, 2)

    print("Output shape:", output.shape)  # 应为 (B, NUM_POINTS, 2)

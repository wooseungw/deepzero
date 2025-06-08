# ──────────────────────────────────────────────────────────────────────────────
# Dezero 기반으로 ConvBlock, TransformerEncoderBlock, GlobalAvgPool, LinearBlock 등
# 블록들을 정의하고, 이름 → 클래스 매핑을 위한 레지스트리도 함께 작성
# ──────────────────────────────────────────────────────────────────────────────

import numpy as np
from flexo import Model
import flexo.functions as F
import flexo.layers as L

class ConvBlock(Model):
    """
    Conv → Norm → Activation → Dropout 구성 블록
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=None, act='silu', norm=None, drop=0.0):
        super().__init__()
        # padding 자동 계산 (same padding)
        if padding is None:
            padding = kernel_size // 2

        # 1) Convolution
        self.conv = L.Conv2d(in_channels, out_channels,
                              kernel_size=kernel_size, stride=stride, pad=padding)

        # 2) Normalization (batchnorm / layernorm 선택)
        if norm == 'bn':
            self.norm = L.BatchNormalization()
        elif norm == 'ln':
            # Dezero의 LayerNorm 사용 시, (C, 1, 1) 형태 지정
            self.norm = L.LayerNormalization((out_channels, 1, 1))
        else:
            self.norm = None

        # 3) Activation (relu, silu 등)
        if act == 'relu':
            self.act = F.relu
        elif act == 'silu':
            self.act = F.silu
        elif act == 'sigmoid':
            self.act = F.sigmoid
        else:
            self.act = None

        # 4) Dropout
        self.drop = drop
        if drop and drop > 0.0:
            self.dropout = L.Dropout(drop)
        else:
            self.dropout = None

    def forward(self, x):
        h = self.conv(x)
        if self.norm is not None:
            h = self.norm(h)
        if self.act is not None:
            h = self.act(h)
        if self.dropout is not None:
            h = self.dropout(h)
        return h


class GlobalAvgPool(Model):
    """
    (B, C, H, W) → (B, C) 로 평균 풀링
    """
    def forward(self, x):
        # Dezero의 mean 사용 (height, width 차원 평균)
        return F.mean(x, axis=(2, 3))


class LinearBlock(Model):
    """
    Linear → Activation → Dropout 구성 블록
    """
    def __init__(self, in_features, out_features, act=None, drop=0.0):
        super().__init__()
        self.linear = L.Linear(in_features, out_features)

        if act == 'relu':
            self.act = F.relu
        elif act == 'silu':
            self.act = F.silu
        else:
            self.act = None

        self.drop = drop
        if drop and drop > 0.0:
            self.dropout = L.Dropout(drop)
        else:
            self.dropout = None

    def forward(self, x):
        h = self.linear(x)
        if self.act is not None:
            h = self.act(h)
        if self.dropout is not None:
            h = self.dropout(h)
        return h


class TransformerEncoderBlock(Model):
    """
    단순화된 Transformer Encoder 블록 예시.
    - embed_dim: 입력 임베딩 차원
    - num_heads: 멀티헤드 어텐션 헤드 수
    - hidden_dim: Feed-Forward 중간차원
    - drop: Dropout 비율
    """
    def __init__(self, embed_dim, num_heads, hidden_dim, drop=0.0):
        super().__init__()
        # 1) LayerNorm + MultiHeadSelfAttention
        self.norm1 = L.LayerNormalization((embed_dim,))
        # Dezero 공식 패키지에 MultiHeadSelfAttention이 있다면 가져오시면 됩니다.
        # 예시에서는 MultiHeadSelfAttention이 있다고 가정
        self.attn = L.MultiHeadSelfAttention(embed_dim, num_heads, dropout=drop)

        # 2) Feed-Forward (Linear → ReLU → Linear)
        self.norm2 = L.LayerNormalization((embed_dim,))
        self.ffn = L.Sequential(
            L.Linear(embed_dim, hidden_dim),
            F.relu,
            L.Linear(hidden_dim, embed_dim)
        )

        self.drop = drop
        if drop and drop > 0.0:
            self.dropout = L.Dropout(drop)
        else:
            self.dropout = None

    def forward(self, x):
        # x: (batch_size, seq_len, embed_dim)
        # 1) Self-Attention 블록
        h1 = self.norm1(x)
        h1 = self.attn(h1, h1, h1)
        if self.dropout is not None:
            h1 = self.dropout(h1)
        x = x + h1  # residual

        # 2) Feed-Forward 블록
        h2 = self.norm2(x)
        h2 = self.ffn(h2)
        if self.dropout is not None:
            h2 = self.dropout(h2)
        return x + h2  # residual


# ──────────────────────────────────────────────────────────────────────────────
# 블록 레지스트리: YAML의 block_type 문자열 → 실제 클래스 매핑
# ──────────────────────────────────────────────────────────────────────────────

BLOCK_REGISTRY = {
    'ConvBlock': ConvBlock,
    'GlobalAvgPool': GlobalAvgPool,
    'LinearBlock': LinearBlock,
    'TransformerEncoderBlock': TransformerEncoderBlock,
}


def get_block_class(name: str):
    if name not in BLOCK_REGISTRY:
        raise ValueError(f"[layers.py] Unknown block type: {name}")
    return BLOCK_REGISTRY[name]
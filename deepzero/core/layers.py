# deepzero/core/block_base.py
import math
import torch.nn as nn
from typing import Any, Dict, List, Type
# ---------------- 공통 헬퍼 ---------------- #
_ACTIVATIONS = {
    "silu":     lambda: nn.SiLU(inplace=True),
    "relu":     lambda: nn.ReLU(inplace=True),
    "gelu":     lambda: nn.GELU(),
    "tanh":     lambda: nn.Tanh(),
    "identity": lambda: nn.Identity(),
    None:       lambda: nn.Identity(),
}
def _get_act(name):
    """name(str|None) → nn.Module"""
    return _ACTIVATIONS[name.lower() if isinstance(name, str) else None]()

class BlockBase(nn.Module):
    """모든 사용자 정의 레이어가 상속 -> 자동 registry"""
    registry: Dict[str, Type["BlockBase"]] = {}

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        BlockBase.registry[cls.__name__] = cls      # 자동 등록

    # 통일된 생성 인터페이스 (c_in, *yaml_args) -> nn.Module
    @classmethod
    def build(cls, c_in: int, *args, **kwargs) -> "BlockBase":
        return cls(c_in, *args, **kwargs)           # 기본 구현

class ConvBlock(BlockBase):
    """Conv2d (+Norm) + Act + Dropout(선택)"""
    def __init__(self, c_in, c_out, k, s,
                 act="silu", norm="bn", drop: float = 0.0):
        super().__init__()
        p = k // 2
        mods = [nn.Conv2d(c_in, c_out, k, s, p, bias=False)]

        if norm and norm.lower() == "bn":
            mods.append(nn.BatchNorm2d(c_out))
        elif norm and norm.lower() == "ln":
            mods.append(nn.GroupNorm(1, c_out))      # LayerNorm 유사
        elif norm and norm.lower() == "gn":
            mods.append(nn.GroupNorm(32, c_out))

        mods.append(_get_act(act))
        if drop and drop > 0:
            mods.append(nn.Dropout2d(drop))

        self.block = nn.Sequential(*mods)

    def forward(self, x): return self.block(x)

class Flatten(BlockBase):
    def __init__(self, *_): super().__init__()
    def forward(self, x): return x.flatten(1)

class LinearBlock(BlockBase):
    """Flatten → Linear (+Norm) + Act + Dropout"""
    def __init__(self, _c_dummy, out_f,
                 act="silu", norm="bn", drop: float = 0.0):
        super().__init__()
        mods = [Flatten(), nn.LazyLinear(out_f, bias=False)]

        if norm and norm.lower() == "bn":
            mods.append(nn.BatchNorm1d(out_f))
        elif norm and norm.lower() == "ln":
            mods.append(nn.LayerNorm(out_f))

        mods.append(_get_act(act))
        if drop and drop > 0:
            mods.append(nn.Dropout(drop))

        self.block = nn.Sequential(*mods)

    def forward(self, x): return self.block(x)

class TransformerEncoderBlock(BlockBase):
    """ViT-style Encoder → 항상 (B,C,H,W) 로 복귀"""
    def __init__(self, dim: int, heads: int, mlp: int = 4):
        super().__init__()
        self.dim = dim
        self.ln1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.ln2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * mlp),
            nn.GELU(),
            nn.Linear(dim * mlp, dim),
        )

    def forward(self, x):
        if x.dim() == 4:                       # (B,C,H,W) -> (B,HW,C)
            B, C, H, W = x.shape
            x = x.flatten(2).transpose(1, 2)
        else:                                  # (B,L,C) 입력
            B, L, C = x.shape
            H = W = int(math.sqrt(L))
            assert H * W == L, "L이 정사각 격자에 대응되지 않습니다."

        x = x + self.attn(self.ln1(x), self.ln1(x), self.ln1(x))[0]
        x = x + self.mlp(self.ln2(x))
        return x.transpose(1, 2).reshape(B, self.dim, H, W)

class GlobalAvgPool(BlockBase):
    def __init__(self, *_): super().__init__()
    def forward(self, x): return x.mean(dim=(2, 3))   # (B,C)

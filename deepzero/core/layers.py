"""
• 모든 BlockBase 서브클래스가 공통 kwargs
  {drop: float, act: str|None, norm: str|None}
  를 받아들일 수 있게 수정
• 필요 없는 인자는 무시해도 되지만 에러 없이 동작해야 함
"""

import math
from typing import Any, Dict, List, Type

import torch.nn as nn

# ---------------------------------------------------------------------- #
# 공통 헬퍼
# ---------------------------------------------------------------------- #
_ACTIVATIONS = {
    "silu":     lambda: nn.SiLU(inplace=True),
    "relu":     lambda: nn.ReLU(inplace=True),
    "gelu":     lambda: nn.GELU(),
    "tanh":     lambda: nn.Tanh(),
    "identity": lambda: nn.Identity(),
    None:       lambda: nn.Identity(),
}
_NORMS = {
    "bn":   lambda c: nn.BatchNorm2d(c) if c > 1 else nn.BatchNorm1d(c),
    "ln":   lambda c: nn.GroupNorm(1, c),            # LayerNorm 유사
    "gn":   lambda c: nn.GroupNorm(32, c),
    "none": lambda c: nn.Identity(),
    None:   lambda c: nn.Identity(),
}


def _get_act(name: str | None):
    return _ACTIVATIONS[name.lower() if isinstance(name, str) else None]()


def _get_norm(name: str | None, c: int):
    return _NORMS[name.lower() if isinstance(name, str) else None](c)


# ---------------------------------------------------------------------- #
# BlockBase
# ---------------------------------------------------------------------- #
class BlockBase(nn.Module):
    """모든 사용자 정의 레이어가 상속 → 자동 registry"""
    registry: Dict[str, Type["BlockBase"]] = {}

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        BlockBase.registry[cls.__name__] = cls  # 자동 등록

    # 통일된 생성 인터페이스 (c_in, *yaml_args, **yaml_kwargs) -> nn.Module
    @classmethod
    def build(cls, c_in: int, *args, **kwargs) -> "BlockBase":
        return cls(c_in, *args, **kwargs)  # 기본 구현


# ---------------------------------------------------------------------- #
# 개별 블록 구현
# ---------------------------------------------------------------------- #
class ConvBlock(BlockBase):
    """Conv2d (+Norm) + Act + Dropout(선택)"""
    def __init__(
        self,
        c_in: int,
        c_out: int,
        k: int,
        s: int,
        *,
        act: str | None = "silu",
        norm: str | None = "ln",
        drop: float = 0.0,
        **unused,
    ):
        super().__init__()
        p = k // 2
        mods: List[nn.Module] = [nn.Conv2d(c_in, c_out, k, s, p, bias=False)]

        # Norm
        if norm and norm.lower() != "none":
            # Conv 차원을 기준으로 2D 정규화
            mods.append(_get_norm(norm, c_out))

        # Act
        mods.append(_get_act(act))

        # Dropout
        if drop and drop > 0:
            mods.append(nn.Dropout2d(drop))

        self.block = nn.Sequential(*mods)

    def forward(self, x):  # noqa: D401
        return self.block(x)


class Flatten(BlockBase):
    """(B,C,H,W)/(B,*) → (B, -1)"""
    def __init__(self, *_ignore, **kwargs):
        super().__init__()
        self.act = _get_act(kwargs.get("act"))
        self.drop = nn.Dropout(kwargs.get("drop", 0.0)) if kwargs.get("drop", 0.0) else nn.Identity()

    def forward(self, x):
        x = x.flatten(1)
        return self.drop(self.act(x))


class LinearBlock(BlockBase):
    """Flatten → Linear (+Norm) + Act + Dropout"""
    def __init__(
        self,
        _c_dummy: int,          # Conv 계열과 시그니처 맞추기 위한 자리값
        out_f: int,
        *,
        act: str | None = "silu",
        norm: str | None = "ln",
        drop: float = 0.0,
        **unused,
    ):
        super().__init__()
        mods: List[nn.Module] = [Flatten(), nn.LazyLinear(out_f, bias=False)]

        # Norm
        if norm and norm.lower() != "none":
            if norm.lower() == "bn":
                mods.append(nn.BatchNorm1d(out_f))
            else:                           # ln / gn
                mods.append(_get_norm(norm, out_f))

        # Act
        mods.append(_get_act(act))

        # Dropout
        if drop and drop > 0:
            mods.append(nn.Dropout(drop))

        self.block = nn.Sequential(*mods)

    def forward(self, x):  # noqa: D401
        return self.block(x)


class TransformerEncoderBlock(BlockBase):
    """ViT-style Encoder → 항상 (B,C,H,W) 로 복귀"""
    def __init__(
        self,
        dim: int,
        heads: int,
        mlp: int = 4,
        *,
        act: str | None = "gelu",   # Attention 뒤 MLP 활성화
        drop: float = 0.0,
        **unused,
    ):
        super().__init__()
        self.dim = dim
        self.ln1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, dropout=drop, batch_first=True)
        self.ln2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * mlp),
            _get_act(act),
            nn.Dropout(drop) if drop else nn.Identity(),
            nn.Linear(dim * mlp, dim),
        )

    def forward(self, x):
        # (B,C,H,W) → (B,HW,C)
        if x.dim() == 4:
            B, C, H, W = x.shape
            x = x.flatten(2).transpose(1, 2)
        else:  # (B,L,C)
            B, L, C = x.shape
            H = W = int(math.sqrt(L))
            assert H * W == L, "L이 정사각 격자에 대응되지 않습니다."

        x = x + self.attn(self.ln1(x), self.ln1(x), self.ln1(x))[0]
        x = x + self.mlp(self.ln2(x))

        # (B,HW,C) → (B,C,H,W)
        return x.transpose(1, 2).reshape(B, self.dim, H, W)


class GlobalAvgPool(BlockBase):
    """(B,C,H,W) → (B,C)"""
    def __init__(self, *_ignore, **kwargs):
        super().__init__()
        self.act = _get_act(kwargs.get("act"))
        self.drop = nn.Dropout(kwargs.get("drop", 0.0)) if kwargs.get("drop", 0.0) else nn.Identity()

    def forward(self, x):
        x = x.mean(dim=(2, 3))  # (B,C)
        return self.drop(self.act(x))
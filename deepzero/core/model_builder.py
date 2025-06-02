"""
model_builder.py

요구사항을 충족하는 YAML 기반 동적 모델 빌더.
- YOLO-style 튜플 표기와 딕셔너리 표기 모두 지원
- depth_multiple / width_multiple 전역 스케일링
- from 인덱스를 통한 잔차·Concat 그래프
- 레이어별 act / norm / drop 지정
"""

from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import torch
import torch.nn as nn
import yaml

# ------------------------------------------------------------------ #
# 레이어 레지스트리 (예: ConvBlock, LinearBlock 등)
# ------------------------------------------------------------------ #
MODULE_REGISTRY: Dict[str, nn.Module] = {}


def register_module(cls):
    """@register_module 데코레이터로 레지스트리에 자동 등록"""
    MODULE_REGISTRY[cls.__name__] = cls
    return cls


# ------------------------------------------------------------------ #
# 유틸
# ------------------------------------------------------------------ #
def _load_cfg(src: Union[str, Path, dict]) -> Dict[str, Any]:
    if isinstance(src, dict):
        return src
    src = Path(src)
    if not src.exists():
        raise FileNotFoundError(src)
    with src.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


# ------------------------------------------------------------------ #
# 핵심 빌더
# ------------------------------------------------------------------ #
class _YAMLNet(nn.Module):
    """YOLO-style YAML → PyTorch 그래프"""

    def __init__(self, cfg: Union[str, Path, dict], mode: str = "train"):
        super().__init__()
        self.cfg = _load_cfg(cfg)
        self.mode = mode

        # 글로벌 설정
        self.in_channels = self.cfg.get("in_channels", 3)
        self.num_classes = self.cfg.get("num_classes", 1000)
        self.d_mult = float(self.cfg.get("depth_multiple", 1.0))
        self.w_mult = float(self.cfg.get("width_multiple", 1.0))

        self.layer_defs: List[Any] = self.cfg["layers"]
        self.model, self.save = self._build()

    # -------------------------------------------------------------- #
    def _norm_layer_def(self, entry: Any) -> Dict[str, Any]:
        """튜플/리스트/딕트를 단일 dict 형태로 변환"""
        if isinstance(entry, (list, tuple)):
            base = list(entry)
            kwargs = {}
            if base and isinstance(base[-1], dict):
                kwargs = base.pop(-1)
            from_idx, repeat, module_name, *args = base
            return dict(from_=from_idx, repeat=repeat, module=module_name, args=args, kwargs=kwargs)
        if isinstance(entry, dict):
            return dict(
                from_=entry.get("from", -1),
                repeat=entry.get("repeat", 1),
                module=entry["type"],
                args=entry.get("args", []),
                kwargs=entry.get("kwargs", {}),
            )
        raise TypeError(f"Invalid layer definition: {entry}")

    # -------------------------------------------------------------- #
    def _build(self):
        layers = nn.ModuleList()
        save: List[int] = []
        ch: List[int] = [self.in_channels]  # outs channel tracker (index aligned)

        for i, raw in enumerate(self.layer_defs):
            info = self._norm_layer_def(raw)
            f, n, m_name, args, kwargs = info["from_"], info["repeat"], info["module"], list(info["args"]), dict(info["kwargs"])

            # depth scaling
            n = max(round(n * self.d_mult), 1) if n > 1 else n

            # width scaling (첫 arg가 채널/피처인 경우)
            if args and isinstance(args[0], int):
                args[0] = int(args[0] * self.w_mult)

            m_cls = MODULE_REGISTRY.get(m_name)
            if m_cls is None:
                raise ValueError(f"Module {m_name} not registered")

            def _in_channels():
                if isinstance(f, int):
                    return ch[f if f != -1 else -1]
                return sum(ch[j] for j in f)

            in_ch = _in_channels()
            blocks = []
            for _ in range(n):
                blk = m_cls(in_ch, *args, **kwargs) if args else m_cls(in_ch, **kwargs)
                blocks.append(blk)
                if args:
                    in_ch = args[0]  # 다음 반복을 위해 out_channels 갱신

            layers.append(blocks[0] if n == 1 else nn.Sequential(*blocks))

            # output channels 추정
            out_ch = args[0] if args else in_ch
            ch.append(out_ch)

            # skip/concat용 save 인덱스 기록
            if isinstance(f, (list, tuple)) or f < i - 1:
                save.append(i)

        return layers, sorted(save)

    # -------------------------------------------------------------- #
    def forward(self, x):
        outs = [x]
        for i, m in enumerate(self.model):
            info = self._norm_layer_def(self.layer_defs[i])
            f = info["from_"]
            if isinstance(f, int):
                x_in = outs[f if f != -1 else -1]
            else:
                x_in = torch.cat([outs[j] for j in f], dim=1)
            outs.append(m(x_in))
        return outs[-1]


# ------------------------------------------------------------------ #
# 사용자 친화적 Dataclass 래퍼
# ------------------------------------------------------------------ #
@dataclass
class ModelBuilder:
    """
    YAML 설정 기반 동적 모델 생성기
    Example:
        builder = ModelBuilder(cfg="configs/example.yaml")
        model   = builder.build()
    """

    cfg: Union[str, Path, dict]
    mode: str = "train"
    _model: nn.Module = field(init=False, repr=False)

    def __post_init__(self):
        self._model = _YAMLNet(self.cfg, self.mode)

    # 내부 nn.Module 직접 접근용 프로퍼티
    @property
    def model(self) -> nn.Module:
        return self._model

    # 빌드된 모델 반환
    def build(self) -> nn.Module:
        return self._model

    # 간단 요약
    def summary(self) -> str:
        lines = ["Model Summary", "=" * 40]
        for i, layer in enumerate(self._model.model):
            lines.append(f"{i:3d}: {layer.__class__.__name__}")
        return "\n".join(lines)


# ------------------------------------------------------------------ #
def build_model_from_yaml(cfg: Union[str, Path, dict], mode: str = "train") -> nn.Module:
    """Convenience 함수 (기존 API 호환)"""
    return _YAMLNet(cfg, mode)
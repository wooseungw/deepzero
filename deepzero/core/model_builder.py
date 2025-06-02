import math
import yaml
import torch
import torch.nn as nn
from typing import Any, Dict, List
from .layers import BlockBase

def _n_rep(n, dm): return max(round(n * dm), 1)

def _make_divisible(v, d=8): return math.ceil(v / d) * d

class Layer(nn.Module):
    """from 인덱스의 출력들을 받아 내부 block 호출"""
    def __init__(self, block: nn.Module, from_idx: int):
        super().__init__()
        self.block, self.f = block, from_idx
    def forward(self, outs: List[torch.Tensor]):
        return self.block(outs[self.f])

def _resolve(v, cfg):
    """
    • v 가 문자열이고 cfg 에 키가 존재하면 cfg 값으로 치환
    • 그 외에는 그대로 반환
    """
    return cfg[v] if isinstance(v, str) and v in cfg else v

def parse_model(cfg: Dict[str, Any]) -> nn.Module:
    gd, gw = cfg["depth_multiple"], cfg["width_multiple"]
    layers_cfg = cfg["layers"]
    ch: List[int] = [cfg["in_channels"]]
    layers: List[nn.Module] = []

    for (f, n, name, *args) in layers_cfg:
        # YAML 마지막 값이 dict라면 공통 kwargs로 분리
        kw = {}
        if args and isinstance(args[-1], dict):
            kw = args[-1]
            args = args[:-1]
        n = _n_rep(n, gd)
        BlockCls = BlockBase.registry[name]

        if name == "ConvBlock":
            c_out, k, s = [_resolve(a, cfg) for a in args]
            c_out = _make_divisible(int(c_out * gw))
            blocks = [BlockCls.build(ch[f] if j == 0 else c_out,
                                     c_out, k, s, **kw)
                      for j in range(n)]
            ch.append(c_out)

        elif name == "LinearBlock":
            out_f = _resolve(args[0], cfg)
            # positional 두 번째 인자를 act 로 간주 (선택)
            if len(args) > 1:
                kw.setdefault("act", args[1])
            blocks = [BlockCls.build(ch[f], out_f, **kw) for _ in range(n)]
            ch.append(out_f)

        elif name == "TransformerEncoderBlock":
            dim, heads, mlp = [_resolve(a, cfg) for a in args]
            blocks = [BlockCls.build(dim, heads, mlp, **kw) for _ in range(n)]
            ch.append(dim)

        else:  # GlobalAvgPool 등
            blocks = [BlockCls.build(ch[f], *args, **kw) for _ in range(n)]
            ch.append(ch[f])

        seq = nn.Sequential(*blocks) if len(blocks) > 1 else blocks[0]
        layers.append(Layer(seq, f))

    return Model(layers)


# -----------------------------------------------------------------
# Model wrapper for tensor-list propagation
# -----------------------------------------------------------------
class Model(nn.Module):
    """Wraps the parsed layer list and handles tensor‑list propagation."""
    def __init__(self, layers: List[nn.Module]):
        super().__init__()
        self.layers = nn.ModuleList(layers)

    def forward(self, x: torch.Tensor):
        outs: List[torch.Tensor] = [x]  # index 0 corresponds to input
        for layer in self.layers:
            y = layer(outs)
            outs.append(y)
        return outs[-1]  # final output

def build_model_from_yaml(path: str, in_channels: int = 3) -> nn.Module:
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cfg["in_channels"] = in_channels
    model = parse_model(cfg)
    model.yaml = cfg
    return model
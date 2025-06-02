import math
import yaml
import torch
import torch.nn as nn
from typing import Any, Dict, List, Union
from .layers import BlockBase   # BlockBase.registry 활용 (ConvBlock, …)

# -------------------- 헬퍼 -------------------- #
def _n_rep(n, dm): return max(round(n * dm), 1)
def _make_divisible(v, d=8): return math.ceil(v / d) * d
def _resolve(v, cfg):        # 문자열 → cfg 치환
    return cfg[v] if isinstance(v, str) and v in cfg else v


# ---------------- 모델 파서 ------------------ #
def parse_model(cfg: Dict[str, Any]) -> "Model":
    gd, gw = cfg["depth_multiple"], cfg["width_multiple"]
    layers_cfg = cfg["layers"]

    ch:  List[int] = [cfg["in_channels"]]   # index 0은 입력
    blocks:    List[nn.Module] = []        # nn.Module 들
    from_idxs: List[Union[int, List[int]]] = []  # 각 모듈 입력 인덱스

    for (f, n, name, *args) in layers_cfg:
        # kwargs 분리
        kw = {}
        if args and isinstance(args[-1], dict):
            kw, args = args[-1], args[:-1]

        n = _n_rep(n, gd)                  # depth scaling
        BlockCls = BlockBase.registry[name]

        # ------ 블록 생성 ------ #
        if name == "ConvBlock":
            c_out, k, s = [_resolve(a, cfg) for a in args]
            c_out = _make_divisible(int(c_out * gw))
            seq = nn.Sequential(*[
                BlockCls.build(ch[f] if j == 0 else c_out,
                               c_out, k, s, **kw)
                for j in range(n)
            ])
            out_c = c_out

        elif name == "LinearBlock":
            out_f = _resolve(args[0], cfg)
            if len(args) > 1:              # positional 2번째 act 처리
                kw.setdefault("act", args[1])
            seq = nn.Sequential(*[
                BlockCls.build(ch[f], out_f, **kw)
                for _ in range(n)
            ])
            out_c = out_f

        elif name == "TransformerEncoderBlock":
            dim, heads, mlp = [_resolve(a, cfg) for a in args]
            seq = nn.Sequential(*[
                BlockCls.build(dim, heads, mlp, **kw)
                for _ in range(n)
            ])
            out_c = dim

        else:                              # GlobalAvgPool 등
            seq = nn.Sequential(*[
                BlockCls.build(ch[f], *args, **kw)
                for _ in range(n)
            ])
            out_c = ch[f]                  # 채널 유지

        # ----------------------- #
        blocks.append(seq)
        from_idxs.append(f)
        ch.append(out_c)

    return Model(blocks, from_idxs)


# ---------------- 모델 래퍼 ------------------ #
class Model(nn.Module):
    """
    • self.blocks   : nn.ModuleList (ConvBlock / LinearBlock / ...)
    • self.from_idxs: 각 블록이 참조할 입력 인덱스(단일 int 또는 list → concat)
    """
    def __init__(self,
                 blocks: List[nn.Module],
                 from_idxs: List[Union[int, List[int]]]):
        super().__init__()
        self.blocks = nn.ModuleList(blocks)
        self.from_idxs = from_idxs

    def forward(self, x: torch.Tensor):
        outs: List[torch.Tensor] = [x]          # 0번 = 원 입력
        for block, f in zip(self.blocks, self.from_idxs):
            if isinstance(f, int):
                x_in = outs[f if f != -1 else -1]
            else:                               # concat
                x_in = torch.cat([outs[j] for j in f], dim=1)
            outs.append(block(x_in))
        return outs[-1]


# ---------------- YAML 빌더 ------------------ #
def build_model_from_yaml(path: str, in_channels: int = 3) -> nn.Module:
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cfg["in_channels"] = in_channels
    model = parse_model(cfg)
    model.yaml = cfg          # 원본 YAML 보존 (선택)
    return model
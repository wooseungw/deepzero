# ──────────────────────────────────────────────────────────────────────────────
# YAML 파일을 해석하여 Dezero 네트워크를 동적으로 구성하는 YOLOModel 클래스
# ──────────────────────────────────────────────────────────────────────────────

import numpy as np
import yaml
from flexzero import Model
import flexzero.functions as F
from .blocks import get_block_class

class YamlModel(Model):
    """
    YOLO 스타일의 YAML 파일을 파싱하여 Dezero 네트워크를 동적으로 생성.
    """
    def __init__(self, config_path: str):
        super().__init__()
        # 1) YAML 로드
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)

        # 2) 기본 설정
        self.in_channels = config['in_channels']
        self.num_classes = config['num_classes']
        self.depth_mul = config.get('depth_multiple', 1.0)
        self.width_mul = config.get('width_multiple', 1.0)
        self.layer_cfgs = config['layers']

        # 3) 모듈(블록) 저장 리스트
        #    각 원소: {'f': from_idx, 'block': Module 인스턴스}
        self.layers = []

        prev_channels = self.in_channels  # 직전 블록의 출력 채널(FC 차원)

        # 4) YAML 레이어 순회
        for i, layer_cfg in enumerate(self.layer_cfgs):
            # Normalize layer configuration to 7 elements
            lc = layer_cfg
            if len(lc) == 3:
                from_idx, n_rep, block_type = lc
                out_ch, kernel, stride, opts = 0, 0, 0, {}
            elif len(lc) == 4:
                from_idx, n_rep, block_type, out_ch = lc
                kernel, stride, opts = 0, 0, {}
            elif len(lc) == 5:
                from_idx, n_rep, block_type, out_ch, extra = lc
                if block_type == 'LinearBlock':
                    kernel, stride, opts = 0, 0, {'act': extra}
                else:
                    raise ValueError(f"Unsupported config length=5 for block_type '{block_type}': {lc}")
            elif len(lc) == 7:
                from_idx, n_rep, block_type, out_ch, kernel, stride, opts = lc
            else:
                raise ValueError(f"Unsupported layer config (length {len(lc)}): {lc}")

            # 4-1) depth scaling
            n_rep_scaled = max(round(n_rep * self.depth_mul), 1)

            # 4-2) width scaling (Conv/Linear 블록에만 적용)
            if block_type in ('GlobalAvgPool',):
                out_ch_scaled = 0
            else:
                # num_classes 문자열이 들어올 수 있으므로 처리
                if isinstance(out_ch, str) and out_ch == 'num_classes':
                    raw_ch = self.num_classes
                else:
                    raw_ch = int(out_ch)
                # divisor=8로 맞추기
                out_ch_scaled = int(np.ceil((raw_ch * self.width_mul) / 8) * 8)

            # 4-3) n_rep_scaled번만큼 반복해서 블록 생성
            for rep in range(n_rep_scaled):
                BlockClass = get_block_class(block_type)

                # ConvBlock
                if block_type == 'ConvBlock':
                    block = BlockClass(
                        in_channels=prev_channels,
                        out_channels=out_ch_scaled,
                        kernel_size=kernel,
                        stride=stride,
                        act=opts.get('act', None),
                        norm=opts.get('norm', None),
                        drop=opts.get('drop', 0.0)
                    )
                    prev_channels = out_ch_scaled

                # GlobalAvgPool
                elif block_type == 'GlobalAvgPool':
                    block = BlockClass()
                    # 풀링 후 (B, C) → 이 채널 수를 그대로 FC에 넘길 예정

                # LinearBlock
                elif block_type == 'LinearBlock':
                    # in_features = prev_channels
                    # out_features: out_ch_scaled(이미 num_classes일 때 처리됨)
                    block = BlockClass(
                        in_features=prev_channels,
                        out_features=(out_ch_scaled if out_ch_scaled > 0 else self.num_classes),
                        act=opts.get('act', None),
                        drop=opts.get('drop', 0.0)
                    )
                    prev_channels = out_ch_scaled if out_ch_scaled > 0 else self.num_classes

                # TransformerEncoderBlock
                elif block_type == 'TransformerEncoderBlock':
                    # YAML에서 kernel → num_heads, stride → hidden_dim 으로 가정
                    block = BlockClass(
                        embed_dim=prev_channels,
                        num_heads=kernel,
                        hidden_dim=stride,
                        drop=opts.get('drop', 0.0)
                    )
                    # Transformer 출력 차원은 prev_channels 그대로 유지

                else:
                    raise ValueError(f"[builder.py] Unknown block_type: {block_type}")

                # 4-4) 서브모듈 등록 & layers 리스트에 추가
                self.add_child(f'layer_{i}_{rep}', block)
                self.layers.append({'f': from_idx, 'block': block})

        # 최종 출력 차원이 self.num_classes인지 확인 (LinearBlock이 마지막에 num_classes로 맞춰짐)

    def forward(self, x):
        """
        YOLO 방식: outputs 리스트를 유지하며, from_idx 에 따라 이전 데이터를 꺼내와 블록 순차 연산
        """
        outputs = [x]  # outputs[0]: 입력

        for layer_info in self.layers:
            f_idx = layer_info['f']
            block = layer_info['block']

            # from_idx가 여러 개일 경우(Residual / Concat) 처리 가능하게 확장할 수 있음.
            # 여기서는 단일 인덱스(정수)만 지원하는 예시.
            if isinstance(f_idx, (list, tuple)):
                # concat 예시: axis=1 (채널 방향)
                inputs = [outputs[i] if i != -1 else outputs[-1] for i in f_idx]
                h_prev = F.concat(inputs, axis=1)
            else:
                h_prev = outputs[f_idx] if f_idx != -1 else outputs[-1]

            h_out = block(h_prev)
            outputs.append(h_out)

        return outputs[-1]
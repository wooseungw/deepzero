from dataclasses import dataclass, field
from typing import Any, Dict, Tuple, Union

@dataclass
class ModelBuilder:
    """
    YAML 설정을 받아 PyTorch 모델을 동적으로 생성하는 빌더 클래스
    """
    name: str
    layers: Union[list, Dict[str, list]]
    input_dim: Union[str, Tuple[int, ...]] = None
    hidden_dim: int = None
    num_classes: int = None
    dropout: float = 0.0
    additional_params: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        import torch.nn as nn
        self.layer_map = {
            'conv': self._build_conv_layer,
            'linear': self._build_linear_layer,
            'lstm': self._build_lstm_layer,
            'transformer': self._build_transformer_layer,
            'relu': lambda: nn.ReLU(),
            'dropout': lambda p=0.5: nn.Dropout(p),
            'batchnorm': self._build_batchnorm_layer
        }

    def _build_conv_layer(self, in_ch, out_ch, kernel, stride=1, padding=0):
        import torch.nn as nn
        return nn.Conv2d(in_ch, out_ch, kernel, stride, padding)

    def _build_linear_layer(self, in_features, out_features):
        import torch.nn as nn
        return nn.Linear(in_features, out_features)

    def _build_lstm_layer(self, input_size, hidden_size, num_layers=1):
        import torch.nn as nn
        return nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)

    def _build_transformer_layer(self, d_model, nhead, num_layers=1):
        import torch.nn as nn
        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead)
        return nn.TransformerEncoder(encoder_layer, num_layers)

    def _build_batchnorm_layer(self, num_features):
        import torch.nn as nn
        return nn.BatchNorm2d(num_features)

    def _parse_layer_config(self, layer_config):
        """레이어 설정을 파싱하여 레이어 객체 생성"""
        if not isinstance(layer_config, list) or len(layer_config) == 0:
            return None
        
        layer_type = layer_config[0]
        layer_params = layer_config[1:]
        
        if layer_type in self.layer_map:
            return self.layer_map[layer_type](*layer_params)
        else:
            raise ValueError(f"Unsupported layer type: {layer_type}")

    def build_model(self):
        """설정값에 따라 실제 PyTorch 모델을 생성합니다."""
        import torch.nn as nn
        
        layers = []
        
        if isinstance(self.layers, list):
            # 단순 리스트 형태: [['conv', 3, 64, 3, 1, 1], ['relu'], ...]
            for layer_config in self.layers:
                layer = self._parse_layer_config(layer_config)
                if layer:
                    layers.append(layer)
        
        elif isinstance(self.layers, dict):
            # 딕셔너리 형태: {"embedding": [...], "encoder": [...], ...}
            for section_name, section_layers in self.layers.items():
                if isinstance(section_layers, list):
                    if section_layers and isinstance(section_layers[0], list):
                        # 중첩 리스트: [['conv', ...], ['relu'], ...]
                        for layer_config in section_layers:
                            layer = self._parse_layer_config(layer_config)
                            if layer:
                                layers.append(layer)
                    else:
                        # 단일 레이어: ['conv', 3, 64, 3, 1, 1]
                        layer = self._parse_layer_config(section_layers)
                        if layer:
                            layers.append(layer)
        
        return nn.Sequential(*layers)

    def summary(self):
        """모델 구조 요약 정보를 출력합니다."""
        print(f"Model: {self.name}")
        print(f"Layers: {self.layers}")
        if self.input_dim:
            print(f"Input Dimension: {self.input_dim}")
        if self.num_classes:
            print(f"Number of Classes: {self.num_classes}")
            

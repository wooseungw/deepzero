from dataclasses import dataclass, field
from typing import Any, Dict, Tuple, Union

@dataclass
class ModelBuilder:
    """
    YAML 설정을 받아 PyTorch 모델을 동적으로 생성하는 빌더 클래스
    """
    name: str
    input_dim: Union[str, Tuple[int, ...]]
    hidden_dim: int
    num_classes: int
    dropout: float = 0.0
    additional_params: Dict[str, Any] = field(default_factory=dict)

    def build_model(self):
        """
        설정값에 따라 실제 PyTorch 모델을 생성합니다.
        """
        pass

    def summary(self):
        """
        모델 구조 요약 정보를 출력합니다.
        """
        pass
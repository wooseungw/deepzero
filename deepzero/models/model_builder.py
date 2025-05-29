from dataclasses import dataclass, field
from typing import Any, Dict

@dataclass
class ModelBuilder:
    name: str
    input_dim: tuple
    hidden_dim: int
    num_classes: int
    dropout: float = 0.0
    additional_params: Dict[str, Any] = field(default_factory=dict)

    def build_model(self):
        # This method will implement the logic to build the model
        # based on the provided parameters.
        pass

    def summary(self):
        # This method will provide a summary of the model architecture.
        pass
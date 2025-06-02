
from deepzero.core.utils import text_summary, graphviz_plot
from deepzero.core.model_builder import build_model_from_yaml  # 사용자의 빌더 모듈

model = build_model_from_yaml("deepzero/configs/model.yaml", in_channels=3)

# 텍스트 요약
text_summary(model, input_size=(1, 3, 224, 224))

# 그래프 저장
graphviz_plot(model, input_size=(1, 3, 224, 224), filename="example_graph.png")
"""
모델 구조를 시각화하기 위한 유틸
• text_summary(model, input_size): 터미널 표 형식 요약 (torchinfo 사용)
• graphviz_plot(model, input_size, filename): 계층 그래프(PNG/PDF) 저장 (torchviz 사용)
"""

import torch
import torch.nn as nn

# ---- 1) 터미널 요약 (torchinfo) ------------------------------------------
def text_summary(model: nn.Module, input_size=(1, 3, 224, 224)):
    """
    터미널에 파라미터/입출력 형태 요약을 출력합니다.
    pip install torchinfo
    """
    from torchinfo import summary
    summary(model, input_size=input_size, col_names=("input_size", "output_size", "num_params"))

# ---- 2) 그래프 그림 (torchviz) -------------------------------------------
def graphviz_plot(model: nn.Module,
                  input_size=(1, 3, 224, 224),
                  filename: str = "model_graph.png"):
    """
    모델을 Graphviz 다이어그램(PNG/PDF/SVG)에 저장합니다.

    Requirements
    ------------
    pip install torchviz graphviz
    • Ubuntu: sudo apt-get install graphviz
    • macOS : brew install graphviz
    """
    from torchviz import make_dot

    device = next(model.parameters()).device
    dummy = torch.randn(*input_size).to(device)
    out = model(dummy)

    dot = make_dot(out, params=dict(model.named_parameters()))
    dot.format = filename.split(".")[-1]  # 확장자에 따라 png/pdf/svg 저장
    dot.render(filename.split(".")[0], cleanup=True)  # e.g. model_graph
    print(f"Saved graph to {filename}")


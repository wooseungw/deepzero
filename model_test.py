from deepzero.core.model_builder import build_model_from_yaml

if __name__ == "__main__":
    import torch

    model = build_model_from_yaml("deepzero/configs/model.yaml")
    x = torch.randn(2, 3, 224, 224)
    logits = model(x)                 # forward
    print("output shape:", logits.shape)
    print("params (M):", sum(p.numel() for p in model.parameters()) / 1e6)
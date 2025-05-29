import torch
import torch.nn.functional as F

class MeanSquaredError:
    def __call__(self, y_true, y_pred):
        return F.mse_loss(y_pred, y_true)

class CrossEntropy:
    def __call__(self, y_true, y_pred):
        # y_pred: (batch, num_classes), raw logits
        # y_true: (batch,) or (batch, num_classes) (one-hot)
        if y_true.dim() == 2:
            y_true = y_true.argmax(dim=1)
        return F.cross_entropy(y_pred, y_true)

def binary_cross_entropy(y_true, y_pred):
    return F.binary_cross_entropy(y_pred, y_true)

def categorical_cross_entropy(y_true, y_pred):
    # y_pred: (batch, num_classes), 확률값
    # y_true: (batch, num_classes), one-hot
    epsilon = 1e-12
    y_pred = torch.clamp(y_pred, epsilon, 1. - epsilon)
    return -torch.sum(y_true * torch.log(y_pred), dim=1).mean()
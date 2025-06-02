import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split

# 예시 모델 (사용자 모델로 교체 가능)
class SimpleNet(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.fc = nn.Linear(input_dim, num_classes)
    def forward(self, x):
        return self.fc(x)

import os
print("현재 작업 디렉토리:", os.getcwd())

# YAML 파일에서 하이퍼파라미터 로드
with open('TrainArgs.yaml', 'r', encoding='utf-8') as f:
    args = yaml.safe_load(f)
    
# 데이터셋 예시 (사용자 데이터셋으로 교체)
from sklearn.datasets import make_classification
from torch.utils.data import TensorDataset

X, y = make_classification(n_samples=1000, n_features=20, n_classes=2)
X = torch.tensor(X, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.long)
dataset = TensorDataset(X, y)

# 데이터 분할
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=args['batch_size'], shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=args['batch_size'])

# 모델, loss, optimizer 정의
model = SimpleNet(input_dim=20, num_classes=2)
loss_fn = getattr(nn, args['loss'])()
optimizer = getattr(optim, args['optimizer'])(model.parameters(), lr=args['learning_rate'])

# 학습 및 검증 루프
for epoch in range(args['epochs']):
    model.train()
    train_loss = 0
    for xb, yb in train_loader:
        optimizer.zero_grad()
        preds = model(xb)
        loss = loss_fn(preds, yb)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    train_loss /= len(train_loader)

    model.eval()
    val_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for xb, yb in val_loader:
            preds = model(xb)
            loss = loss_fn(preds, yb)
            val_loss += loss.item()
            correct += (preds.argmax(1) == yb).sum().item()
            total += yb.size(0)
    val_loss /= len(val_loader)
    val_acc = correct / total

    print(f"Epoch {epoch+1}/{args['epochs']} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")
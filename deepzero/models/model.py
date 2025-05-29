import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleCNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes, dropout):
        super(SimpleCNN, self).__init__()
        # input_dim: (채널, 높이, 너비) 또는 "3,32,32" 형태의 문자열
        if isinstance(input_dim, str):
            input_dim = tuple(map(int, input_dim.split(',')))
        self.input_dim = input_dim

        self.conv1 = nn.Conv2d(input_dim[0], 32, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(2)
        # 입력 크기에서 풀링 두 번 적용 후의 feature map 크기 계산
        h, w = input_dim[1] // 4, input_dim[2] // 4
        self.fc1 = nn.Linear(64 * h * w, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.output = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.output(x)
        return x

class SimpleRNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes, dropout):
        super(SimpleRNN, self).__init__()
        # input_dim: feature 크기 또는 "28" 형태의 문자열
        if isinstance(input_dim, str):
            input_dim = int(input_dim)
        self.rnn = nn.RNN(input_dim, hidden_dim, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.output = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        # x: (batch, seq_len, input_dim)
        out, _ = self.rnn(x)
        out = out[:, -1, :]  # 마지막 타임스텝의 hidden state
        out = self.dropout(out)
        out = self.output(out)
        return out
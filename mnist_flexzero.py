import flexzero
import flexzero.functions as F
from flexzero import DataLoader
from flexzero.models import MLP
from flexzero.autobuilder import YamlModel


max_epoch = 5
batch_size = 100
hidden_size = 1000

train_set = flexzero.datasets.MNIST(train=True)
test_set = flexzero.datasets.MNIST(train=False)
train_loader = DataLoader(train_set, batch_size)
test_loader = DataLoader(test_set, batch_size, shuffle=False)


config_path = "configs/model.yaml"
# 2) 모델 생성
model = YamlModel(config_path)
optimizer = flexzero.optimizers.Adam().setup(model)
optimizer.add_hook(flexzero.optimizers.WeightDecay(1e-4))  # Weight decay

if flexzero.cuda.gpu_enable:
    train_loader.to_gpu()
    test_loader.to_gpu()
    model.to_gpu()

for epoch in range(max_epoch):
    sum_loss, sum_acc = 0, 0

    for x, t in train_loader:
        y = model(x)
        loss = F.softmax_cross_entropy(y, t)
        acc = F.accuracy(y, t)
        model.cleargrads()
        loss.backward()
        optimizer.update()

        sum_loss += float(loss.data) * len(t)
        sum_acc += float(acc.data) * len(t)

    print('epoch: {}'.format(epoch+1))
    print('train loss: {}, accuracy: {}'.format(
        sum_loss / len(train_set), sum_acc / len(train_set)))

    sum_loss, sum_acc = 0, 0
    with flexzero.no_grad():
        for x, t in test_loader:
            y = model(x)
            loss = F.softmax_cross_entropy(y, t)
            acc = F.accuracy(y, t)
            sum_loss += float(loss.data) * len(t)
            sum_acc += float(acc.data) * len(t)

    print('test loss: {}, accuracy: {}'.format(
        sum_loss / len(test_set), sum_acc / len(test_set)))
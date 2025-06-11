import numpy as np
import matplotlib.pyplot as plt
import flexo
import flexo.functions as F
from flexo import DataLoader
from flexo.optimizers import Adam
from flexo.autobuilder import YamlModel
from torchvision.models import vgg16


import torch

class VGG16Model(YamlModel):
    def __init__(self, config_path, num_classes=10):
        super().__init__(config_path)
        self.num_classes = num_classes

    def load_partial_pretrained(self):
        """PyTorch VGG16 pretrained weight를 일부 레이어에만 로드"""
        vgg = vgg16(weights="IMAGENET1K_V1")
        vgg_state = vgg.state_dict()
        matched = 0
        for param in self.params():
            # param.data가 None이면 건너뜀
            if getattr(param, "data", None) is None:
                continue
            for vgg_k, vgg_param in vgg_state.items():
                if 'features' in vgg_k and 'weight' in vgg_k:
                    if param.data.shape == vgg_param.shape:
                        param.data[...] = vgg_param.cpu().numpy()
                        matched += 1
                        break
                if 'features' in vgg_k and 'bias' in vgg_k:
                    if param.data.shape == vgg_param.shape:
                        param.data[...] = vgg_param.cpu().numpy()
                        matched += 1
                        break
        print(f'Pretrained weight {matched}개 레이어에 적용됨')


def train_mnist():
    # 설정
    gpu_enable = False
    max_epoch = 10
    batch_size = 100
    lr = 0.001
    
    # 모델 생성
    model = VGG16Model('configs/vgg16.yaml')
    model.load_partial_pretrained()
    
    # 손실 함수와 옵티마이저
    optimizer = Adam().setup(model)
    optimizer.alpha = lr
    
    # 데이터 준비
    transform = lambda x: (x / 255.0).astype(np.float32)
    train_set = flexo.datasets.MNIST(train=True, transform=transform)
    test_set = flexo.datasets.MNIST(train=False, transform=transform)
    
    train_loader = DataLoader(train_set, batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size, shuffle=False)
    
    # GPU 설정
    if gpu_enable:
        model.to_gpu()
        train_loader.to_gpu()
        test_loader.to_gpu()
        print('Using GPU')
    else:
        print('Using CPU')
    
    # 학습 히스토리 저장
    history = {
        'train_loss': [],
        'train_acc': [],
        'test_loss': [],
        'test_acc': []
    }
    
    # 학습 과정
    for epoch in range(max_epoch):
        # 학습 단계
        train_loss, train_acc = 0, 0
        train_count = 0
        
        for x, t in train_loader:
            # 순전파
            y = model(x)
            loss = F.softmax_cross_entropy(y, t)
            acc = F.accuracy(y, t)
            
            # 역전파
            model.cleargrads()
            loss.backward()
            optimizer.update()
            
            # 학습 통계 갱신
            train_loss += loss.data
            train_acc += acc.data
            train_count += 1
            
            # 진행 표시
            if train_count % 10 == 0:
                print(f'Epoch {epoch+1}/{max_epoch}, Batch {train_count}/{len(train_loader)}, '
                      f'Loss: {loss.data:.4f}, Acc: {acc.data:.4f}')
        
        # 에폭별 학습 통계
        avg_train_loss = train_loss / train_count
        avg_train_acc = train_acc / train_count
        history['train_loss'].append(float(avg_train_loss))
        history['train_acc'].append(float(avg_train_acc))
        
        # 테스트 단계
        test_loss, test_acc = 0, 0
        test_count = 0
        
        with flexo.no_grad():
            for x, t in test_loader:
                y = model(x)
                loss = F.softmax_cross_entropy(y, t)
                acc = F.accuracy(y, t)
                
                test_loss += loss.data
                test_acc += acc.data
                test_count += 1
        
        # 에폭별 테스트 통계
        avg_test_loss = test_loss / test_count
        avg_test_acc = test_acc / test_count
        history['test_loss'].append(float(avg_test_loss))
        history['test_acc'].append(float(avg_test_acc))
        
        # 에폭 결과 출력
        print(f'Epoch {epoch+1}/{max_epoch} - '
              f'Train Loss: {avg_train_loss:.4f}, Train Acc: {avg_train_acc:.4f}, '
              f'Test Loss: {avg_test_loss:.4f}, Test Acc: {avg_test_acc:.4f}')
    
    # 학습 결과 시각화
    plot_history(history)
    
    return model, history


def plot_history(history):
    """학습 히스토리를 시각화합니다."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    # 손실 그래프
    axes[0].plot(history['train_loss'], label='Train')
    axes[0].plot(history['test_loss'], label='Test')
    axes[0].set_title('Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].legend()
    
    # 정확도 그래프
    axes[1].plot(history['train_acc'], label='Train')
    axes[1].plot(history['test_acc'], label='Test')
    axes[1].set_title('Accuracy')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].legend()
    
    plt.tight_layout()
    plt.show()
    
    # 그래프 저장
    fig.savefig('vgg16_mnist_history.png')


def test_model_with_examples(model, num_examples=5):
    """무작위로 선택된 테스트 이미지에 대한 모델 예측을 시각화합니다."""
    # 데이터 준비
    transform = lambda x: (x / 255.0).astype(np.float32)
    test_set = flexo.datasets.MNIST(train=False, transform=transform)
    
    # 무작위 인덱스 선택
    indices = np.random.choice(len(test_set), num_examples, replace=False)
    
    fig, axes = plt.subplots(1, num_examples, figsize=(15, 3))
    
    for i, idx in enumerate(indices):
        x, t = test_set[idx]
        x_tensor = flexo.Variable(x[np.newaxis, :])  # 배치 차원 추가
        
        # GPU 사용 시 변환
        if flexo.cuda.gpu_enable:
            x_tensor.to_gpu()
        
        # 예측
        with flexo.no_grad():
            y = model(x_tensor)
        
        # 예측 결과 CPU로 변환
        if flexo.cuda.gpu_enable:
            y.to_cpu()
        
        # 예측 클래스 및 확률 구하기
        pred_class = np.argmax(y.data)
        pred_probs = F.softmax(y).data[0]
        
        # 이미지 표시
        axes[i].imshow(x[0], cmap='gray')
        axes[i].set_title(f'True: {t}\nPred: {pred_class}\nProb: {pred_probs[pred_class]:.2f}')
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.show()
    
    # 그래프 저장
    fig.savefig('vgg16_mnist_examples.png')


if __name__ == '__main__':
    # 모델 학습
    trained_model, history = train_mnist()
    
    # 예시 테스트 실행
    test_model_with_examples(trained_model)

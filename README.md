# FlexoZero YAML 모델 레포지토리

> **YAML 기반 딥러닝 모델 빌더 - AutoBuilder**
> 파이썬 코드를 직접 수정하지 않고도 다양한 신경망 아키텍처를 설계, 학습, 공유할 수 있는 프레임워크입니다.

---

## ✨ 주요 특징

| 기능 | 설명 |
| --- | --- |
| **YAML 기반 모델 정의** | 단일 YAML 파일만으로 복잡한 신경망 구조를 정의 - 레이어 순서, 반복, 채널 수, 활성화 함수, 정규화까지 설정 가능 |
| **YOLO 스타일 문법** | `[from, n, module, *args, {kwargs}]` 형태의 직관적인 튜플 문법으로 간결하게 모델 구조 표현 |
| **동적 스케일링** | `depth_multiple`, `width_multiple` 파라미터로 모델 크기를 쉽게 조절 |
| **다양한 블록 지원** | ConvBlock, LinearBlock, GlobalAvgPool, TransformerEncoderBlock 등 기본 블록 제공 |
| **자동 데이터셋 처리** | MNIST, CIFAR10 등 표준 데이터셋을 위한 DataLoader 통합 |
| **GPU 가속** | CUDA 지원으로 빠른 학습 및 추론 성능 |
| **PyTorch 호환** | PyTorch 텐서와 함께 사용 가능한 유연한 확장성 |

---

## 📦 설치

**Python ≥ 3.8** 및 **NumPy** 필요 (PyTorch 텐서 지원 선택적)

```bash
# 레포지토리 복제
git clone https://github.com/yourusername/flexozero.git
cd flexozero

# 가상환경 생성 및 활성화 (선택 사항)
python -m venv venv
.\venv\Scripts\activate  # Windows

# 의존성 설치
pip install -r requirements.txt
```

---

## 🚀 빠른 시작

1. **YAML 모델 설정파일 작성** (`configs/` 디렉토리의 예제 참고)
2. **모델 생성 및 학습**

   ```python
   import flexo
   from flexo.autobuilder import YamlModel
   
   # YAML 파일에서 모델 생성
   model = YamlModel('configs/vgg16.yaml')
   
   # 데이터 준비
   train_loader = flexo.DataLoader(flexo.datasets.MNIST(train=True), batch_size=32)
   
   # 학습 루프
   for epoch in range(10):
       for x, t in train_loader:
           y = model(x)
           loss = F.softmax_cross_entropy(y, t)
           model.cleargrads()
           loss.backward()
           optimizer.update()
   ```

3. **학습 스크립트 실행**

   ```bash
   # 예제 VGG16 모델로 MNIST 학습
   python vgg16_mnist.py
   ```

4. **모델 정보 확인**

   ```bash
   # 모델 아키텍처 정보 출력
   python model_info.py
   ```

---

## 📝 YAML 모델 정의 형식

```yaml
# vgg16.yaml - MNIST용 간소화된 VGG16 모델 예제

# 기본 설정
in_channels: 1          # 입력 채널 수 (MNIST는 흑백 이미지)
num_classes: 10         # 출력 클래스 수
input_size: [28, 28]    # 입력 이미지 크기

# 스케일링 인자
depth_multiple: 1.0     # 레이어 반복 횟수 배율
width_multiple: 0.5     # 채널 수 배율 (작은 모델)

# 레이어 정의 [from, repeat, block_type, out_channels, kernel_size, stride, {params}]
layers:
  # Block 1
  - [-1, 1, ConvBlock, 64, 3, 1, {act: relu, norm: bn}]   # Conv1_1
  - [-1, 1, ConvBlock, 64, 3, 1, {act: relu, norm: bn}]   # Conv1_2
  - [-1, 1, ConvBlock, 64, 2, 2, {act: relu}]             # MaxPool1

  # Block 2
  - [-1, 1, ConvBlock, 128, 3, 1, {act: relu, norm: bn}]  # Conv2_1
  
  # 특성 평탄화 (Flatten)
  - [-1, 1, GlobalAvgPool, 0]                           # 전역 평균 풀링
  
  # 완전 연결 레이어 (FC)
  - [-1, 1, LinearBlock, 512, 0, 0, {act: relu, drop: 0.5}]   # FC1
  - [-1, 1, LinearBlock, 10, 0, 0, {}]                        # 출력 레이어
```

### YAML 필드 설명

| 필드 | 타입 | 설명 |
| --- | --- | --- |
| **from** | `int` | 입력으로 사용할 이전 레이어 인덱스 (-1 = 직전 레이어) |
| **repeat** | `int` | 블록 반복 횟수 (`depth_multiple`로 스케일 조정) |
| **block_type** | `str` | 블록 유형 (ConvBlock, LinearBlock, GlobalAvgPool 등) |
| **out_channels** | `int` | 출력 채널 수 (ConvBlock) 또는 출력 특성 수 (LinearBlock) |
| **kernel_size** | `int` | 커널 크기 (ConvBlock에만 적용) |
| **stride** | `int` | 스트라이드 (ConvBlock에만 적용) |
| **params** | `dict` | 선택적 매개변수 (활성화 함수, 정규화, 드롭아웃 등) |

#### 지원되는 활성화 함수 (`act`)
`relu`, `silu`, `sigmoid`, `tanh` 등

#### 정규화 옵션 (`norm`)
`bn` (BatchNormalization), `ln` (LayerNormalization)

---

## 🔧 커스텀 블록 추가

새로운 블록을 추가하려면 `flexo/blocks.py`에 클래스를 정의하고 레지스트리에 등록하세요:

```python
class MyCustomBlock(Model):
    def __init__(self, in_features, out_features, **kwargs):
        super().__init__()
        # 블록 구현
        
    def forward(self, x):
        # 순전파 구현
        return output

# 레지스트리에 등록
BLOCK_REGISTRY['MyCustomBlock'] = MyCustomBlock
```

YAML 파일에서 새 블록 사용:
```yaml
- [-1, 1, MyCustomBlock, 256, {custom_param: value}]
```

---

## 📁 프로젝트 구조

```
├─ flexo/                # 핵심 프레임워크 코드
│  ├─ autobuilder.py     # YAML 모델 빌더
│  ├─ blocks.py          # 블록 정의 및 레지스트리
│  ├─ core.py            # 변수, 함수, 역전파 등 핵심 기능
│  ├─ cuda.py            # GPU 지원 기능
│  ├─ functions.py       # 수학 함수 및 연산
│  ├─ layers.py          # 기본 레이어 정의
│  └─ models.py          # 모델 기본 클래스
├─ configs/              # YAML 모델 정의
├─ model_info.py         # 모델 정보 출력
└─ vgg16_mnist.py        # VGG16-MNIST 학습 예제
```

---

## 📊 구현된 예제 모델

* **VGG16**: MNIST 데이터셋에 최적화된 VGG16 모델 (`configs/vgg16.yaml`)
* **ResNet**: 간소화된 ResNet 아키텍처 (`configs/resnet.yaml`)
* **VAE**: Variational Autoencoder 구현 (`configs/vae.yaml`)

---

## 🙏 감사의 글

이 프로젝트는 [Chainer](https://github.com/chainer/chainer), [PyTorch](https://pytorch.org/), [YOLOv5](https://github.com/ultralytics/yolov5)에 영감을 받았습니다.

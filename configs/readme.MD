# 📑 FlexoZero 모델 YAML 가이드

> **FlexoZero AutoBuilder** (`flexo/autobuilder.py`)에서 사용할 **YAML 기반 네트워크 정의 문법**을 설명합니다.
>
> ```python
> from flexo.autobuilder import YamlModel
> model = YamlModel("configs/my_model.yaml")
> ```
>
> 위처럼 호출하면 곧바로 모델 객체가 생성됩니다.

---

## 1. 전체 YAML 골격

```yaml
in_channels: 3          # (필수) 입력 채널 수
num_classes: 100        # (선택) 전역 변수, 문자열 치환용
depth_multiple: 1.0     # (선택) 레이어 반복 스케일
width_multiple: 1.0     # (선택) 채널 수 스케일

layers:
  #   from  repeat  Block                     positional-args                 kwargs
  - [  -1,     1,   ConvBlock,                64, 3, 2,                      {act: relu, drop: 0.1} ]
  - [  -1,     2,   TransformerEncoderBlock,  128, 4, 8 ]
  - [  -1,     1,   GlobalAvgPool ]
  - [  -1,     1,   LinearBlock,              num_classes,                   {norm: ln} ]
```

## 2. 필드 설명

### from
입력으로 사용할 이전 레이어 인덱스 (-1 = 직전 레이어, -2 = 두 단계 전)

### repeat
블록 반복 횟수 (depth_multiple 값이 곱해진 뒤 반올림)

### Block
사용할 블록 유형: `ConvBlock`, `LinearBlock`, `TransformerEncoderBlock`, `GlobalAvgPool` 등

### positional args
블록별 필수 인자:
- **ConvBlock**: out_channels, kernel_size, stride
- **LinearBlock**: out_features
- **TransformerEncoderBlock**: embed_dim, num_heads, hidden_dim

### kwargs
선택적 하이퍼파라미터: act, norm, drop 등 공통 매개변수

## 3. 공통 하이퍼파라미터

| 옵션 | 값 | 적용 블록 | 기본값 |
|------|------|------------|--------|
| act | relu, silu, gelu, tanh | Conv / Linear | relu |
| norm | bn, ln, none | Conv / Linear | bn |
| drop | 0 ~ 1 (확률) | Conv / Linear | 0 |

예시: `{act: gelu, norm: ln, drop: 0.25}`

## 4. 지원 블록 & 시그니처

| 블록 | 생성자 시그니처 | 설명 |
|------|-------------|------|
| ConvBlock | in_channels, out_channels, kernel_size, stride, act, norm, drop | 합성곱 블록 |
| LinearBlock | in_features, out_features, act, drop | 완전 연결 레이어 |
| TransformerEncoderBlock | embed_dim, num_heads, hidden_dim, drop | Transformer 인코더 |
| GlobalAvgPool | - | (B,C,H,W) → (B,C) |

새 블록을 추가하려면:

```python
class MyBlock(Model):
    # 블록 구현
    ...

# 레지스트리에 등록
BLOCK_REGISTRY['MyBlock'] = MyBlock
```

## 5. 전역 변수 치환

```yaml
num_classes: 10
layers:
  - [ -1, 1, LinearBlock, num_classes ]   # ⇒ out_features = 10
```

## 6. 예시 – VGG16 모델

```yaml
# vgg16.yaml - MNIST용 VGG16 모델
in_channels: 1
num_classes: 10
depth_multiple: 1.0
width_multiple: 0.5   # 채널 절반으로 축소

layers:
  # Block 1
  - [-1, 1, ConvBlock, 64, 3, 1, {act: relu, norm: bn}]
  - [-1, 1, ConvBlock, 64, 3, 1, {act: relu, norm: bn}]
  - [-1, 1, ConvBlock, 64, 2, 2, {act: relu}]

  # Block 2
  - [-1, 1, ConvBlock, 128, 3, 1, {act: relu, norm: bn}]
  - [-1, 1, ConvBlock, 128, 3, 1, {act: relu, norm: bn}]
  - [-1, 1, ConvBlock, 128, 2, 2, {act: relu}]

  # 특성 평탄화 (Flatten)
  - [-1, 1, GlobalAvgPool, 0]

  # 완전 연결 레이어
  - [-1, 1, LinearBlock, 512, 0, 0, {act: relu, drop: 0.5}]
  - [-1, 1, LinearBlock, 10, 0, 0, {}]
```

## 7. 고급 팁

| 목적 | 방법 |
|------|------|
| 모델 크기 조정 | depth_multiple, width_multiple 값 조정 |
| Dropout 비활성화 | drop: 0 또는 생략 |
| BatchNorm 없는 Conv | {norm: none} |
| LayerNorm 사용 | {norm: ln} |
| 다양한 활성화 함수 | 반복 블록을 별도 레이어로 선언하고 각각 다른 act 지정 |

## 8. 자주 발생하는 오류 & 해결법

| 오류 | 원인 / 해결 |
|------|------------|
| 차원 불일치 오류 | 레이어 간 입출력 채널 수 확인, 특히 GlobalAvgPool 후 차원 확인 |
| 블록 레지스트리 오류 | YAML에 명시된 블록 이름이 실제 등록된 클래스와 일치하는지 확인 |
| 형상 변환 문제 | Conv와 Linear 연결 시 차원 계산 문제, reshape 필요 여부 확인 |

## 9. 모델 빌드 & 테스트 코드

```python
import numpy as np
import flexo
from flexo.autobuilder import YamlModel

# 모델 생성
model = YamlModel("configs/vgg16.yaml")

# 테스트 입력
x = flexo.Variable(np.random.randn(1, 1, 28, 28).astype(np.float32))

# 순전파
y = model(x)
print(f"출력 형상: {y.shape}")  # 예상 출력: (1, 10)

# 모델 학습
optimizer = flexo.optimizers.Adam().setup(model)
# ... 학습 루프 구현 ...
```

이 가이드를 참고하여 다양한 모델을 자유롭게 설계해 보세요!

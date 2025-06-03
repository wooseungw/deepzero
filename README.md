# Torch YAML Model Zoo

> **유연한 PyTorch 모델 빌더 & 트레이너 – YAML 기반**
> Python 코드를 직접 수정하지 않고도 커스텀 신경망을 정의·학습·공유할 수 있습니다.

---

## ✨ 주요 특징

| 기능                 | 설명                                                                                      |
| ------------------ | --------------------------------------------------------------------------------------- |
| **YAML 우선 설계**     | 단일 YAML 파일에 네트워크를 기술합니다 ‑ 레이어 순서, 반복 횟수, 스킵/컨캣 연결, 활성화 함수, 정규화, 드롭아웃까지 모두 설정 가능         |
| **YOLO 스타일 문법**    | `[from, n, module, *args, {kwargs}]` 형태의 직관적인 튜플 문법( YOLOv5/8 계열과 동일 )                  |
| **동적 스케일링**        | `depth_multiple`, `width_multiple` 로 반복 수와 채널 수를 전역 비율로 자동 조정                           |
| **플러그인 모듈**        | ConvBlock · LinearBlock · TransformerEncoderBlock 등 기본 블록 제공 + 간단한 등록 API로 사용자 정의 블록 추가 |
| **잔차·Concat 그래프**  | **from** 에 음수 인덱스나 리스트를 사용하여 스킵 연결과 멀티 브랜치 컨캣 자유롭게 구성                                   |
| **즉시 사용 가능한 트레이너** | `train.py`, `val.py` CLI 한 줄로 학습·평가 (Mixed Precision, LR 스케줄러, EMA, 체크포인트 포함)           |
| **배포 지원**          | TorchScript·ONNX 내보내기 단일 명령 지원                                                          |

---

## 📦 설치

**Python ≥ 3.9** 및 **PyTorch ≥ 2.2** 필요

```bash
# 레포 클론
$ git clone https://github.com/<your-org>/torch-yaml-model-zoo.git
$ cd torch-yaml-model-zoo

# (선택) 가상환경 생성/활성화
$ python -m venv .venv && source .venv/bin/activate

# 필수 + 추가 패키지 설치( Lightning, tqdm 등 )
$ pip install -r requirements.txt
```

---

## 🚀 빠른 시작

1. **모델 설정파일** 작성 (또는 `configs/` 샘플 활용)
2. **데이터셋 준비** (COCO‑style YAML, ImageFolder, 혹은 사용자 정의 `Dataset`)
3. **학습**

   ```bash
   python train.py \
       --cfg configs/example_vit.yaml \
       --data data/coco.yaml \
       --epochs 300 \
       --device 0,1
   ```
4. **검증 / 테스트**

   ```bash
   python val.py --weights runs/exp/weights/best.pt --data data/coco.yaml
   ```
5. **추론**

   ```python
   from zoo import load_model
   model = load_model('runs/exp/weights/best.pt').eval()
   preds = model(img)  # BCHW 텐서
   ```

---

## 📝 YAML 스펙

```yaml
# example_vit.yaml

# ── 메타 ─────────────────────────────────────
model: MyHybridNet           # (선택) 모델 이름(가독성용)
in_channels: 3
num_classes: 100
# 스케일링
depth_multiple: 1.0          # 반복 횟수 전역 배수
width_multiple: 1.0          # 채널 수   전역 배수

# ── 레이어 정의 ──────────────────────────────
layers:
  # [from, repeat, module,          args..., {kwargs}]
  - [ -1, 1, ConvBlock,      64, 3, 2, {drop: 0.1} ]
  - [ -1, 1, ConvBlock,     128, 3, 2, {act: relu, norm: ln} ]
  - [ -1, 1, TransformerEncoderBlock, 128, 4, 8 ]
  - [ -1, 1, ConvBlock,     256, 3, 2 ]
  - [ -1, 1, GlobalAvgPool ]
  - [ -1, 1, LinearBlock, 1024, silu ]
  - [ -1, 1, LinearBlock, num_classes ]
```

### 필드 설명

| 필드         | 타입                   | 의미                                             |
| ---------- | -------------------- | ---------------------------------------------- |
| **from**   | `int` \| `list[int]` | 입력 인덱스(음수 = 현재 위치 기준). 리스트는 concat 입력을 의미      |
| **repeat** | `int`                | 블록 반복 횟수 (`depth_multiple`로 스케일 조정)            |
| **module** | `str`                | `core/layers.py` 에 등록된 클래스 이름                  |
| **args…**  | mixed                | 모듈로 전달되는 위치 인자. 채널 관련 값은 `width_multiple`로 스케일 |
| **kwargs** | dict                 | 선택적 키워드 인자 (활성화 `act`, 정규화 `norm`, `drop` 등)   |

#### 활성화 함수(`act`) 예시

`silu`, `relu`, `gelu`, `leaky`, `tanh`, `sigmoid` 또는 사용자 정의 callable

#### 정규화(`norm`) 옵션

`bn`(BatchNorm), `gn`(GroupNorm), `ln`(LayerNorm), `in`(InstanceNorm), `none`

#### 드롭아웃(`drop`)

0≤ 실수 <1.0. **ConvBlock**, **LinearBlock** 에 적용

---

## 🔧 커스텀 레이어 추가하기

1. `deepzero/core/layers.py` 에 새 클래스 작성

```python
@register_module
autograd.no_grad()  # 필요 시
class MyAwesomeBlock(nn.Module):
    def __init__(self, c_in, c_out, *args, **kwargs):
        super().__init__()
        # 레이어 구성 …
    def forward(self, x):
        return x + 42  # 멋진 동작
```

2. YAML에서 사용

```yaml
- [ -1, 1, MyAwesomeBlock, 256, 256 ]
```

`build_model_from_yaml()` 가 자동으로 찾아 빌드합니다.

---

## 📂 프로젝트 구조

```
├─ core/              # 모델 빌더 & 레이어 정의
│  ├─ model_builder.py
│  └─ layers.py
├─ configs/           # YAML 모델 설정
├─ data/              # 데이터셋 YAML / 스크립트
├─ train.py           # 학습 엔트리포인트
├─ val.py             # 평가 스크립트
└─ utils/             # 시각화, 메트릭 등
```

---

## 📅 로드맵

* [x] 기본 YOLO 스타일 파서
* [x] Conv/Linear/Transformer 블록
* [x] 시각화 도구
* [ ] 사전학습 모델 지원
* [ ] 모델 python 코드로 변환



---

## 📜 라이선스

본 프로젝트는 **Apache-2.0** 라이선스로 배포됩니다. 상세 내용은 [LICENSE](LICENSE) 파일 참조.

---

## 🙌 Acknowledgements

ultralytics/YOLOv8, PyTorch Lightning, 그리고 수많은 오픈소스 기여자에게 영감을 받았습니다.

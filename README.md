# DeepZero

`deepzero`는 CUDA와 NumPy를 기반으로, Python `@dataclass`를 활용하여 딥러닝 모델을 작성하고 학습 및 검증까지 실행할 수 있는 경량 프레임워크입니다. `dezero`의 설계 철학을 계승하면서, 명확한 설정 파일(YAML) 구조와 모듈화된 코드 베이스를 제공합니다.

---

## 📦 주요 기능

- **모델 정의**: Python `@dataclass`로 간결하고 직관적인 모델 구성
- **YAML 기반 설정**: `train.yaml`과 `model.yaml` 파일만으로 학습/검증 파이프라인 자동 구성
- **CUDA 가속**: CuPy(PyCUDA)와 NumPy 연산 지원으로 GPU 연산 최적화
- **학습·검증 스크립트**: 단일 진입점(`train.py`, `validate.py`)으로 빠른 실험 반복
- **로깅 & 체크포인트**: TensorBoard 호환 로그, 주기적 모델 저장

---

## 📂 프로젝트 구조

```
deepzero/               # 최상위 패키지
├── core/               # 핵심 모듈 (엔진, 옵티마이저, 손실 함수 등)
│   ├── engine.py       # 학습 및 검증 엔진
│   ├── optim.py        # Optimizer 구현체
│   └── losses.py       # Loss 함수 모음
├── models/             # 사용자 정의 모델 템플릿
│   └── example_model.py# 샘플 모델 구현 (dataclass 기반)
├── utils/              # 유틸리티 모듈
│   ├── yaml_loader.py  # YAML 설정 로더
│   └── logger.py       # 로깅 및 체크포인트 관리
├── configs/            # 기본 설정 파일
│   ├── train.yaml      # 학습 파라미터 (에포크, 학습률 등)
│   └── model.yaml      # 모델 구조 및 하이퍼파라미터
├── scripts/            # 도움 스크립트 (환경설정, 배포 등)
│   └── setup_env.sh    # Conda/venv 환경 생성 예시
├── requirements.txt    # Python 패키지 의존성 목록
├── train.py            # 학습 실행 스크립트
├── validate.py         # 검증 실행 스크립트
└── README.md           # 프로젝트 설명 (이 파일)
```

---

## 🛠️ 설치 및 환경 설정

아래 예시는 Conda 환경을 기준으로 설명합니다.

1. **Conda 환경 생성 및 활성화**  
   ```bash
   conda create -n deepzero python=3.10 -y
   conda activate deepzero
   ```

2. **의존성 설치**  
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

3. **CUDA Toolkit 확인**  
   - 시스템에 알맞은 CUDA Toolkit (예: 11.x 이상) 설치 및 `nvcc --version`으로 버전 확인
   - CuPy 설치 예 (CUDA 11.x):  
     ```bash
     pip install cupy-cuda11x
     ```

---

## 📑 requirements.txt 예시

```
numpy>=1.23
cupy>=11.0         # GPU 가속 연산
pyyaml>=6.0        # 설정 파일 파싱
tqdm>=4.64         # 진행바
tensorboard>=2.12  # 로그 시각화
```

---

## 🚀 사용법

### 1. 학습 실행

```bash
python train.py \
  --config configs/train.yaml \
  --model-config configs/model.yaml
```

- `--config`: 학습 파라미터(YAML)
- `--model-config`: 모델 구조 및 하이퍼파라미터(YAML)

### 2. 검증 실행

학습과 동일한 방식으로, `validate.py`를 사용하여 검증만 수행할 수 있습니다.

```bash
python validate.py \
  --config configs/train.yaml \
  --model-config configs/model.yaml
```

- 자동으로 학습된 체크포인트를 로드하여 검증 진행

---

## ⚙️ 설정 파일 (YAML) 설명

### configs/train.yaml

```yaml
training:
  epochs: 50
  batch_size: 32
  learning_rate: 0.001
  checkpoint_interval: 5
  log_dir: './logs'
```

### configs/model.yaml

```yaml
model:
  name: ExampleModel
  input_dim: 3,224,224
  hidden_dim: 512
  num_classes: 10
  dropout: 0.2
```

- **training**: 학습 루프 관련 파라미터
- **model**: `@dataclass` 기반 모델 초기화 인자

---

## 📄 기여

1. Fork 후 브랜치 생성 (`git checkout -b feature/your-feature`)  
2. 코드 작성 및 테스트  
3. Pull Request 생성  
4. 리뷰 및 병합

---

## 📜 라이선스

MIT License

---

*Happy Coding!*


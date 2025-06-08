# ──────────────────────────────────────────────────────────────────────────────
# 예제 스크립트: config/model.yaml을 로드해서 YOLOModel 생성 후 더미 입력으로 테스트
# ──────────────────────────────────────────────────────────────────────────────
import numpy as np
from flexo.autobuilder import YamlModel
from flexo import display_model_info
from flexo import Variable

def main():
    # 1) YAML 파일 경로 지정
    config_path = "configs/vgg16.yaml"

    # 2) 모델 생성
    model = YamlModel(config_path)
    
    display_model_info(config_path)  # 모델 정보 출력

    # 3) 더미 입력 (배치 크기 4, 채널 1, 28x28 이미지)
    # Dezero에선 NCHW 형태의 numpy 배열을 바로 넣어도 자동으로 Variable로 래핑
    x = np.random.randn(4, 1, 28, 28).astype(np.float32)
    # 시각화
    model.plot(x)
    # 4) 순전파 출력 확인
    y = model(x)
    print("Output shape:", y.shape)  # → (4, num_classes)

    # 5) 간단한 손실계산 예시 (CrossEntropy)
    #    d := 타깃 레이블 (0~num_classes-1 중 난수 생성)
    targets = np.random.randint(0, model.num_classes, size=(4,))
    loss = model.cleargrads();  # 클리어해두고
    out = model(x)             # 순전파
    import flexo.functions as F
    import flexo.layers as L

    # Dezero에서는 softmax_cross_entropy에 레이블(정수) 바로 넣을 수 있음
    loss = F.softmax_cross_entropy(out, targets)
    model.cleargrads()
    loss.backward()
    # (이후 옵티마이저에 넘겨서 학습하면 됨)

    print("Sample loss:", float(loss.data))

if __name__ == "__main__":
    main()
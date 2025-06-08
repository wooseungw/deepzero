import flexo
from flexo.autobuilder import YamlModel

def display_model_info(yaml_path):
    """
    YAML 파일로부터 모델을 로드하고 레이어 구조를 분석합니다.
    """
    # 모델 로드
    model = YamlModel(yaml_path)
    
    # 기본 정보 출력
    print(f"Model from: {yaml_path}")
    print(f"Input channels: {model.in_channels}")
    print(f"Number of classes: {model.num_classes}")
    print(f"Depth multiplier: {model.depth_mul}")
    print(f"Width multiplier: {model.width_mul}")
    print(f"Total layers: {len(model.layers)}")
    print("\n" + "=" * 50 + "\n")
    
    # 레이어 상세 정보 출력
    print("Layer Structure:")
    print("-" * 50)
    print(f"{'Index':<6}{'From':<6}{'Type':<20}{'Shape/Params':<30}")
    print("-" * 50)
    
    # 가상의 입력 생성 (MNIST 기준 [1, 1, 28, 28])
    import numpy as np
    x = flexo.Variable(np.zeros((1, model.in_channels, 28, 28), dtype=np.float32))
    outputs = [x]
    
    # 각 레이어 실행하며 출력 형태 추적
    for i, layer_info in enumerate(model.layers):
        f_idx = layer_info['f']
        block = layer_info['block']
        block_type = block.__class__.__name__
        
        # 입력 가져오기
        h_prev = outputs[f_idx] if f_idx != -1 else outputs[-1]
        
        # 레이어 통과
        h_out = block(h_prev)
        outputs.append(h_out)
        
        # 파라미터 수 계산 (근사치)
        params = 0
        if hasattr(block, 'conv'):
            if hasattr(block.conv, 'W'):
                params += np.prod(block.conv.W.shape)
            if hasattr(block.conv, 'b') and block.conv.b is not None:
                params += np.prod(block.conv.b.shape)
        elif hasattr(block, 'linear'):
            if hasattr(block.linear, 'W'):
                params += np.prod(block.linear.W.shape)
            if hasattr(block.linear, 'b') and block.linear.b is not None:
                params += np.prod(block.linear.b.shape)
        
        # 레이어 정보 출력
        print(f"{i:<6}{f_idx:<6}{block_type:<20}{str(h_out.shape):<20}{params:,} params")
    
    print("-" * 50)
    
    # 총 파라미터 수 추정
    total_params = 0
    for layer_info in model.layers:
        block = layer_info['block']
        if hasattr(block, 'conv'):
            if hasattr(block.conv, 'W'):
                total_params += np.prod(block.conv.W.shape)
            if hasattr(block.conv, 'b') and block.conv.b is not None:
                total_params += np.prod(block.conv.b.shape)
        elif hasattr(block, 'linear'):
            if hasattr(block.linear, 'W'):
                total_params += np.prod(block.linear.W.shape)
            if hasattr(block.linear, 'b') and block.linear.b is not None:
                total_params += np.prod(block.linear.b.shape)
    
    print(f"\nTotal parameters: {total_params:,}")

if __name__ == "__main__":
    display_model_info("configs/vgg16.yaml")

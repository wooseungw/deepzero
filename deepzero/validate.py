#!/usr/bin/env python3
"""
DeepZero Validation Script
학습된 모델의 성능 평가
"""

import argparse
import sys
import os
import torch
from torch.utils.data import DataLoader, TensorDataset

# deepzero 패키지 경로 추가
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from deepzero.core.engine import ValidationEngine
from deepzero.utils.yaml_loader import load_yaml


def create_dummy_data(num_samples=200, input_shape=(3, 32, 32), num_classes=10):
    """테스트용 더미 데이터 생성"""
    X = torch.randn(num_samples, *input_shape)
    y = torch.randint(0, num_classes, (num_samples,))
    return TensorDataset(X, y)


def main():
    # 인자 파싱
    parser = argparse.ArgumentParser(
        description='DeepZero Validation Script - Evaluate trained models'
    )
    parser.add_argument(
        '--model-config', 
        type=str, 
        required=True,
        help='Path to the model configuration YAML file'
    )
    parser.add_argument(
        '--checkpoint',
        type=str,
        required=True,
        help='Path to the model checkpoint to validate'
    )
    parser.add_argument(
        '--data-path',
        type=str,
        default=None,
        help='Path to the validation dataset'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=32,
        help='Batch size for validation'
    )
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Path to save validation results (JSON format)'
    )
    
    args = parser.parse_args()
    
    # 모델 설정 로드
    print(f"Loading model config from: {args.model_config}")
    model_config = load_yaml(args.model_config)
    
    # ValidationEngine 초기화
    val_engine = ValidationEngine(model_config)
    
    # 데이터 로드
    if args.data_path:
        # TODO: 실제 데이터셋 로딩 로직 구현
        print(f"Loading validation data from: {args.data_path}")
        raise NotImplementedError("Custom data loading not yet implemented")
    else:
        print("Using dummy data for demonstration")
        # 모델 설정에서 입력 차원과 클래스 수 추출
        if 'input_dim' in model_config.get('model', model_config):
            input_dim = model_config.get('model', model_config)['input_dim']
            if isinstance(input_dim, str):
                input_shape = tuple(map(int, input_dim.split(',')))
            else:
                input_shape = (3, 32, 32)  # 기본값
        else:
            input_shape = (3, 32, 32)
            
        num_classes = model_config.get('model', model_config).get('num_classes', 10)
        
        # 더미 검증 데이터셋 생성
        val_dataset = create_dummy_data(200, input_shape, num_classes)
        
        # 데이터 로더 생성
        val_loader = DataLoader(
            val_dataset, 
            batch_size=args.batch_size, 
            shuffle=False,
            num_workers=0
        )
    
    # 검증 실행
    print(f"\nValidating model from: {args.checkpoint}")
    print("=" * 50)
    
    results = val_engine.validate(val_loader, args.checkpoint)
    
    print("=" * 50)
    print("\nValidation Results:")
    print(f"  Loss: {results['loss']:.4f}")
    print(f"  Accuracy: {results['accuracy']:.2f}%")
    
    # 결과 저장 (선택사항)
    if args.output:
        import json
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to: {args.output}")
    
    # 추가 메트릭 계산 (옵션)
    print("\nAdditional metrics can be implemented:")
    print("  - Confusion matrix")
    print("  - Per-class accuracy")
    print("  - Precision, Recall, F1-score")
    print("  - ROC curves")


if __name__ == '__main__':
    main()
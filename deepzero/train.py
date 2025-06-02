#!/usr/bin/env python3
"""
DeepZero Training Script
YAML 설정 기반 유연한 모델 학습
"""

import argparse
import sys
import os

# deepzero 패키지 경로 추가
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from deepzero.core.engine import Engine
from deepzero.utils.yaml_loader import load_yaml
from torch.utils.data import DataLoader, TensorDataset
import torch


def create_dummy_data(num_samples=1000, input_shape=(3, 32, 32), num_classes=10):
    """테스트용 더미 데이터 생성"""
    X = torch.randn(num_samples, *input_shape)
    y = torch.randint(0, num_classes, (num_samples,))
    return TensorDataset(X, y)


def main():
    # 인자 파싱
    parser = argparse.ArgumentParser(
        description='DeepZero Training Script - Train models using YAML configurations'
    )
    parser.add_argument(
        '--config', 
        type=str, 
        default='deepzero/configs/train.yaml',
        help='Path to the training configuration YAML file'
    )
    parser.add_argument(
        '--model-config', 
        type=str, 
        default='deepzero/configs/model.yaml',
        help='Path to the model configuration YAML file'
    )
    parser.add_argument(
        '--data-path',
        type=str,
        default=None,
        help='Path to the dataset (optional, will use dummy data if not provided)'
    )
    parser.add_argument(
        '--resume',
        type=str,
        default=None,
        help='Path to checkpoint to resume training from'
    )
    parser.add_argument(
        '--validate-only',
        action='store_true',
        help='Only run validation with the loaded model'
    )
    
    args = parser.parse_args()
    
    # 설정 파일 로드
    print(f"Loading training config from: {args.config}")
    train_config = load_yaml(args.config)
    
    print(f"Loading model config from: {args.model_config}")
    model_config = load_yaml(args.model_config)
    
    # Engine 초기화
    engine = Engine(train_config, model_config)
    
    # 체크포인트에서 재개
    if args.resume:
        print(f"Resuming from checkpoint: {args.resume}")
        engine.load_checkpoint(args.resume)
    
    # 데이터 로드 (실제 구현에서는 사용자 데이터셋 로더 구현 필요)
    if args.data_path:
        # TODO: 실제 데이터셋 로딩 로직 구현
        print(f"Loading data from: {args.data_path}")
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
        
        # 더미 데이터셋 생성
        train_dataset = create_dummy_data(1000, input_shape, num_classes)
        val_dataset = create_dummy_data(200, input_shape, num_classes)
        
        # 데이터 로더 생성
        batch_size = train_config['training']['batch_size']
        train_loader = DataLoader(
            train_dataset, 
            batch_size=batch_size, 
            shuffle=True,
            num_workers=0
        )
        val_loader = DataLoader(
            val_dataset, 
            batch_size=batch_size, 
            shuffle=False,
            num_workers=0
        )
    
    # 검증만 수행
    if args.validate_only:
        print("Running validation only...")
        val_loss = engine._validate(val_loader)
        print(f"Validation complete. Loss: {val_loss:.4f}")
    else:
        # 학습 시작
        print("\nStarting training...")
        print("=" * 50)
        engine.train(train_loader, val_loader)
        print("=" * 50)
        print("Training complete!")
        
        # 최종 모델 저장
        final_checkpoint_path = os.path.join(engine.checkpoint_dir, 'final_model.pth')
        engine.save_checkpoint(engine.current_epoch - 1, is_best=False)
        print(f"Final model saved to: {final_checkpoint_path}")


if __name__ == '__main__':
    main()
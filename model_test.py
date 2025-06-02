#!/usr/bin/env python3
"""
새로운 ModelBuilder 테스트
YOLOv5/v8 스타일 모델 정의 테스트
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import yaml
from deepzero.models.model_builder import ModelBuilder


def test_basic_model():
    """기본 모델 생성 테스트"""
    print("Testing basic model creation...")
    
    config = {
        'name': 'TestModel',
        'in_channels': 3,
        'num_classes': 10,
        'depth_multiple': 1.0,
        'width_multiple': 1.0,
        'layers': [
            [-1, 1, 'ConvBlock', 64, 3, 2, {'drop': 0.1}],
            [-1, 1, 'ConvBlock', 128, 3, 2],
            [-1, 1, 'GlobalAvgPool'],
            [-1, 1, 'LinearBlock', 'num_classes']
        ]
    }
    
    builder = ModelBuilder(**config)
    model = builder.build_model()
    
    # 순전파 테스트
    x = torch.randn(2, 3, 32, 32)
    output = model(x)
    
    assert output.shape == torch.Size([2, 10])
    print("✓ Basic model test passed")
    
    # 모델 구조 출력
    model.summary()


def test_transformer_model():
    """Transformer 포함 모델 테스트"""
    print("\nTesting transformer model...")
    
    config = {
        'name': 'TransformerTest',
        'in_channels': 3,
        'num_classes': 100,
        'layers': [
            [-1, 1, 'ConvBlock', 128, 3, 2],
            [-1, 1, 'TransformerEncoderBlock', 128, 4, 2],  # d_model=128, nhead=4, layers=2
            [-1, 1, 'ConvBlock', 256, 3, 2],
            [-1, 1, 'GlobalAvgPool'],
            [-1, 1, 'LinearBlock', 'num_classes']
        ]
    }
    
    builder = ModelBuilder(**config)
    model = builder.build_model()
    
    x = torch.randn(2, 3, 64, 64)
    output = model(x)
    
    assert output.shape == torch.Size([2, 100])
    print("✓ Transformer model test passed")


def test_depth_width_multiple():
    """Depth/Width multiplier 테스트"""
    print("\nTesting depth/width multipliers...")
    
    config = {
        'name': 'ScaledModel',
        'in_channels': 3,
        'num_classes': 10,
        'depth_multiple': 0.5,  # 레이어 반복 횟수 절반
        'width_multiple': 0.5,  # 채널 수 절반
        'layers': [
            [-1, 4, 'ConvBlock', 64, 3, 1],  # 4개 반복 -> 2개
            [-1, 1, 'ConvBlock', 128, 3, 2],  # 128 채널 -> 64 채널
            [-1, 1, 'GlobalAvgPool'],
            [-1, 1, 'LinearBlock', 'num_classes']
        ]
    }
    
    builder = ModelBuilder(**config)
    model = builder.build_model()
    
    # 첫 번째 레이어가 2개의 ConvBlock으로 구성되었는지 확인
    assert len(model.layers[0]) == 2
    
    # 두 번째 레이어의 출력 채널이 64인지 확인 (128 * 0.5)
    conv_layer = model.layers[1]
    if hasattr(conv_layer, 'conv'):
        assert conv_layer.conv.out_channels == 64
    
    print("✓ Depth/width multiplier test passed")


def test_complex_model():
    """복잡한 모델 구조 테스트"""
    print("\nTesting complex model architecture...")
    
    config = {
        'name': 'ComplexModel',
        'in_channels': 3,
        'num_classes': 1000,
        'layers': [
            # CNN backbone
            [-1, 1, 'ConvBlock', 64, 7, 2, {'act': 'relu', 'norm': 'bn'}],
            [-1, 1, 'MaxPool', 3, 2, 1],
            [-1, 3, 'Bottleneck', 64, 1],
            [-1, 4, 'Bottleneck', 128, 2],
            
            # Transformer processing
            [-1, 2, 'TransformerEncoderBlock', 512, 8],
            
            # Output
            [-1, 1, 'GlobalAvgPool'],
            [-1, 1, 'LinearBlock', 2048, 'relu', {'drop': 0.5}],
            [-1, 1, 'LinearBlock', 'num_classes']
        ]
    }
    
    builder = ModelBuilder(**config)
    model = builder.build_model()
    
    x = torch.randn(1, 3, 224, 224)
    output = model(x)
    
    assert output.shape == torch.Size([1, 1000])
    print("✓ Complex model test passed")


def test_from_yaml():
    """YAML 파일에서 모델 로드 테스트"""
    print("\nTesting model loading from YAML...")
    
    yaml_content = """
name: YAMLModel
in_channels: 3
num_classes: 100
depth_multiple: 1.0
width_multiple: 1.0

layers:
  - [-1,
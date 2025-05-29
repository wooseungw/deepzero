import pytest
import torch
from deepzero.models.model_builder import ModelBuilder

import torch.nn as nn


def test_model_builder_initialization():
    """ModelBuilder 초기화 테스트"""
    builder = ModelBuilder(
        name="test_model",
        layers=[['linear', 10, 5]],
        input_dim=(10,),
        num_classes=5
    )
    
    if builder.name != "test_model":
        assert False, "Name initialization failed"
    if builder.input_dim != (10,):
        assert False, "Input dimension initialization failed"
    if builder.num_classes != 5:
        assert False, "Number of classes initialization failed"


def test_conv_layer_building():
    """Conv 레이어 생성 테스트"""
    builder = ModelBuilder(name="conv_test", layers=[])
    layer = builder._build_conv_layer(3, 64, 3, 1, 1)
    
    if not isinstance(layer, nn.Conv2d):
        assert False, "Conv layer type mismatch"
    if layer.in_channels != 3:
        assert False, "Conv layer input channels mismatch"
    if layer.out_channels != 64:
        assert False, "Conv layer output channels mismatch"


def test_linear_layer_building():
    """Linear 레이어 생성 테스트"""
    builder = ModelBuilder(name="linear_test", layers=[])
    layer = builder._build_linear_layer(128, 64)
    
    if not isinstance(layer, nn.Linear):
        assert False, "Linear layer type mismatch"
    if layer.in_features != 128:
        assert False, "Linear layer input features mismatch"
    if layer.out_features != 64:
        assert False, "Linear layer output features mismatch"


def test_lstm_layer_building():
    """LSTM 레이어 생성 테스트"""
    builder = ModelBuilder(name="lstm_test", layers=[])
    layer = builder._build_lstm_layer(100, 50, 2)
    
    if not isinstance(layer, nn.LSTM):
        assert False, "LSTM layer type mismatch"
    if layer.input_size != 100:
        assert False, "LSTM input size mismatch"
    if layer.hidden_size != 50:
        assert False, "LSTM hidden size mismatch"
    if layer.num_layers != 2:
        assert False, "LSTM num layers mismatch"


def test_transformer_layer_building():
    """Transformer 레이어 생성 테스트"""
    builder = ModelBuilder(name="transformer_test", layers=[])
    layer = builder._build_transformer_layer(512, 8, 6)
    
    if not isinstance(layer, nn.TransformerEncoder):
        assert False, "Transformer layer type mismatch"


def test_batchnorm_layer_building():
    """BatchNorm 레이어 생성 테스트"""
    builder = ModelBuilder(name="bn_test", layers=[])
    layer = builder._build_batchnorm_layer(64)
    
    if not isinstance(layer, nn.BatchNorm2d):
        assert False, "BatchNorm layer type mismatch"
    if layer.num_features != 64:
        assert False, "BatchNorm num features mismatch"


def test_parse_layer_config_conv():
    """레이어 설정 파싱 테스트 - Conv"""
    builder = ModelBuilder(name="parse_test", layers=[])
    layer_config = ['conv', 3, 64, 3, 1, 1]
    layer = builder._parse_layer_config(layer_config)
    
    if not isinstance(layer, nn.Conv2d):
        assert False, "Parsed conv layer type mismatch"


def test_parse_layer_config_relu():
    """레이어 설정 파싱 테스트 - ReLU"""
    builder = ModelBuilder(name="parse_test", layers=[])
    layer_config = ['relu']
    layer = builder._parse_layer_config(layer_config)
    
    if not isinstance(layer, nn.ReLU):
        assert False, "Parsed ReLU layer type mismatch"


def test_parse_layer_config_invalid():
    """잘못된 레이어 설정 파싱 테스트"""
    builder = ModelBuilder(name="parse_test", layers=[])
    layer_config = ['invalid_layer']
    
    try:
        builder._parse_layer_config(layer_config)
        assert False, "Should have raised ValueError for invalid layer"
    except ValueError as e:
        if "Unsupported layer type" not in str(e):
            assert False, "Wrong error message for invalid layer"


def test_build_model_list_format():
    """리스트 형태 레이어로 모델 생성 테스트"""
    layers = [
        ['linear', 10, 5],
        ['relu'],
        ['linear', 5, 1]
    ]
    builder = ModelBuilder(name="list_model", layers=layers)
    model = builder.build_model()
    
    if not isinstance(model, nn.Sequential):
        assert False, "Built model should be Sequential"
    if len(model) != 3:
        assert False, f"Expected 3 layers, got {len(model)}"


def test_build_model_dict_format():
    """딕셔너리 형태 레이어로 모델 생성 테스트"""
    layers = {
        "encoder": [['linear', 10, 20], ['relu']],
        "decoder": [['linear', 20, 5]]
    }
    builder = ModelBuilder(name="dict_model", layers=layers)
    model = builder.build_model()
    
    if not isinstance(model, nn.Sequential):
        assert False, "Built model should be Sequential"
    if len(model) != 3:
        assert False, f"Expected 3 layers, got {len(model)}"


def test_build_model_dict_single_layer():
    """딕셔너리에서 단일 레이어 형태 테스트"""
    layers = {
        "layer1": ['linear', 10, 5]
    }
    builder = ModelBuilder(name="single_layer_model", layers=layers)
    model = builder.build_model()
    
    if not isinstance(model, nn.Sequential):
        assert False, "Built model should be Sequential"
    if len(model) != 1:
        assert False, f"Expected 1 layer, got {len(model)}"


def test_model_forward_pass():
    """모델 순전파 테스트"""
    layers = [
        ['linear', 10, 5],
        ['relu'],
        ['linear', 5, 1]
    ]
    builder = ModelBuilder(name="forward_test", layers=layers)
    model = builder.build_model()
    
    # 테스트 입력 생성
    x = torch.randn(2, 10)
    output = model(x)
    
    if output.shape != torch.Size([2, 1]):
        assert False, f"Expected output shape [2, 1], got {output.shape}"


def test_summary_method():
    """summary 메서드 테스트"""
    builder = ModelBuilder(
        name="summary_test",
        layers=[['linear', 10, 5]],
        input_dim=(10,),
        num_classes=5
    )
    
    # summary는 print만 하므로 에러가 발생하지 않으면 성공
    try:
        builder.summary()
    except Exception as e:
        assert False, f"Summary method failed: {e}"


if __name__ == "__main__":
    test_model_builder_initialization()
    test_conv_layer_building()
    test_linear_layer_building()
    test_lstm_layer_building()
    test_transformer_layer_building()
    test_batchnorm_layer_building()
    test_parse_layer_config_conv()
    test_parse_layer_config_relu()
    test_parse_layer_config_invalid()
    test_build_model_list_format()
    test_build_model_dict_format()
    test_build_model_dict_single_layer()
    test_model_forward_pass()
    test_summary_method()
    print("All tests passed!")
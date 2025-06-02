import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import yaml
import tempfile
import os
import sys

# 프로젝트 루트를 Python 경로에 추가
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from deepzero.models.model_builder import ModelBuilder, DynamicModel
from deepzero.core.engine import Engine, ValidationEngine
from deepzero.utils.yaml_loader import load_yaml, save_yaml


class TestModelBuilder:
    """ModelBuilder 테스트"""
    
    def test_simple_cnn_creation(self):
        """간단한 CNN 모델 생성 테스트"""
        config = {
            'name': 'TestCNN',
            'input_shape': '3,32,32',
            'layers': [
                {'type': 'conv2d', 'in_channels': 3, 'out_channels': 16, 'kernel_size': 3, 'padding': 1},
                {'type': 'relu'},
                {'type': 'maxpool2d', 'kernel_size': 2},
                {'type': 'flatten'},
                {'type': 'linear', 'in_features': 4096, 'out_features': 10}
            ]
        }
        
        builder = ModelBuilder(**config)
        model = builder.build_model()
        
        assert isinstance(model, DynamicModel)
        assert len(model.layers) == 5
        
        # 순전파 테스트
        x = torch.randn(2, 3, 32, 32)
        output = model(x)
        assert output.shape == torch.Size([2, 10])
    
    def test_lstm_model_creation(self):
        """LSTM 모델 생성 테스트"""
        config = {
            'name': 'TestLSTM',
            'layers': [
                {'type': 'lstm', 'input_size': 100, 'hidden_size': 64, 'num_layers': 2, 'return_last': True},
                {'type': 'linear', 'in_features': 64, 'out_features': 5}
            ]
        }
        
        builder = ModelBuilder(**config)
        model = builder.build_model()
        
        # 순전파 테스트
        x = torch.randn(4, 10, 100)  # batch, seq_len, features
        output = model(x)
        assert output.shape == torch.Size([4, 5])
    
    def test_transformer_model_creation(self):
        """Transformer 모델 생성 테스트"""
        config = {
            'name': 'TestTransformer',
            'layers': [
                {'type': 'transformer', 'd_model': 128, 'nhead': 8, 'num_layers': 2},
                {'type': 'linear', 'in_features': 128, 'out_features': 2}
            ]
        }
        
        builder = ModelBuilder(**config)
        model = builder.build_model()
        
        # 순전파 테스트
        x = torch.randn(2, 20, 128)  # batch, seq_len, d_model
        output = model(x)
        assert output.shape == torch.Size([2, 2])
    
    def test_model_summary(self):
        """모델 요약 테스트"""
        config = {
            'name': 'SummaryTest',
            'layers': [
                {'type': 'linear', 'in_features': 10, 'out_features': 5}
            ]
        }
        
        builder = ModelBuilder(**config)
        summary = builder.summary()
        
        assert 'Model: SummaryTest' in summary
        assert 'linear' in summary


class TestEngine:
    """Engine 테스트"""
    
    def setup_method(self):
        """테스트 설정"""
        self.temp_dir = tempfile.mkdtemp()
        
        # 간단한 데이터셋 생성
        X = torch.randn(100, 10)
        y = torch.randint(0, 2, (100,))
        self.dataset = TensorDataset(X, y)
        
        # 기본 설정
        self.train_config = {
            'training': {
                'epochs': 2,
                'batch_size': 10,
                'learning_rate': 0.01,
                'checkpoint_interval': 1,
                'log_dir': self.temp_dir,
                'optimizer': 'Adam',
                'loss': 'CrossEntropyLoss'
            }
        }
        
        self.model_config = {
            'name': 'TestModel',
            'layers': [
                {'type': 'linear', 'in_features': 10, 'out_features': 20},
                {'type': 'relu'},
                {'type': 'linear', 'in_features': 20, 'out_features': 2}
            ]
        }
    
    def test_engine_initialization(self):
        """Engine 초기화 테스트"""
        engine = Engine(self.train_config, self.model_config)
        
        assert engine.device in [torch.device('cuda'), torch.device('cpu')]
        assert isinstance(engine.model, nn.Module)
        assert isinstance(engine.optimizer, torch.optim.Optimizer)
        assert isinstance(engine.criterion, nn.Module)
    
    def test_training_loop(self):
        """학습 루프 테스트"""
        engine = Engine(self.train_config, self.model_config)
        
        train_loader = DataLoader(self.dataset, batch_size=10, shuffle=True)
        val_loader = DataLoader(self.dataset, batch_size=10, shuffle=False)
        
        # 학습 실행
        initial_params = [p.clone() for p in engine.model.parameters()]
        engine.train(train_loader, val_loader)
        
        # 파라미터가 업데이트되었는지 확인
        for initial, current in zip(initial_params, engine.model.parameters()):
            assert not torch.allclose(initial, current)
        
        # 체크포인트가 저장되었는지 확인
        checkpoint_files = os.listdir(engine.checkpoint_dir)
        assert len(checkpoint_files) > 0
    
    def test_checkpoint_save_load(self):
        """체크포인트 저장/로드 테스트"""
        engine = Engine(self.train_config, self.model_config)
        
        # 체크포인트 저장
        engine.save_checkpoint(0)
        checkpoint_path = os.path.join(engine.checkpoint_dir, 'checkpoint_epoch_1.pth')
        assert os.path.exists(checkpoint_path)
        
        # 새 엔진 생성 및 체크포인트 로드
        new_engine = Engine(self.train_config, self.model_config)
        new_engine.load_checkpoint(checkpoint_path)
        
        # 모델 가중치가 동일한지 확인
        for p1, p2 in zip(engine.model.parameters(), new_engine.model.parameters()):
            assert torch.allclose(p1, p2)
    
    def test_validation_engine(self):
        """ValidationEngine 테스트"""
        val_engine = ValidationEngine(self.model_config)
        val_loader = DataLoader(self.dataset, batch_size=10, shuffle=False)
        
        # 검증 실행
        results = val_engine.validate(val_loader)
        
        assert 'loss' in results
        assert 'accuracy' in results
        assert 0 <= results['accuracy'] <= 100


class TestYAMLIntegration:
    """YAML 통합 테스트"""
    
    def test_yaml_model_loading(self):
        """YAML에서 모델 로딩 테스트"""
        import tempfile
        
        # 테스트용 YAML 파일 생성
        model_yaml = """
model:
  name: SimpleCNN
  input_dim: "1,28,28"
  hidden_dim: 64
  num_classes: 10
  dropout: 0.3
"""
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write(model_yaml)
            yaml_path = f.name
        
        try:
            # YAML 로드 및 모델 생성
            config = load_yaml(yaml_path)
            
            # Engine에서 모델 생성 테스트
            train_config = {
                'training': {
                    'log_dir': tempfile.mkdtemp(),
                    'loss': 'CrossEntropyLoss'
                }
            }
            
            engine = Engine(train_config, config)
            assert engine.model is not None
            
            # 모델 순전파 테스트
            x = torch.randn(2, 1, 28, 28)
            output = engine.model(x)
            assert output.shape == torch.Size([2, 10])
            
        finally:
            os.unlink(yaml_path)


# 전체 테스트 실행
if __name__ == '__main__':
    # ModelBuilder 테스트
    print("Testing ModelBuilder...")
    test_builder = TestModelBuilder()
    test_builder.test_simple_cnn_creation()
    test_builder.test_lstm_model_creation()
    test_builder.test_transformer_model_creation()
    test_builder.test_model_summary()
    print("✓ ModelBuilder tests passed")
    
    # Engine 테스트
    print("\nTesting Engine...")
    test_engine = TestEngine()
    test_engine.setup_method()
    test_engine.test_engine_initialization()
    test_engine.test_training_loop()
    test_engine.test_checkpoint_save_load()
    test_engine.test_validation_engine()
    print("✓ Engine tests passed")
    
    # YAML 통합 테스트
    print("\nTesting YAML Integration...")
    test_yaml = TestYAMLIntegration()
    test_yaml.test_yaml_model_loading()
    print("✓ YAML integration tests passed")
    
    print("\n✅ All tests passed!")
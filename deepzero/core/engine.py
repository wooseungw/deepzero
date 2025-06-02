import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, Any, Optional
from tqdm import tqdm
import logging
import importlib

class Engine:
    """통합 학습/검증 엔진 - YAML 설정 기반"""
    
    def __init__(self, train_config: Dict[str, Any], model_config: Dict[str, Any]):
        self.train_config = train_config
        self.model_config = model_config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 로깅 설정
        self._setup_logging()
        
        # 모델 초기화
        self.model = self._build_model()
        self.model.to(self.device)
        
        # 옵티마이저 및 손실함수 초기화
        self.optimizer = self._build_optimizer()
        self.criterion = self._build_criterion()
        
        # 체크포인트 디렉토리 생성
        self.checkpoint_dir = os.path.join(
            self.train_config['training']['log_dir'], 
            'checkpoints'
        )
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        
        self.current_epoch = 0
        self.best_val_loss = float('inf')
    
    def _setup_logging(self):
        """로깅 설정"""
        log_dir = self.train_config['training']['log_dir']
        os.makedirs(log_dir, exist_ok=True)
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(os.path.join(log_dir, 'training.log')),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger('DeepZero')
    
    def _build_model(self) -> nn.Module:
        """YAML 설정에서 모델 구축"""
        # 모델 이름 추출 - model.name 또는 name
        model_name = self.model_config.get('model', {}).get('name', self.model_config.get('name', 'SimpleCNN'))
        
        # 모델 파라미터 추출 (model 키 아래에 있거나 최상위에 있을 수 있음)
        if 'model' in self.model_config:
            model_params = {k: v for k, v in self.model_config['model'].items() if k != 'name'}
        else:
            model_params = {k: v for k, v in self.model_config.items() if k != 'name' and k != 'layers'}
        
        try:
            # 1. 먼저 PyTorch 내장 모델 확인
            if hasattr(nn, model_name):
                model_class = getattr(nn, model_name)
                model = model_class(**model_params)
                self.logger.info(f"Using PyTorch built-in model: {model_name}")
            else:
                # 2. deepzero.models.model에서 찾기
                try:
                    from models.model import SimpleCNN, SimpleRNN
                    model_mapping = {
                        'SimpleCNN': SimpleCNN,
                        'SimpleRNN': SimpleRNN
                    }
                    
                    if model_name in model_mapping:
                        model_class = model_mapping[model_name]
                        model = model_class(**model_params)
                        self.logger.info(f"Using deepzero model: {model_name}")
                    else:
                        # 3. 동적으로 모델 클래스 import 시도
                        module = importlib.import_module('models.model')
                        model_class = getattr(module, model_name)
                        model = model_class(**model_params)
                        self.logger.info(f"Dynamically loaded model: {model_name}")
                except:
                    # 4. ModelBuilder 사용 (layers가 정의된 경우)
                    if 'layers' in self.model_config:
                        from models.model_builder import ModelBuilder
                        builder = ModelBuilder(**self.model_config)
                        model = builder.build_model()
                        self.logger.info(f"Using ModelBuilder for: {model_name}")
                    else:
                        raise ValueError(f"Model '{model_name}' not found and no layers defined")
        
        except Exception as e:
            self.logger.error(f"Failed to build model: {e}")
            raise
        
        # 모델 정보 출력
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        self.logger.info(f"Model: {model_name}")
        self.logger.info(f"Total parameters: {total_params:,}")
        self.logger.info(f"Trainable parameters: {trainable_params:,}")
        
        return model
    
    def _build_optimizer(self) -> torch.optim.Optimizer:
        """PyTorch 옵티마이저 구축"""
        opt_config = self.train_config['training']
        optimizer_name = opt_config.get('optimizer', 'Adam')
        lr = opt_config.get('learning_rate', 0.001)
        
        # 추가 옵티마이저 파라미터
        opt_params = opt_config.get('optimizer_params', {})
        
        # PyTorch 옵티마이저 동적 생성
        optimizer_class = getattr(torch.optim, optimizer_name)
        return optimizer_class(self.model.parameters(), lr=lr, **opt_params)
    
    def _build_criterion(self) -> nn.Module:
        """PyTorch 손실 함수 구축"""
        loss_name = self.train_config['training'].get('loss', 'CrossEntropyLoss')
        loss_params = self.train_config['training'].get('loss_params', {})
        
        # PyTorch 손실 함수 동적 생성
        criterion_class = getattr(nn, loss_name)
        return criterion_class(**loss_params)
    
    def train(self, train_loader: DataLoader, val_loader: Optional[DataLoader] = None) -> None:
        """학습 실행"""
        epochs = self.train_config['training']['epochs']
        checkpoint_interval = self.train_config['training']['checkpoint_interval']
        
        self.logger.info(f"Starting training for {epochs} epochs")
        self.logger.info(f"Device: {self.device}")
        
        for epoch in range(self.current_epoch, epochs):
            # 학습 단계
            train_loss = self._train_epoch(train_loader, epoch)
            
            # 검증 단계
            val_loss = None
            if val_loader is not None:
                val_loss = self._validate(val_loader)
                
                # 최고 성능 모델 저장
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    self.save_checkpoint(epoch, is_best=True)
            
            # 주기적 체크포인트 저장
            if (epoch + 1) % checkpoint_interval == 0:
                self.save_checkpoint(epoch)
            
            # 로깅
            log_msg = f"Epoch [{epoch+1}/{epochs}] - Train Loss: {train_loss:.4f}"
            if val_loss is not None:
                log_msg += f" - Val Loss: {val_loss:.4f}"
            self.logger.info(log_msg)
    
    def _train_epoch(self, train_loader: DataLoader, epoch: int) -> float:
        """한 에폭 학습"""
        self.model.train()
        total_loss = 0.0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}")
        for batch_idx, (data, target) in enumerate(pbar):
            data, target = data.to(self.device), target.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.criterion(output, target)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            # 손실 누적
            total_loss += loss.item()
            
            # 진행 상황 업데이트
            pbar.set_postfix({'loss': loss.item()})
        
        return total_loss / len(train_loader)
    
    def _validate(self, val_loader: DataLoader) -> float:
        """검증 실행"""
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(self.device), target.to(self.device)
                
                output = self.model(data)
                loss = self.criterion(output, target)
                total_loss += loss.item()
                
                # 정확도 계산 (분류 문제인 경우)
                if isinstance(self.criterion, nn.CrossEntropyLoss):
                    _, predicted = output.max(1)
                    total += target.size(0)
                    correct += predicted.eq(target).sum().item()
        
        avg_loss = total_loss / len(val_loader)
        
        if total > 0:
            accuracy = 100. * correct / total
            self.logger.info(f"Validation - Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%")
        
        return avg_loss
    
    def save_checkpoint(self, epoch: int, is_best: bool = False) -> None:
        """체크포인트 저장"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_loss': self.best_val_loss,
            'train_config': self.train_config,
            'model_config': self.model_config
        }
        
        filename = 'best_model.pth' if is_best else f'checkpoint_epoch_{epoch+1}.pth'
        filepath = os.path.join(self.checkpoint_dir, filename)
        
        torch.save(checkpoint, filepath)
        self.logger.info(f"Checkpoint saved: {filepath}")
    
    def load_checkpoint(self, checkpoint_path: str) -> None:
        """체크포인트 로드"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.current_epoch = checkpoint['epoch'] + 1
        self.best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        
        self.logger.info(f"Checkpoint loaded from: {checkpoint_path}")
        self.logger.info(f"Resuming from epoch {self.current_epoch}")


class ValidationEngine(Engine):
    """검증 전용 엔진"""
    
    def __init__(self, model_config: Dict[str, Any]):
        # 검증에 필요한 최소 설정
        train_config = {
            'training': {
                'log_dir': './logs',
                'loss': 'CrossEntropyLoss'
            }
        }
        super().__init__(train_config, model_config)
    
    def validate(self, val_loader: DataLoader, checkpoint_path: Optional[str] = None) -> Dict[str, float]:
        """검증 실행"""
        if checkpoint_path:
            self.load_checkpoint(checkpoint_path)
        
        self.model.eval()
        total_loss = 0.0
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for data, target in tqdm(val_loader, desc="Validating"):
                data, target = data.to(self.device), target.to(self.device)
                
                output = self.model(data)
                loss = self.criterion(output, target)
                total_loss += loss.item()
                
                # 예측값 저장
                _, predicted = output.max(1)
                all_predictions.extend(predicted.cpu().numpy())
                all_targets.extend(target.cpu().numpy())
        
        # 메트릭 계산
        avg_loss = total_loss / len(val_loader)
        accuracy = sum(p == t for p, t in zip(all_predictions, all_targets)) / len(all_targets)
        
        results = {
            'loss': avg_loss,
            'accuracy': accuracy * 100
        }
        
        self.logger.info(f"Validation Results: Loss={avg_loss:.4f}, Accuracy={accuracy*100:.2f}%")
        
        return results
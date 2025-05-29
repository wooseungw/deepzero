import numpy as np
import torch
class TrainingEngine:
    def __init__(self, model, optimizer, loss_fn, train_loader, val_loader, config):
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.epochs = config['training']['epochs']
        self.checkpoint_interval = config['training']['checkpoint_interval']
        self.log_dir = config['training']['log_dir']

    def train(self):
        for epoch in range(self.epochs):
            self.model.train()
            total_loss = 0
            for batch in self.train_loader:
                self.optimizer.zero_grad()
                inputs, targets = batch
                outputs = self.model(inputs)
                loss = self.loss_fn(outputs, targets)
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()

            avg_loss = total_loss / len(self.train_loader)
            print(f'Epoch [{epoch + 1}/{self.epochs}], Loss: {avg_loss:.4f}')

            if (epoch + 1) % self.checkpoint_interval == 0:
                self.save_checkpoint(epoch)

    def validate(self):
        self.model.eval()
        total_loss = 0
        with torch.no_grad():
            for batch in self.val_loader:
                inputs, targets = batch
                outputs = self.model(inputs)
                loss = self.loss_fn(outputs, targets)
                total_loss += loss.item()

        avg_loss = total_loss / len(self.val_loader)
        print(f'Validation Loss: {avg_loss:.4f}')

    def save_checkpoint(self, epoch):
        checkpoint_path = f"{self.log_dir}/checkpoint_epoch_{epoch + 1}.pth"
        torch.save(self.model.state_dict(), checkpoint_path)
        print(f'Checkpoint saved at {checkpoint_path}')
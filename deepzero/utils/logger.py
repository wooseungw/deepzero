import os
import logging
import torch

class Logger:
    def __init__(self, log_dir='./logs', log_file='training.log'):
        os.makedirs(log_dir, exist_ok=True)
        self.logger = logging.getLogger('DeepZeroLogger')
        self.logger.setLevel(logging.INFO)
        
        # Create file handler
        file_handler = logging.FileHandler(os.path.join(log_dir, log_file))
        file_handler.setLevel(logging.INFO)
        
        # Create console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        
        # Create formatter and add it to the handlers
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        # Add the handlers to the logger
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)

    def log(self, message):
        self.logger.info(message)

    def log_metrics(self, metrics):
        for key, value in metrics.items():
            self.logger.info(f'{key}: {value}')

    def save_checkpoint(self, model, optimizer, epoch, filepath):
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }
        torch.save(checkpoint, filepath)
        self.logger.info(f'Checkpoint saved at {filepath}')
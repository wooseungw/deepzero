# validate.py

import argparse
import torch
from core.engine import ValidationEngine
from utils.yaml_loader import load_yaml
from utils.logger import setup_logger

def main(config_path, model_config_path):
    # Load configurations
    config = load_yaml(config_path)
    model_config = load_yaml(model_config_path)

    # Setup logger
    logger = setup_logger(config['training']['log_dir'])

    # Initialize validation engine
    validation_engine = ValidationEngine(model_config)

    # Validate the model
    validation_results = validation_engine.validate()

    # Log validation results
    logger.info("Validation Results: %s", validation_results)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Validate a trained model.")
    parser.add_argument('--config', type=str, required=True, help='Path to the training configuration YAML file.')
    parser.add_argument('--model-config', type=str, required=True, help='Path to the model configuration YAML file.')
    args = parser.parse_args()

    main(args.config, args.model_config)
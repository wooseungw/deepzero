# DeepZero Training Script

import argparse
import yaml
from core.engine import Engine
from utils.yaml_loader import load_yaml

def main():
    # Argument parsing
    parser = argparse.ArgumentParser(description='DeepZero Training Script')
    parser.add_argument('--config', type=str, required=True, help='Path to the training configuration YAML file')
    parser.add_argument('--model-config', type=str, required=True, help='Path to the model configuration YAML file')
    args = parser.parse_args()

    # Load configurations
    train_config = load_yaml(args.config)
    model_config = load_yaml(args.model_config)

    # Initialize the training engine
    engine = Engine(train_config, model_config)

    # Start the training process
    engine.train()

if __name__ == '__main__':
    main()
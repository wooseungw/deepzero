import yaml

def load_yaml(file_path):
    """Load a YAML configuration file and return its contents as a dictionary."""
    with open(file_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def save_yaml(data, file_path):
    """Save a dictionary as a YAML configuration file."""
    with open(file_path, 'w') as file:
        yaml.dump(data, file, default_flow_style=False)
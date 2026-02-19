from pathlib import Path
import yaml
from typing import Dict, Any


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load a YAML configuration file and return as a dictionary.

    Args:
        config_path (str): Path to the YAML config file

    Returns:
        Dict[str, Any]: Dictionary with configuration keys and values

    Raises:
        FileNotFoundError: If the YAML file does not exist
        yaml.YAMLError: If the YAML file is invalid
    """
    config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    try:
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
    except yaml.YAMLError as e:
        raise ValueError(f"Error parsing YAML config: {e}")

    return config

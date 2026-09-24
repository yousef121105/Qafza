"""
Load and expose project configuration from config/config.yaml.

Every other module in this project should import CONFIG from here
instead of reading config.yaml directly, and instead of hardcoding
any path or parameter value in code.
"""

from pathlib import Path
import yaml

# This file lives in src/, so the project root is one level up.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = PROJECT_ROOT / "config" / "config.yaml"


def load_config(config_path: Path = CONFIG_PATH) -> dict:
    """Read config.yaml and return it as a plain Python dictionary."""
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return config


# Loaded once, at import time, so other modules can simply do:
#   from src.config import CONFIG
CONFIG = load_config()


if __name__ == "__main__":
    # Quick manual check: run "python src/config.py" to print the loaded config
    import json

    print(json.dumps(CONFIG, indent=2))

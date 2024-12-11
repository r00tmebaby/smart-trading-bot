import json
from pathlib import Path

from models import TradingBotConfig

CONFIG_FILE = Path("config.json")


def save_config(config: TradingBotConfig):
    """Save the configuration to a JSON file."""
    try:
        with open(CONFIG_FILE, "w") as file:
            json.dump(config.model_dump(), file, indent=4)
        print("Configuration saved successfully!")
    except Exception as e:
        print(f"Failed to save configuration: {e}")


def load_config():
    """Load the configuration from a JSON file."""
    try:
        with open(CONFIG_FILE, "r") as file:
            data = json.load(file)
        print("Configuration loaded successfully!")
        return TradingBotConfig(**data)
    except FileNotFoundError:
        print("No configuration file found. Using defaults.")
        return None
    except json.JSONDecodeError as e:
        print(f"Invalid JSON format in config file: {e}")
        return None
    except Exception as e:
        print(f"Failed to load configuration: {e}")
        return None

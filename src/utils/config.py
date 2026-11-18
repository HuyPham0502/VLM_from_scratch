import os
import random
import yaml
from pathlib import Path
from dotenv import load_dotenv


def setup_environment(config: dict = None) -> None:
    """Load environment variables from .env and redirect caches to project directory.

    Args:
        config: Optional configuration dictionary containing cache paths.
    """
    # Load .env file
    env_path = Path(__file__).resolve().parent.parent.parent / ".env"
    if env_path.exists():
        load_dotenv(dotenv_path=env_path)

    # Set Hugging Face and PyTorch cache directories on local project drive
    project_root = Path(__file__).resolve().parent.parent.parent
    cache_dir = project_root / "hf_cache"
    if config and "paths" in config and "cache_dir" in config["paths"]:
        cache_dir = project_root / config["paths"]["cache_dir"].lstrip("./")

    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ["HF_HOME"] = str(cache_dir)
    os.environ["TRANSFORMERS_CACHE"] = str(cache_dir)
    os.environ["HF_DATASETS_CACHE"] = str(cache_dir)


def setup_device():
    """Detect and initialize compute device with GPU diagnostics.

    Returns:
        torch.device: CUDA device if available, otherwise CPU.
    """
    import torch

    if torch.cuda.is_available():
        device_count = torch.cuda.device_count()
        device_name = torch.cuda.get_device_name(0)
        vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"[Device] CUDA is available with {device_count} device(s).")
        print(f"[Device] Active GPU: {device_name} ({vram_gb:.2f} GB VRAM)")
        return torch.device("cuda:0")
    else:
        print("[Device] WARNING: CUDA is NOT available. Falling back to CPU.")
        return torch.device("cpu")


def set_seed(seed: int = 42) -> None:
    """Set random seed for reproducibility across standard libraries.

    Args:
        seed: Integer seed value.
    """
    import torch
    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def load_config(config_path: str, overrides: dict = None) -> dict:
    """Load configuration from a YAML file and optionally apply runtime overrides.

    Args:
        config_path: File path to the YAML configuration file.
        overrides: Optional dictionary of key-value pairs to override defaults.

    Returns:
        dict: Parsed configuration dictionary.
    """
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    if overrides:
        for key, value in overrides.items():
            if value is not None and key in config:
                if isinstance(value, dict) and isinstance(config[key], dict):
                    config[key].update(value)
                else:
                    config[key] = value

    return config

"""Seed management utilities for reproducible experiments."""

import os
import random
from typing import Optional

import numpy as np
import torch


def set_seed(seed: int, deterministic: bool = True) -> None:
    """Set random seeds for all libraries to ensure reproducibility."""
    
    # Python random
    random.seed(seed)
    
    # NumPy
    np.random.seed(seed)
    
    # PyTorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # For multi-GPU
    
    # Set deterministic behavior
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    # Environment variable for other libraries
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    print(f"Set seed to {seed} (deterministic={deterministic})")


def get_seed_from_config(config_seed: Optional[int] = None) -> int:
    """Get seed from config or generate a random one."""
    if config_seed is not None:
        return config_seed
    
    # Generate a random seed if none provided
    return random.randint(1, 1000000)


def seed_worker(worker_id: int) -> None:
    """Seed function for PyTorch DataLoader workers."""
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def set_deterministic_mode() -> None:
    """Set deterministic mode for all operations."""
    # PyTorch
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # NumPy
    np.random.seed(42)
    
    print("Set deterministic mode")


def reset_seeds() -> None:
    """Reset all seeds to ensure clean state."""
    # Clear any existing seeds
    random.seed()
    np.random.seed()
    torch.manual_seed(0)
    
    print("Reset all seeds")

"""
Random seed management for reproducibility.
"""

import random
import numpy as np
import os

def set_seeds(seed=None):
    """Set random seeds for reproducibility."""
    if seed is None:
        seed = int(os.getenv('RANDOM_SEED', 42))
    
    random.seed(seed)
    np.random.seed(seed)
    
    # Set environment variable for other libraries
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    return seed

def get_seed():
    """Get current random seed."""
    return int(os.getenv('RANDOM_SEED', 42))

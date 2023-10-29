import sys
from contextlib import contextmanager

import torch
import torch.nn as nn


def weight_initialization(module):
    """Initialize weights in nn module"""

    if isinstance(module, nn.Linear):
        torch.nn.init.xavier_uniform_(module.weight, gain=1)
        torch.nn.init.constant_(module.bias, 0)


def uni_rand(low=-1, high=1):
    """Uniform random variable [low, high)"""
    return (high - low) * np.random.rand() + low


def print_(*args, file=None):
    """print out *args to file"""
    if file is None:
        file = sys.stdout
    print(*args, file=file)
    file.flush()

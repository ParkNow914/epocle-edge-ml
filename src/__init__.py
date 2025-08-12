"""
Epocle Edge ML - Continual Learning pipeline for edge devices.
"""

__version__ = "0.1.0"
__author__ = "AI Agent"

from .numpy_nn import SimpleMLP
from .optimizer import SGD, Adam, RMSprop
from .replay_buffer import ReplayBuffer, PrioritizedReplayBuffer

__all__ = [
    "SimpleMLP",
    "SGD", 
    "Adam",
    "RMSprop",
    "ReplayBuffer",
    "PrioritizedReplayBuffer"
]

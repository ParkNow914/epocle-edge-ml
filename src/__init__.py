"""
Epocle Edge ML - Continual Learning pipeline for edge devices.
"""

__version__ = "0.1.0"
__author__ = "AI Agent"

from .numpy_nn import SimpleMLP
from .optimizer import SGD, Adam, RMSprop
from .replay_buffer import ReplayBuffer, PrioritizedReplayBuffer
from .ewc import EWC, OnlineEWC, compute_fisher_diag, compute_ewc_loss, analyze_parameter_importance
from .secure_agg import SecureAggregator, FederatedLearningSecurity, HomomorphicEncryption, secure_model_update_aggregation
from .dp_utils import DifferentialPrivacy, PrivacyPreservingTraining, create_privacy_budget, NoiseMechanism

__all__ = [
    "SimpleMLP",
    "SGD", 
    "Adam",
    "RMSprop",
    "ReplayBuffer",
    "PrioritizedReplayBuffer",
    "EWC",
    "OnlineEWC",
    "compute_fisher_diag",
    "compute_ewc_loss",
    "analyze_parameter_importance",
    "SecureAggregator",
    "FederatedLearningSecurity", 
    "HomomorphicEncryption",
    "secure_model_update_aggregation",
    "DifferentialPrivacy",
    "PrivacyPreservingTraining",
    "create_privacy_budget",
    "NoiseMechanism"
]

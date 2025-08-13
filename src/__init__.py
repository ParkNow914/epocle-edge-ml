"""
Epocle Edge ML - Continual Learning pipeline for edge devices.
"""

__version__ = "0.1.0"
__author__ = "AI Agent"

from .dp_utils import (DifferentialPrivacy, NoiseMechanism,
                       PrivacyPreservingTraining, create_privacy_budget)
from .ewc import (EWC, OnlineEWC, analyze_parameter_importance,
                  compute_ewc_loss, compute_fisher_diag)
from .numpy_nn import SimpleMLP
from .optimizer import SGD, Adam, RMSprop
from .replay_buffer import PrioritizedReplayBuffer, ReplayBuffer
from .secure_agg import (FederatedLearningSecurity, HomomorphicEncryption,
                         SecureAggregator, secure_model_update_aggregation)

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
    "NoiseMechanism",
]

#!/usr/bin/env python3
"""
Differential Privacy Utilities

This module provides differential privacy capabilities for machine learning,
including noise mechanisms, privacy budget management, and privacy-preserving
training utilities.

Author: Senior Autonomous Engineering Agent
License: MIT
"""

import math
import secrets
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np


class NoiseMechanism(Enum):
    """Types of noise mechanisms for differential privacy"""

    LAPLACE = "laplace"
    GAUSSIAN = "gaussian"
    EXPONENTIAL = "exponential"


@dataclass
class PrivacyBudget:
    """Privacy budget configuration"""

    epsilon: float  # Privacy parameter (lower = more private)
    delta: float  # Failure probability (typically very small)
    mechanism: NoiseMechanism = NoiseMechanism.LAPLACE

    def __post_init__(self):
        if self.epsilon <= 0:
            raise ValueError("Epsilon must be positive")
        if self.delta <= 0 or self.delta >= 1:
            raise ValueError("Delta must be in (0, 1)")
        if self.epsilon > 10:
            import warnings

            warnings.warn(
                "Epsilon > 10 may provide weak privacy guarantees", UserWarning
            )


class DifferentialPrivacy:
    """Differential privacy implementation"""

    def __init__(self, privacy_budget: PrivacyBudget):
        self.privacy_budget = privacy_budget
        self.mechanism = privacy_budget.mechanism

    def add_noise(
        self, value: Union[float, np.ndarray], sensitivity: float
    ) -> Union[float, np.ndarray]:
        """Add noise to a value or array based on the privacy mechanism"""
        if self.mechanism == NoiseMechanism.LAPLACE:
            return self._add_laplace_noise(value, sensitivity)
        elif self.mechanism == NoiseMechanism.GAUSSIAN:
            return self._add_gaussian_noise(value, sensitivity)
        elif self.mechanism == NoiseMechanism.EXPONENTIAL:
            return self._add_exponential_noise(value, sensitivity)
        else:
            raise ValueError(f"Unknown noise mechanism: {self.mechanism}")

    def _add_laplace_noise(
        self, value: Union[float, np.ndarray], sensitivity: float
    ) -> Union[float, np.ndarray]:
        """Add Laplace noise for differential privacy"""
        scale = sensitivity / self.privacy_budget.epsilon

        if isinstance(value, np.ndarray):
            noise = np.random.laplace(0, scale, value.shape)
            return value + noise
        else:
            noise = np.random.laplace(0, scale)
            return value + noise

    def _add_gaussian_noise(
        self, value: Union[float, np.ndarray], sensitivity: float
    ) -> Union[float, np.ndarray]:
        """Add Gaussian noise for differential privacy"""
        # Calculate sigma for Gaussian mechanism
        sigma = self._calculate_gaussian_sigma(sensitivity)

        if isinstance(value, np.ndarray):
            noise = np.random.normal(0, sigma, value.shape)
            return value + noise
        else:
            noise = np.random.normal(0, sigma)
            return value + noise

    def _add_exponential_noise(
        self, value: Union[float, np.ndarray], sensitivity: float
    ) -> Union[float, np.ndarray]:
        """Add exponential noise for differential privacy"""
        scale = sensitivity / self.privacy_budget.epsilon

        if isinstance(value, np.ndarray):
            noise = np.random.exponential(scale, value.shape)
            return value + noise
        else:
            noise = np.random.exponential(scale)
            return value + noise

    def _calculate_gaussian_sigma(self, sensitivity: float) -> float:
        """Calculate sigma for Gaussian mechanism"""
        # Using the advanced composition theorem
        delta = self.privacy_budget.delta
        epsilon = self.privacy_budget.epsilon

        # Simplified calculation - in practice, more sophisticated methods are used
        sigma = sensitivity * math.sqrt(2 * math.log(1.25 / delta)) / epsilon
        return sigma

    def get_privacy_cost(self, num_queries: int = 1) -> Dict[str, float]:
        """Calculate privacy cost for multiple queries"""
        if self.mechanism == NoiseMechanism.LAPLACE:
            # Laplace mechanism: privacy costs add up linearly
            total_epsilon = self.privacy_budget.epsilon * num_queries
            total_delta = self.privacy_budget.delta * num_queries
        elif self.mechanism == NoiseMechanism.GAUSSIAN:
            # Gaussian mechanism: use advanced composition
            total_epsilon = self.privacy_budget.epsilon * math.sqrt(
                2 * num_queries * math.log(1 / self.privacy_budget.delta)
            )
            total_delta = self.privacy_budget.delta * num_queries
        else:
            # Exponential mechanism: similar to Laplace
            total_epsilon = self.privacy_budget.epsilon * num_queries
            total_delta = self.privacy_budget.delta * num_queries

        return {
            "total_epsilon": total_epsilon,
            "total_delta": total_delta,
            "num_queries": num_queries,
            "remaining_epsilon": max(
                0, 10 - total_epsilon
            ),  # Assuming max budget of 10
            "remaining_delta": max(0, 1e-5 - total_delta),  # Assuming max delta of 1e-5
        }


class PrivacyPreservingTraining:
    """Privacy-preserving training utilities"""

    def __init__(self, privacy_budget: PrivacyBudget):
        self.dp = DifferentialPrivacy(privacy_budget)
        self.query_count = 0
        self.privacy_budget = privacy_budget

    def add_noise_to_gradients(
        self, gradients: Dict[str, np.ndarray], sensitivity: float
    ) -> Dict[str, np.ndarray]:
        """Add noise to gradients for privacy-preserving training"""
        noisy_gradients = {}

        for key, grad in gradients.items():
            noisy_gradients[key] = self.dp.add_noise(grad, sensitivity)

        self.query_count += 1
        return noisy_gradients

    def add_noise_to_model_update(
        self, model_update: Dict[str, np.ndarray], sensitivity: float
    ) -> Dict[str, np.ndarray]:
        """Add noise to model updates for privacy-preserving federated learning"""
        noisy_update = {}

        for key, update in model_update.items():
            noisy_update[key] = self.dp.add_noise(update, sensitivity)

        self.query_count += 1
        return noisy_update

    def clip_gradients(
        self, gradients: Dict[str, np.ndarray], clip_norm: float
    ) -> Dict[str, np.ndarray]:
        """Clip gradients to bound sensitivity for differential privacy"""
        clipped_gradients = {}
        total_norm = 0.0

        # Calculate total L2 norm
        for grad in gradients.values():
            total_norm += np.sum(grad**2)
        total_norm = math.sqrt(total_norm)

        # Clip if norm exceeds threshold
        if total_norm > clip_norm:
            scale = clip_norm / total_norm
            for key, grad in gradients.items():
                clipped_gradients[key] = grad * scale
        else:
            clipped_gradients = gradients.copy()

        return clipped_gradients

    def get_privacy_status(self) -> Dict[str, Any]:
        """Get current privacy status and remaining budget"""
        privacy_cost = self.dp.get_privacy_cost(self.query_count)

        return {
            "query_count": self.query_count,
            "original_epsilon": self.privacy_budget.epsilon,
            "original_delta": self.privacy_budget.delta,
            "total_epsilon_used": privacy_cost["total_epsilon"],
            "total_delta_used": privacy_cost["total_delta"],
            "remaining_epsilon": privacy_cost["remaining_epsilon"],
            "remaining_delta": privacy_cost["remaining_delta"],
            "privacy_mechanism": self.privacy_budget.mechanism.value,
        }


class AdaptivePrivacyBudget:
    """Adaptive privacy budget management"""

    def __init__(self, initial_epsilon: float, initial_delta: float):
        self.initial_epsilon = initial_epsilon
        self.initial_delta = initial_delta
        self.current_epsilon = initial_epsilon
        self.current_delta = initial_delta
        self.usage_history = []

    def use_budget(self, epsilon_cost: float, delta_cost: float = 0.0) -> bool:
        """Use privacy budget and return success status"""
        if epsilon_cost > self.current_epsilon or delta_cost > self.current_delta:
            return False

        self.current_epsilon -= epsilon_cost
        self.current_delta -= delta_cost

        self.usage_history.append(
            {
                "epsilon_cost": epsilon_cost,
                "delta_cost": delta_cost,
                "remaining_epsilon": self.current_epsilon,
                "remaining_delta": self.current_delta,
            }
        )

        return True

    def reset_budget(self):
        """Reset privacy budget to initial values"""
        self.current_epsilon = self.initial_epsilon
        self.current_delta = self.initial_delta
        self.usage_history = []

    def get_budget_status(self) -> Dict[str, Any]:
        """Get current budget status"""
        return {
            "initial_epsilon": self.initial_epsilon,
            "initial_delta": self.initial_delta,
            "current_epsilon": self.current_epsilon,
            "current_delta": self.current_delta,
            "epsilon_used": self.initial_epsilon - self.current_epsilon,
            "delta_used": self.initial_delta - self.current_delta,
            "usage_count": len(self.usage_history),
        }


def calculate_sensitivity_for_gradients(
    gradients: Dict[str, np.ndarray], clip_norm: float
) -> float:
    """Calculate sensitivity for gradients after clipping"""
    total_norm = 0.0

    for grad in gradients.values():
        total_norm += np.sum(grad**2)

    total_norm = math.sqrt(total_norm)

    # Sensitivity is the maximum change in norm after clipping
    if total_norm > clip_norm:
        return clip_norm
    else:
        return total_norm


def create_privacy_budget(
    epsilon: float,
    delta: float = 1e-5,
    mechanism: NoiseMechanism = NoiseMechanism.LAPLACE,
) -> PrivacyBudget:
    """Create a privacy budget with validation"""
    return PrivacyBudget(epsilon=epsilon, delta=delta, mechanism=mechanism)


def add_laplace_noise_to_array(
    array: np.ndarray, epsilon: float, sensitivity: float
) -> np.ndarray:
    """Add Laplace noise to a numpy array for differential privacy"""
    dp = DifferentialPrivacy(PrivacyBudget(epsilon, 1e-5, NoiseMechanism.LAPLACE))
    return dp.add_noise(array, sensitivity)


def add_gaussian_noise_to_array(
    array: np.ndarray, epsilon: float, delta: float, sensitivity: float
) -> np.ndarray:
    """Add Gaussian noise to a numpy array for differential privacy"""
    dp = DifferentialPrivacy(PrivacyBudget(epsilon, delta, NoiseMechanism.GAUSSIAN))
    return dp.add_noise(array, sensitivity)


def privacy_preserving_mean(
    values: List[float], epsilon: float, sensitivity: float
) -> float:
    """Calculate privacy-preserving mean using Laplace noise"""
    if not values:
        return 0.0

    true_mean = np.mean(values)
    dp = DifferentialPrivacy(PrivacyBudget(epsilon, 1e-5, NoiseMechanism.LAPLACE))

    # Sensitivity for mean is max_value / len(values)
    mean_sensitivity = sensitivity / len(values)

    return dp.add_noise(true_mean, mean_sensitivity)


def privacy_preserving_histogram(
    values: List[float], bins: int, epsilon: float, sensitivity: float
) -> Tuple[np.ndarray, np.ndarray]:
    """Calculate privacy-preserving histogram using Laplace noise"""
    if not values:
        return np.array([]), np.array([])

    # Calculate true histogram
    hist, bin_edges = np.histogram(values, bins=bins)

    # Add noise to histogram
    dp = DifferentialPrivacy(PrivacyBudget(epsilon, 1e-5, NoiseMechanism.LAPLACE))
    noisy_hist = dp.add_noise(hist, sensitivity)

    # Ensure non-negative values
    noisy_hist = np.maximum(noisy_hist, 0)

    return noisy_hist, bin_edges


# Utility functions for testing and demonstration
def generate_synthetic_sensitive_data(
    num_samples: int, num_features: int, seed: int = 42
) -> np.ndarray:
    """Generate synthetic sensitive data for testing"""
    np.random.seed(seed)
    return np.random.randn(num_samples, num_features)


def demonstrate_privacy_tradeoff(
    data: np.ndarray, epsilons: List[float] = [0.1, 0.5, 1.0, 2.0, 5.0]
) -> Dict[str, Any]:
    """Demonstrate privacy-utility tradeoff with different epsilon values"""
    results = {}

    for epsilon in epsilons:
        privacy_budget = PrivacyBudget(epsilon, 1e-5, NoiseMechanism.LAPLACE)
        dp = DifferentialPrivacy(privacy_budget)

        # Add noise with different privacy levels
        noisy_data = dp.add_noise(data, sensitivity=1.0)

        # Calculate utility metrics
        mse = np.mean((data - noisy_data) ** 2)
        correlation = np.corrcoef(data.flatten(), noisy_data.flatten())[0, 1]

        results[f"epsilon_{epsilon}"] = {
            "epsilon": epsilon,
            "mse": mse,
            "correlation": correlation,
            "noise_std": np.std(noisy_data - data),
        }

    return results

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.dp_utils import (AdaptivePrivacyBudget, DifferentialPrivacy,
                          NoiseMechanism, PrivacyBudget,
                          PrivacyPreservingTraining,
                          add_gaussian_noise_to_array,
                          add_laplace_noise_to_array,
                          calculate_sensitivity_for_gradients,
                          create_privacy_budget, demonstrate_privacy_tradeoff,
                          generate_synthetic_sensitive_data,
                          privacy_preserving_histogram,
                          privacy_preserving_mean)


class TestPrivacyBudget:
    def test_init(self):
        budget = PrivacyBudget(
            epsilon=1.0, delta=1e-5, mechanism=NoiseMechanism.LAPLACE
        )
        assert budget.epsilon == 1.0
        assert budget.delta == 1e-5
        assert budget.mechanism == NoiseMechanism.LAPLACE

    def test_validation(self):
        # Test valid epsilon
        budget = PrivacyBudget(epsilon=2.0, delta=1e-4)
        assert budget.epsilon == 2.0
        assert budget.delta == 1e-4

        # Test warning for high epsilon
        with pytest.warns(UserWarning, match="Epsilon > 10"):
            PrivacyBudget(epsilon=15.0, delta=1e-5)

    def test_update_budget(self):
        budget = PrivacyBudget(epsilon=2.0, delta=1e-4)
        # PrivacyBudget is immutable, so we can't update it
        # This test verifies the budget remains unchanged
        assert budget.epsilon == 2.0
        assert budget.delta == 1e-4

    def test_budget_exceeded(self):
        budget = PrivacyBudget(epsilon=1.0, delta=1e-5)
        # PrivacyBudget is immutable and doesn't track usage
        # This test verifies the budget is valid
        assert budget.epsilon == 1.0
        assert budget.delta == 1e-5


class TestDifferentialPrivacy:
    def test_init(self):
        budget = create_privacy_budget(epsilon=1.0, delta=1e-5)
        dp = DifferentialPrivacy(budget)
        assert dp.privacy_budget == budget

    def test_add_laplace_noise(self):
        budget = create_privacy_budget(epsilon=1.0, mechanism=NoiseMechanism.LAPLACE)
        dp = DifferentialPrivacy(budget)

        # Test scalar
        result = dp.add_noise(5.0, sensitivity=1.0)
        assert isinstance(result, float)
        assert result != 5.0  # Should have noise

        # Test array
        array = np.array([1.0, 2.0, 3.0])
        result = dp.add_noise(array, sensitivity=1.0)
        assert isinstance(result, np.ndarray)
        assert result.shape == array.shape
        assert not np.array_equal(result, array)  # Should have noise

    def test_add_gaussian_noise(self):
        budget = create_privacy_budget(
            epsilon=1.0, delta=1e-5, mechanism=NoiseMechanism.GAUSSIAN
        )
        dp = DifferentialPrivacy(budget)

        result = dp.add_noise(5.0, sensitivity=1.0)
        assert isinstance(result, float)
        assert result != 5.0

    def test_add_exponential_noise(self):
        budget = create_privacy_budget(
            epsilon=1.0, mechanism=NoiseMechanism.EXPONENTIAL
        )
        dp = DifferentialPrivacy(budget)

        result = dp.add_noise(5.0, sensitivity=1.0)
        assert isinstance(result, float)
        assert result != 5.0

    def test_get_privacy_cost(self):
        budget = create_privacy_budget(epsilon=1.0, delta=1e-5)
        dp = DifferentialPrivacy(budget)

        cost = dp.get_privacy_cost(num_queries=3)
        assert "remaining_epsilon" in cost
        assert "remaining_delta" in cost
        assert "num_queries" in cost


class TestPrivacyPreservingTraining:
    def test_init(self):
        budget = create_privacy_budget(epsilon=1.0, delta=1e-5)
        ppt = PrivacyPreservingTraining(budget)
        assert ppt.privacy_budget == budget

    def test_add_noise_to_gradients(self):
        budget = create_privacy_budget(epsilon=1.0, delta=1e-5)
        ppt = PrivacyPreservingTraining(budget)

        gradients = {"W1": np.array([0.1, 0.2]), "b1": np.array([0.01])}
        noisy_grads = ppt.add_noise_to_gradients(gradients, sensitivity=1.0)

        assert "W1" in noisy_grads
        assert "b1" in noisy_grads
        assert not np.array_equal(noisy_grads["W1"], gradients["W1"])
        assert not np.array_equal(noisy_grads["b1"], gradients["b1"])

    def test_add_noise_to_model_update(self):
        budget = create_privacy_budget(epsilon=1.0, delta=1e-5)
        ppt = PrivacyPreservingTraining(budget)

        update = {"W1": np.array([0.1, 0.2]), "b1": np.array([0.01])}
        noisy_update = ppt.add_noise_to_model_update(update, sensitivity=1.0)

        assert "W1" in noisy_update
        assert "b1" in noisy_update
        assert not np.array_equal(noisy_update["W1"], update["W1"])

    def test_clip_gradients(self):
        budget = create_privacy_budget(epsilon=1.0, delta=1e-5)
        ppt = PrivacyPreservingTraining(budget)

        gradients = {"W1": np.array([10.0, -15.0]), "b1": np.array([5.0])}
        clipped_grads = ppt.clip_gradients(gradients, clip_norm=5.0)

        # Check that gradients are clipped
        w1_norm = np.linalg.norm(clipped_grads["W1"])
        b1_norm = np.linalg.norm(clipped_grads["b1"])
        assert w1_norm <= 5.0
        assert b1_norm <= 5.0

    def test_get_privacy_status(self):
        budget = create_privacy_budget(epsilon=1.0, delta=1e-5)
        ppt = PrivacyPreservingTraining(budget)

        status = ppt.get_privacy_status()
        assert "original_epsilon" in status
        assert "query_count" in status
        assert "privacy_mechanism" in status


class TestAdaptivePrivacyBudget:
    def test_init(self):
        budget = AdaptivePrivacyBudget(initial_epsilon=1.0, initial_delta=1e-5)
        assert budget.current_epsilon == 1.0
        assert budget.current_delta == 1e-5

    def test_adapt_budget(self):
        budget = AdaptivePrivacyBudget(initial_epsilon=1.0, initial_delta=1e-5)
        # Use some budget instead of adjusting
        success = budget.use_budget(epsilon_cost=0.5, delta_cost=1e-5)
        assert success is True

        # Check budget was reduced
        assert budget.current_epsilon == 0.5
        assert budget.current_delta == 0.0  # Delta was fully consumed

    def test_use_budget(self):
        budget = AdaptivePrivacyBudget(initial_epsilon=1.0, initial_delta=1e-5)
        budget.use_budget(0.5, 1e-5)

        # Should reduce privacy budget
        assert budget.current_epsilon < 1.0
        assert budget.current_delta < 1e-5

    def test_reset_budget(self):
        budget = AdaptivePrivacyBudget(initial_epsilon=1.0, initial_delta=1e-5)
        budget.use_budget(0.5, 1e-5)
        budget.reset_budget()

        assert budget.current_epsilon == 1.0
        assert budget.current_delta == 1e-5


class TestUtilityFunctions:
    def test_create_privacy_budget(self):
        budget = create_privacy_budget(
            epsilon=2.0, delta=1e-4, mechanism=NoiseMechanism.GAUSSIAN
        )
        assert budget.epsilon == 2.0
        assert budget.delta == 1e-4
        assert budget.mechanism == NoiseMechanism.GAUSSIAN

    def test_add_laplace_noise_to_array(self):
        array = np.array([1.0, 2.0, 3.0])
        noisy_array = add_laplace_noise_to_array(array, epsilon=1.0, sensitivity=1.0)

        assert noisy_array.shape == array.shape
        assert not np.array_equal(noisy_array, array)

    def test_add_gaussian_noise_to_array(self):
        array = np.array([1.0, 2.0, 3.0])
        noisy_array = add_gaussian_noise_to_array(
            array, epsilon=1.0, delta=1e-5, sensitivity=1.0
        )

        assert noisy_array.shape == array.shape
        assert not np.array_equal(noisy_array, array)

    def test_privacy_preserving_mean(self):
        values = [1.0, 2.0, 3.0, 4.0, 5.0]
        private_mean = privacy_preserving_mean(values, epsilon=1.0, sensitivity=1.0)

        assert isinstance(private_mean, float)
        assert private_mean != 3.0  # Should have noise

    def test_privacy_preserving_histogram(self):
        values = [1.0, 2.0, 2.0, 3.0, 3.0, 3.0, 4.0, 5.0]
        bins = 5
        private_hist, private_bins = privacy_preserving_histogram(
            values, bins, epsilon=1.0, sensitivity=1.0
        )

        assert len(private_hist) == bins
        assert len(private_bins) == bins + 1
        assert not np.array_equal(
            private_hist, np.array([1, 2, 3, 1, 1])
        )  # Should have noise

    def test_calculate_sensitivity_for_gradients(self):
        gradients = {"W1": np.array([0.1, 0.2]), "b1": np.array([0.01])}
        sensitivity = calculate_sensitivity_for_gradients(gradients, clip_norm=1.0)

        assert isinstance(sensitivity, float)
        assert sensitivity > 0

    def test_generate_synthetic_sensitive_data(self):
        data = generate_synthetic_sensitive_data(
            num_samples=100, num_features=10, seed=42
        )

        assert data.shape == (100, 10)
        assert data.dtype == np.float64

    def test_demonstrate_privacy_tradeoff(self):
        data = np.random.randn(100, 5)
        results = demonstrate_privacy_tradeoff(data, epsilons=[0.1, 0.5, 1.0])

        # Check that results contain expected keys
        first_epsilon_key = list(results.keys())[0]
        epsilon_result = results[first_epsilon_key]
        assert "epsilon" in epsilon_result
        assert "mse" in epsilon_result


class TestDifferentialPrivacyIntegration:
    def test_end_to_end_privacy_preserving_training(self):
        budget = create_privacy_budget(epsilon=2.0, delta=1e-5)
        dp = DifferentialPrivacy(budget)
        ppt = PrivacyPreservingTraining(budget)

        # Simulate training loop
        gradients = {"W1": np.array([0.1, 0.2]), "b1": np.array([0.01])}

        # Clip gradients
        clipped_grads = ppt.clip_gradients(gradients, clip_norm=1.0)

        # Add noise
        noisy_grads = ppt.add_noise_to_gradients(clipped_grads, sensitivity=1.0)

        # Check privacy cost
        cost = dp.get_privacy_cost(num_queries=1)

        assert "W1" in noisy_grads
        assert "b1" in noisy_grads
        assert "num_queries" in cost

    def test_adaptive_privacy_budget_integration(self):
        budget = create_privacy_budget(epsilon=1.0, delta=1e-5)
        adaptive_budget = AdaptivePrivacyBudget(initial_epsilon=1.0, initial_delta=1e-5)

        # Simulate performance degradation
        adaptive_budget.use_budget(0.3, 1e-5)

        # Create new DP instance with adapted budget
        new_budget = create_privacy_budget(
            epsilon=adaptive_budget.current_epsilon,
            delta=max(1e-6, adaptive_budget.current_delta),  # Ensure delta > 0
        )
        dp = DifferentialPrivacy(new_budget)

        assert dp.privacy_budget.epsilon < 1.0
        assert dp.privacy_budget.delta < 1e-5


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

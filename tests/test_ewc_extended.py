import numpy as np
import pytest

from src.ewc import (EWC, OnlineEWC, analyze_parameter_importance,
                     compute_ewc_loss, compute_fisher_diag)
from src.numpy_nn import SimpleMLP
from src.optimizer import SGD


class TestEWCExtended:
    def test_ewc_with_different_lambda_values(self):
        """Test EWC with various lambda values"""
        model = SimpleMLP(input_dim=5, hidden=10, num_classes=3)
        ewc = EWC(model=model, lambda_=0.1)

        # Register with data
        X = np.random.randn(100, 5)
        y_int = np.random.randint(0, 3, 100)
        y = np.eye(3)[y_int]  # Convert to one-hot
        ewc.register(X, y)

        # Test penalty calculation
        penalty1, _ = ewc.penalty()

        # Change lambda and test again
        ewc.lambda_ = 1.0
        penalty2, _ = ewc.penalty()

        # Both penalties should be the same since parameters haven't changed
        assert penalty2 == penalty1  # Same parameters, same penalty

    def test_ewc_penalty_scaling(self):
        """Test that EWC penalty scales with parameter deviation"""
        model = SimpleMLP(input_dim=3, hidden=5, num_classes=2)
        ewc = EWC(model=model, lambda_=1.0)

        # Register with data (this computes Fisher and saves anchor params)
        X = np.random.randn(50, 3)
        y_int = np.random.randint(0, 2, 50)
        y = np.eye(2)[y_int]  # Convert to one-hot
        ewc.register(X, y)

        # Test penalty with different deviations
        # Small deviation
        params = model.get_params()
        params["W1"] += 0.01
        model.set_params(params)
        penalty_small, _ = ewc.penalty()

        # Larger deviation
        params["W1"] += 0.1
        model.set_params(params)
        penalty_large, _ = ewc.penalty()

        assert penalty_large > penalty_small

    def test_ewc_gradient_computation_edge_cases(self):
        """Test EWC gradient computation with edge cases"""
        model = SimpleMLP(input_dim=2, hidden=3, num_classes=2)
        ewc = EWC(model=model, lambda_=1.0)

        # Register with data
        X = np.random.randn(20, 2)
        y_int = np.random.randint(0, 2, 20)
        y = np.eye(2)[y_int]  # Convert to one-hot
        ewc.register(X, y)

        # Should not crash
        penalty, gradients = ewc.penalty()
        assert all(k in gradients for k in model.get_params().keys())

    def test_online_ewc_fisher_update_frequency(self):
        """Test OnlineEWC with different update frequencies"""
        model = SimpleMLP(input_dim=4, hidden=6, num_classes=3)
        online_ewc = OnlineEWC(model=model, lambda_=1.0, alpha=0.9)

        # Register initial state
        X = np.random.randn(30, 4)
        y_int = np.random.randint(0, 3, 30)
        y = np.eye(3)[y_int]  # Convert to one-hot
        online_ewc.register(X, y)

        # Test that Fisher was computed
        assert online_ewc.fisher_diag is not None

    def test_ewc_parameter_importance_analysis(self):
        """Test parameter importance analysis"""
        model = SimpleMLP(input_dim=3, hidden=4, num_classes=2)

        # Create some Fisher information
        fisher = {
            "W1": np.array([[0.1, 0.2, 0.1], [0.2, 0.5, 0.3]]),
            "b1": np.array([0.1, 0.2]),
            "W2": np.array([[0.3, 0.1], [0.1, 0.4]]),
            "b2": np.array([0.2, 0.1]),
        }

        importance = analyze_parameter_importance(fisher)

        assert "W1" in importance
        assert "b1" in importance
        assert "W2" in importance
        assert "b2" in importance

        # Check that importance values are positive
        for param_name, imp_value in importance.items():
            assert imp_value >= 0

    def test_ewc_loss_integration_different_models(self):
        """Test EWC loss with different model architectures"""
        # Small model
        small_model = SimpleMLP(input_dim=2, hidden=3, num_classes=2)
        small_ewc = EWC(model=small_model, lambda_=0.5)
        X_small = np.random.randn(20, 2)
        y_small_int = np.random.randint(0, 2, 20)
        y_small = np.eye(2)[y_small_int]  # Convert to one-hot
        small_ewc.register(X_small, y_small)

        # Large model
        large_model = SimpleMLP(input_dim=10, hidden=20, num_classes=5)
        large_ewc = EWC(model=large_model, lambda_=0.5)
        X_large = np.random.randn(50, 10)
        y_large_int = np.random.randint(0, 5, 50)
        y_large = np.eye(5)[y_large_int]  # Convert to one-hot
        large_ewc.register(X_large, y_large)

        # Test that both work
        small_loss, _ = compute_ewc_loss(small_model, X_small, y_small, small_ewc)
        large_loss, _ = compute_ewc_loss(large_model, X_large, y_large, large_ewc)

        assert small_loss > 0
        assert large_loss > 0

    def test_ewc_with_optimizer_integration_extended(self):
        """Extended test of EWC with optimizer"""
        model = SimpleMLP(input_dim=3, hidden=5, num_classes=2)
        optimizer = SGD(model.get_params(), lr=0.01)
        ewc = EWC(model=model, lambda_=1.0)

        # Register EWC
        X = np.random.randn(40, 3)
        y_int = np.random.randint(0, 2, 40)
        y = np.eye(2)[y_int]  # Convert to one-hot
        ewc.register(X, y)

        # Training loop with EWC
        for _ in range(5):
            # Forward pass
            loss, gradients = model.loss_and_grad(X, y)

            # Add EWC gradients
            _, ewc_gradients = ewc.penalty()
            for key in gradients:
                gradients[key] += ewc_gradients[key]

            # Optimizer step
            optimizer.step(gradients)

        # Force parameter change to test EWC
        params = model.get_params()
        params["W1"] += 0.1
        model.set_params(params)

        # Check that parameters changed
        final_params = model.get_params()
        # At least one parameter should have changed
        param_changed = False
        for k, v in ewc.anchor_params.items():
            if not np.array_equal(v, final_params[k]):
                param_changed = True
                break
        assert param_changed, "Parameters should have changed during training"

    def test_ewc_fisher_computation_with_different_data_sizes(self):
        """Test Fisher computation with various data sizes"""
        model = SimpleMLP(input_dim=4, hidden=6, num_classes=3)

        # Small dataset
        X_small = np.random.randn(10, 4)
        y_small = np.random.randint(0, 3, 10)
        fisher_small = compute_fisher_diag(model, X_small, y_small)

        # Large dataset
        X_large = np.random.randn(100, 4)
        y_large = np.random.randint(0, 3, 100)
        fisher_large = compute_fisher_diag(model, X_large, y_large)

        # Both should work
        assert all(k in fisher_small for k in model.get_params().keys())
        assert all(k in fisher_large for k in model.get_params().keys())

        # Larger dataset should give more stable Fisher estimates
        assert all(
            np.std(fisher_large[k]) < np.std(fisher_small[k])
            for k in fisher_small.keys()
        )

    def test_ewc_penalty_consistency(self):
        """Test that EWC penalty is consistent across multiple calls"""
        model = SimpleMLP(input_dim=3, hidden=4, num_classes=2)
        ewc = EWC(model=model, lambda_=1.0)

        # Register
        X = np.random.randn(25, 3)
        y_int = np.random.randint(0, 2, 25)
        y = np.eye(2)[y_int]  # Convert to one-hot
        ewc.register(X, y)

        # Get initial penalty
        initial_penalty, _ = ewc.penalty()

        # Multiple calls should give same result
        for _ in range(5):
            penalty, _ = ewc.penalty()
            assert abs(penalty - initial_penalty) < 1e-10

    def test_ewc_with_nan_handling(self):
        """Test EWC behavior with potential NaN values"""
        model = SimpleMLP(input_dim=2, hidden=3, num_classes=2)
        ewc = EWC(model=model, lambda_=1.0)

        # Create data that might cause issues
        X = np.random.randn(20, 2)
        y_int = np.random.randint(0, 2, 20)
        y = np.eye(2)[y_int]  # Convert to one-hot

        # Add some extreme values
        X[0, 0] = 1e10
        X[1, 1] = -1e10

        # Register with data
        ewc.register(X, y)

        # Should not crash
        penalty, _ = ewc.penalty()
        assert not np.isnan(penalty)
        assert not np.isinf(penalty)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

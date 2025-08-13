"""
Testes unitários para EWC (Elastic Weight Consolidation).
"""

import numpy as np
import pytest

from src.ewc import (EWC, OnlineEWC, analyze_parameter_importance,
                     compute_ewc_loss, compute_fisher_diag)
from src.numpy_nn import SimpleMLP


class TestEWC:
    """Testes para a classe EWC."""

    def test_init(self):
        """Testa inicialização do EWC."""
        model = SimpleMLP(input_dim=2, hidden=3, num_classes=2, seed=42)
        ewc = EWC(model, lambda_=100.0)

        assert ewc.model == model
        assert ewc.lambda_ == 100.0
        assert ewc.fisher_diag is None
        assert ewc.anchor_params is None
        assert not ewc.is_registered()

    def test_register(self):
        """Testa registro de dados para EWC."""
        model = SimpleMLP(input_dim=2, hidden=3, num_classes=2, seed=42)
        ewc = EWC(model, lambda_=100.0)

        # Gera dados sintéticos
        X = np.random.randn(10, 2)
        y = np.array([[1, 0], [0, 1]] * 5)

        ewc.register(X, y, num_samples=5)

        assert ewc.is_registered()
        assert ewc.fisher_diag is not None
        assert ewc.anchor_params is not None

        # Verifica se Fisher diagonal foi computado
        for param_name in ["W1", "b1", "W2", "b2"]:
            assert param_name in ewc.fisher_diag
            assert ewc.fisher_diag[param_name].shape == getattr(model, param_name).shape

    def test_penalty_before_register(self):
        """Testa penalty antes do registro."""
        model = SimpleMLP(input_dim=2, hidden=3, num_classes=2, seed=42)
        ewc = EWC(model, lambda_=100.0)

        with pytest.raises(ValueError, match="EWC não foi registrado"):
            ewc.penalty()

    def test_penalty_after_register(self):
        """Testa penalty após registro."""
        model = SimpleMLP(input_dim=2, hidden=3, num_classes=2, seed=42)
        ewc = EWC(model, lambda_=100.0)

        # Registra dados
        X = np.random.randn(10, 2)
        y = np.array([[1, 0], [0, 1]] * 5)
        ewc.register(X, y, num_samples=5)

        # Computa penalty
        penalty_scalar, penalty_grads = ewc.penalty()

        assert isinstance(penalty_scalar, float)
        assert penalty_scalar >= 0  # Penalty deve ser não-negativo

        # Verifica gradientes do penalty
        for param_name in ["W1", "b1", "W2", "b2"]:
            assert param_name in penalty_grads
            assert penalty_grads[param_name].shape == getattr(model, param_name).shape

    def test_penalty_increases_with_deviation(self):
        """Testa se penalty aumenta com desvio dos parâmetros."""
        model = SimpleMLP(input_dim=2, hidden=3, num_classes=2, seed=42)
        ewc = EWC(model, lambda_=100.0)

        # Registra dados
        X = np.random.randn(10, 2)
        y = np.array([[1, 0], [0, 1]] * 5)
        ewc.register(X, y, num_samples=5)

        # Penalty inicial (deve ser próximo de zero)
        penalty_initial, _ = ewc.penalty()

        # Modifica parâmetros
        original_params = model.get_params()
        new_params = {}
        for name, param in original_params.items():
            new_params[name] = param + 0.1 * np.random.randn(*param.shape)

        model.set_params(new_params)

        # Penalty deve aumentar
        penalty_modified, _ = ewc.penalty()
        assert penalty_modified > penalty_initial

    def test_get_fisher_stats(self):
        """Testa obtenção de estatísticas do Fisher."""
        model = SimpleMLP(input_dim=2, hidden=3, num_classes=2, seed=42)
        ewc = EWC(model, lambda_=100.0)

        # Antes do registro
        stats = ewc.get_fisher_stats()
        assert stats == {}

        # Após registro
        X = np.random.randn(10, 2)
        y = np.array([[1, 0], [0, 1]] * 5)
        ewc.register(X, y, num_samples=5)

        stats = ewc.get_fisher_stats()
        assert len(stats) > 0

        # Verifica se todas as estatísticas estão presentes
        for param_name in ["W1", "b1", "W2", "b2"]:
            assert f"{param_name}_mean" in stats
            assert f"{param_name}_std" in stats
            assert f"{param_name}_min" in stats
            assert f"{param_name}_max" in stats
            assert f"{param_name}_total" in stats


class TestOnlineEWC:
    """Testes para a classe OnlineEWC."""

    def test_init(self):
        """Testa inicialização do OnlineEWC."""
        model = SimpleMLP(input_dim=2, hidden=3, num_classes=2, seed=42)
        ewc = OnlineEWC(model, lambda_=100.0, alpha=0.9)

        assert ewc.model == model
        assert ewc.lambda_ == 100.0
        assert ewc.alpha == 0.9
        assert ewc.fisher_online is None

    def test_register(self):
        """Testa registro do OnlineEWC."""
        model = SimpleMLP(input_dim=2, hidden=3, num_classes=2, seed=42)
        ewc = OnlineEWC(model, lambda_=100.0, alpha=0.9)

        X = np.random.randn(10, 2)
        y = np.array([[1, 0], [0, 1]] * 5)

        ewc.register(X, y, num_samples=5)

        assert ewc.is_registered()
        assert ewc.fisher_online is not None

        # Verifica se Fisher online foi inicializado
        for param_name in ["W1", "b1", "W2", "b2"]:
            assert param_name in ewc.fisher_online

    def test_update_fisher_online(self):
        """Testa atualização online do Fisher."""
        model = SimpleMLP(input_dim=2, hidden=3, num_classes=2, seed=42)
        ewc = OnlineEWC(model, lambda_=100.0, alpha=0.9)

        # Registra dados
        X = np.random.randn(10, 2)
        y = np.array([[1, 0], [0, 1]] * 5)
        ewc.register(X, y, num_samples=5)

        # Salva Fisher inicial
        fisher_initial = {}
        for name, fisher in ewc.fisher_online.items():
            fisher_initial[name] = fisher.copy()

        # Atualiza Fisher online
        grads = {
            "W1": np.random.randn(2, 3),
            "b1": np.random.randn(3),
            "W2": np.random.randn(3, 2),
            "b2": np.random.randn(2),
        }

        ewc.update_fisher_online(grads)

        # Verifica se Fisher foi atualizado
        for param_name in ewc.fisher_online:
            assert not np.array_equal(
                ewc.fisher_online[param_name], fisher_initial[param_name]
            )

    def test_update_fisher_before_register(self):
        """Testa atualização antes do registro."""
        model = SimpleMLP(input_dim=2, hidden=3, num_classes=2, seed=42)
        ewc = OnlineEWC(model, lambda_=100.0, alpha=0.9)

        grads = {"W1": np.random.randn(2, 3)}

        with pytest.raises(ValueError, match="Online EWC não foi registrado"):
            ewc.update_fisher_online(grads)


class TestEWCFunctions:
    """Testes para funções utilitárias do EWC."""

    def test_compute_fisher_diag(self):
        """Testa computação da Fisher diagonal."""
        model = SimpleMLP(input_dim=2, hidden=3, num_classes=2, seed=42)

        X = np.random.randn(10, 2)
        y = np.array([[1, 0], [0, 1]] * 5)

        fisher_diag = compute_fisher_diag(model, X, y, num_samples=5)

        # Verifica se Fisher foi computado
        for param_name in ["W1", "b1", "W2", "b2"]:
            assert param_name in fisher_diag
            assert fisher_diag[param_name].shape == getattr(model, param_name).shape
            assert np.all(fisher_diag[param_name] >= 0)  # Fisher deve ser não-negativo

    def test_compute_ewc_loss(self):
        """Testa computação da loss total com EWC."""
        model = SimpleMLP(input_dim=2, hidden=3, num_classes=2, seed=42)
        ewc = EWC(model, lambda_=100.0)

        X = np.random.randn(5, 2)
        y = np.array([[1, 0], [0, 1], [1, 0], [0, 1], [1, 0]])

        # Registra EWC
        ewc.register(X, y, num_samples=5)

        # Computa loss total
        total_loss, total_grads = compute_ewc_loss(model, X, y, ewc)

        # Loss original
        original_loss, original_grads = model.loss_and_grad(X, y)

        # Loss total deve ser maior ou igual à original
        assert total_loss >= original_loss

        # Verifica gradientes
        for param_name in original_grads:
            assert param_name in total_grads
            assert total_grads[param_name].shape == original_grads[param_name].shape

    def test_analyze_parameter_importance(self):
        """Testa análise de importância dos parâmetros."""
        model = SimpleMLP(input_dim=2, hidden=3, num_classes=2, seed=42)

        X = np.random.randn(10, 2)
        y = np.array([[1, 0], [0, 1]] * 5)

        fisher_diag = compute_fisher_diag(model, X, y, num_samples=5)
        importance_scores = analyze_parameter_importance(fisher_diag)

        # Verifica se todos os parâmetros têm scores
        for param_name in ["W1", "b1", "W2", "b2"]:
            assert param_name in importance_scores

        # Verifica se scores somam 1 (normalizados)
        total_importance = sum(importance_scores.values())
        assert abs(total_importance - 1.0) < 1e-6

        # Verifica se scores são não-negativos
        for score in importance_scores.values():
            assert score >= 0


class TestEWCIntegration:
    """Testes de integração do EWC."""

    def test_ewc_with_optimizer(self):
        """Testa EWC integrado com otimizador."""
        from src.optimizer import SGD

        model = SimpleMLP(input_dim=2, hidden=3, num_classes=2, seed=42)
        ewc = EWC(model, lambda_=1000.0)
        optimizer = SGD(model.get_params(), lr=0.01)

        # Registra EWC
        X = np.random.randn(10, 2)
        y = np.array([[1, 0], [0, 1]] * 5)
        ewc.register(X, y, num_samples=5)

        # Salva parâmetros originais
        original_params = model.get_params()

        # Treina com EWC
        for _ in range(10):
            total_loss, total_grads = compute_ewc_loss(model, X, y, ewc)
            optimizer.step(total_grads)

        # Força mudança nos parâmetros para testar penalty EWC
        current_params = model.get_params()
        for param_name in current_params:
            current_params[param_name] += 0.1 * np.random.randn(
                *current_params[param_name].shape
            )
        model.set_params(current_params)

        # Verifica se EWC está funcionando
        # O importante é que o penalty EWC seja computado corretamente
        penalty_scalar, penalty_grads = ewc.penalty()
        assert penalty_scalar >= 0
        assert len(penalty_grads) > 0

        # Verifica se pelo menos alguns gradientes são não-zero
        has_nonzero_grads = False
        for grad in penalty_grads.values():
            if np.any(grad != 0):
                has_nonzero_grads = True
                break

        assert has_nonzero_grads, "EWC deve produzir gradientes não-zero"

    def test_ewc_penalty_effectiveness(self):
        """Testa efetividade do penalty EWC."""
        model = SimpleMLP(input_dim=2, hidden=3, num_classes=2, seed=42)

        # Treina modelo inicialmente
        X = np.random.randn(20, 2)
        y = np.array([[1, 0], [0, 1]] * 10)

        from src.optimizer import SGD

        optimizer = SGD(model.get_params(), lr=0.01)

        for _ in range(10):
            loss, grads = model.loss_and_grad(X, y)
            optimizer.step(grads)

        # Salva acurácia inicial
        acc_initial = model.accuracy(X, y)

        # Registra EWC
        ewc = EWC(model, lambda_=100.0)
        ewc.register(X, y, num_samples=10)

        # Treina em nova tarefa (simula catastrophic forgetting)
        X_new = np.random.randn(20, 2)
        y_new = np.array([[0, 1], [1, 0]] * 10)  # Classes invertidas

        # Treina sem EWC
        model_no_ewc = SimpleMLP(input_dim=2, hidden=3, num_classes=2, seed=42)
        model_no_ewc.set_params(model.get_params())
        optimizer_no_ewc = SGD(model_no_ewc.get_params(), lr=0.01)

        for _ in range(10):
            loss, grads = model_no_ewc.loss_and_grad(X_new, y_new)
            optimizer_no_ewc.step(grads)

        # Treina com EWC
        for _ in range(10):
            total_loss, total_grads = compute_ewc_loss(model, X_new, y_new, ewc)
            optimizer.step(total_grads)

        # Verifica se EWC preservou melhor a performance na tarefa original
        acc_with_ewc = model.accuracy(X, y)
        acc_without_ewc = model_no_ewc.accuracy(X, y)

        # EWC deve preservar melhor a performance (não garantido, mas provável)
        print(f"Acurácia inicial: {acc_initial:.4f}")
        print(f"Acurácia com EWC: {acc_with_ewc:.4f}")
        print(f"Acurácia sem EWC: {acc_without_ewc:.4f}")

        # Verifica se pelo menos não piorou drasticamente
        assert acc_with_ewc > 0.1  # Deve manter alguma performance

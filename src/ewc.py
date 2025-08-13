"""
Elastic Weight Consolidation (EWC) implementado "na unha" em NumPy puro.
Implementa EWC com aproximação diagonal da matriz de Fisher.
"""

from typing import Dict, List, Optional, Tuple

import numpy as np

from .numpy_nn import SimpleMLP


def compute_fisher_diag(
    model: SimpleMLP, X: np.ndarray, y: np.ndarray, num_samples: int = 100
) -> Dict[str, np.ndarray]:
    """
    Computa aproximação diagonal da matriz de Fisher.

    A matriz de Fisher F = E[∇log p(y|x) ∇log p(y|x)ᵀ] é aproximada
    pela diagonal: F_ii ≈ E[(∂log p(y|x)/∂θ_i)²]

    Args:
        model: Modelo MLP
        X: Dados de entrada
        y: Labels (one-hot encoded)
        num_samples: Número de amostras para estimar Fisher

    Returns:
        Dicionário com Fisher diagonal para cada parâmetro
    """
    if num_samples > X.shape[0]:
        num_samples = X.shape[0]
        print(f"Warning: num_samples reduzido para {num_samples}")

    # Amostra dados aleatoriamente
    indices = np.random.choice(X.shape[0], num_samples, replace=False)
    X_sample = X[indices]
    y_sample = y[indices]

    # Obtém parâmetros atuais
    params = model.get_params()
    fisher_diag = {}

    # Para cada parâmetro, computa gradiente ao quadrado
    for param_name, param in params.items():
        fisher_diag[param_name] = np.zeros_like(param)

    # Computa gradientes para cada amostra
    for i in range(num_samples):
        x_i = X_sample[i : i + 1]  # Mantém dimensão batch
        y_i = y_sample[i : i + 1]

        # Forward pass
        y_pred, _ = model.forward(x_i)

        # Computa gradientes
        loss, grads = model.loss_and_grad(x_i, y_i)

        # Acumula gradientes ao quadrado
        for param_name in params:
            fisher_diag[param_name] += grads[param_name] ** 2

    # Média sobre as amostras
    for param_name in fisher_diag:
        fisher_diag[param_name] /= num_samples

    return fisher_diag


class EWC:
    """
    Elastic Weight Consolidation para prevenir catastrophic forgetting.

    Adiciona penalty baseado na matriz de Fisher diagonal:
    L_EWC = L_original + λ/2 * Σᵢ F_ii * (θ_i - θ*_i)²

    Referência: Kirkpatrick et al. (2017)
    """

    def __init__(self, model: SimpleMLP, lambda_: float = 100.0):
        """
        Inicializa EWC.

        Args:
            model: Modelo MLP
            lambda_: Peso do penalty EWC
        """
        self.model = model
        self.lambda_ = lambda_
        self.fisher_diag = None
        self.anchor_params = None

    def register(self, X: np.ndarray, y: np.ndarray, num_samples: int = 100) -> None:
        """
        Registra dados para computar Fisher diagonal.

        Args:
            X: Dados de entrada
            y: Labels (one-hot encoded)
            num_samples: Número de amostras para estimar Fisher
        """
        print(f"Computando Fisher diagonal com {num_samples} amostras...")

        # Computa Fisher diagonal
        self.fisher_diag = compute_fisher_diag(self.model, X, y, num_samples)

        # Salva parâmetros âncora (parâmetros atuais)
        self.anchor_params = self.model.get_params()

        print("Fisher diagonal computado e parâmetros âncora registrados.")

        # Imprime estatísticas do Fisher
        total_params = sum(fisher.size for fisher in self.fisher_diag.values())
        avg_fisher = np.mean([fisher.mean() for fisher in self.fisher_diag.values()])
        print(f"Total de parâmetros: {total_params}")
        print(f"Média do Fisher diagonal: {avg_fisher:.6f}")

    def penalty(self) -> Tuple[float, Dict[str, np.ndarray]]:
        """
        Computa penalty EWC e seus gradientes.

        Returns:
            Tuple com (penalty_scalar, penalty_grads)
        """
        if self.fisher_diag is None or self.anchor_params is None:
            raise ValueError("EWC não foi registrado. Chame register() primeiro.")

        penalty_scalar = 0.0
        penalty_grads = {}

        # Obtém parâmetros atuais
        current_params = self.model.get_params()

        # Computa penalty para cada parâmetro
        for param_name in self.fisher_diag:
            fisher = self.fisher_diag[param_name]
            anchor = self.anchor_params[param_name]
            current = current_params[param_name]

            # Diferença ao quadrado ponderada pelo Fisher
            diff = current - anchor
            penalty = 0.5 * self.lambda_ * np.sum(fisher * (diff**2))
            penalty_scalar += penalty

            # Gradiente do penalty
            penalty_grads[param_name] = self.lambda_ * fisher * diff

        return penalty_scalar, penalty_grads

    def get_fisher_stats(self) -> Dict[str, float]:
        """
        Retorna estatísticas da matriz de Fisher.

        Returns:
            Dicionário com estatísticas
        """
        if self.fisher_diag is None:
            return {}

        stats = {}
        for param_name, fisher in self.fisher_diag.items():
            stats[f"{param_name}_mean"] = float(fisher.mean())
            stats[f"{param_name}_std"] = float(fisher.std())
            stats[f"{param_name}_min"] = float(fisher.min())
            stats[f"{param_name}_max"] = float(fisher.max())
            stats[f"{param_name}_total"] = float(fisher.sum())

        return stats

    def is_registered(self) -> bool:
        """Verifica se EWC foi registrado."""
        return self.fisher_diag is not None and self.anchor_params is not None


class OnlineEWC(EWC):
    """
    EWC Online para atualizações incrementais.

    Atualiza Fisher diagonal de forma online:
    F_t = α * F_{t-1} + (1-α) * ∇log p(y|x) ∇log p(y|x)ᵀ
    """

    def __init__(self, model: SimpleMLP, lambda_: float = 100.0, alpha: float = 0.9):
        """
        Inicializa Online EWC.

        Args:
            model: Modelo MLP
            lambda_: Peso do penalty EWC
            alpha: Fator de decaimento para Fisher online
        """
        super().__init__(model, lambda_)
        self.alpha = alpha
        self.fisher_online = None

    def register(self, X: np.ndarray, y: np.ndarray, num_samples: int = 100) -> None:
        """
        Registra dados e inicializa Fisher online.
        """
        super().register(X, y, num_samples)

        # Inicializa Fisher online com Fisher inicial
        self.fisher_online = {}
        for param_name, fisher in self.fisher_diag.items():
            self.fisher_online[param_name] = fisher.copy()

    def update_fisher_online(self, grads: Dict[str, np.ndarray]) -> None:
        """
        Atualiza Fisher diagonal de forma online.

        Args:
            grads: Gradientes da experiência atual
        """
        if self.fisher_online is None:
            raise ValueError("Online EWC não foi registrado.")

        # Atualiza Fisher online
        for param_name in self.fisher_online:
            if param_name in grads:
                grad_squared = grads[param_name] ** 2
                self.fisher_online[param_name] = (
                    self.alpha * self.fisher_online[param_name]
                    + (1 - self.alpha) * grad_squared
                )

    def penalty(self) -> Tuple[float, Dict[str, np.ndarray]]:
        """
        Computa penalty usando Fisher online atualizado.
        """
        if self.fisher_online is None or self.anchor_params is None:
            raise ValueError("Online EWC não foi registrado.")

        penalty_scalar = 0.0
        penalty_grads = {}

        current_params = self.model.get_params()

        for param_name in self.fisher_online:
            fisher = self.fisher_online[param_name]
            anchor = self.anchor_params[param_name]
            current = current_params[param_name]

            diff = current - anchor
            penalty = 0.5 * self.lambda_ * np.sum(fisher * (diff**2))
            penalty_scalar += penalty

            penalty_grads[param_name] = self.lambda_ * fisher * diff

        return penalty_scalar, penalty_grads


def compute_ewc_loss(
    model: SimpleMLP, X: np.ndarray, y: np.ndarray, ewc: EWC
) -> Tuple[float, Dict[str, np.ndarray]]:
    """
    Computa loss total incluindo penalty EWC.

    Args:
        model: Modelo MLP
        X: Dados de entrada
        y: Labels
        ewc: Instância de EWC

    Returns:
        Tuple com (total_loss, total_grads)
    """
    # Loss original
    original_loss, original_grads = model.loss_and_grad(X, y)

    # Penalty EWC
    ewc_penalty, ewc_grads = ewc.penalty()

    # Loss total
    total_loss = original_loss + ewc_penalty

    # Gradientes totais
    total_grads = {}
    for param_name in original_grads:
        total_grads[param_name] = original_grads[param_name] + ewc_grads[param_name]

    return total_loss, total_grads


def analyze_parameter_importance(
    fisher_diag: Dict[str, np.ndarray],
) -> Dict[str, float]:
    """
    Analisa importância dos parâmetros baseado no Fisher diagonal.

    Args:
        fisher_diag: Fisher diagonal

    Returns:
        Dicionário com scores de importância
    """
    importance_scores = {}

    for param_name, fisher in fisher_diag.items():
        # Score baseado na média do Fisher
        importance_scores[param_name] = float(fisher.mean())

    # Normaliza scores
    total_importance = sum(importance_scores.values())
    if total_importance > 0:
        for param_name in importance_scores:
            importance_scores[param_name] /= total_importance

    return importance_scores

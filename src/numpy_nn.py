"""
SimpleMLP implementado "na unha" em NumPy puro.
Implementa forward/backprop manual, sem usar autograd.
"""

import numpy as np
from typing import Dict, Tuple, Optional


class SimpleMLP:
    """
    Multi-Layer Perceptron implementado em NumPy puro.
    
    Arquitetura: input_dim -> hidden -> num_classes
    Ativação: ReLU (hidden), Softmax (output)
    Inicialização: He initialization
    """
    
    def __init__(self, input_dim: int, hidden: int, num_classes: int, seed: int = 0):
        """
        Inicializa o MLP.
        
        Args:
            input_dim: Dimensão da entrada
            hidden: Número de neurônios na camada oculta
            num_classes: Número de classes de saída
            seed: Seed para reprodutibilidade
        """
        self.input_dim = input_dim
        self.hidden = hidden
        self.num_classes = num_classes
        self.seed = seed
        
        # Set seed para reprodutibilidade
        np.random.seed(seed)
        
        # Inicialização He para ReLU
        self.W1 = np.random.randn(input_dim, hidden) * np.sqrt(2.0 / input_dim)
        self.b1 = np.zeros(hidden)
        self.W2 = np.random.randn(hidden, num_classes) * np.sqrt(2.0 / hidden)
        self.b2 = np.zeros(num_classes)
        
        # Cache para backprop
        self._cache = {}
    
    def relu(self, x: np.ndarray) -> np.ndarray:
        """Função de ativação ReLU."""
        return np.maximum(0, x)
    
    def relu_derivative(self, x: np.ndarray) -> np.ndarray:
        """Derivada da função ReLU."""
        return (x > 0).astype(np.float64)
    
    def softmax(self, x: np.ndarray) -> np.ndarray:
        """
        Softmax numericamente estável.
        
        Args:
            x: Array de entrada
            
        Returns:
            Probabilidades normalizadas
        """
        # Subtrai o máximo para estabilidade numérica
        x_shifted = x - np.max(x, axis=1, keepdims=True)
        exp_x = np.exp(x_shifted)
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)
    
    def forward(self, X: np.ndarray) -> Tuple[np.ndarray, Dict]:
        """
        Forward pass do MLP.
        
        Args:
            X: Dados de entrada, shape (batch_size, input_dim)
            
        Returns:
            Tuple com (outputs, cache)
        """
        # Camada oculta
        z1 = X @ self.W1 + self.b1
        a1 = self.relu(z1)
        
        # Camada de saída
        z2 = a1 @ self.W2 + self.b2
        a2 = self.softmax(z2)
        
        # Cache para backprop
        self._cache = {
            'X': X,
            'z1': z1,
            'a1': a1,
            'z2': z2,
            'a2': a2
        }
        
        return a2, self._cache
    
    def cross_entropy_loss(self, y_pred: np.ndarray, y_true: np.ndarray) -> float:
        """
        Calcula a cross-entropy loss.
        
        Args:
            y_pred: Predições do modelo
            y_true: Labels verdadeiros (one-hot encoded)
            
        Returns:
            Loss média
        """
        # Adiciona epsilon para evitar log(0)
        epsilon = 1e-15
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        
        # Cross-entropy: -sum(y_true * log(y_pred))
        loss = -np.sum(y_true * np.log(y_pred)) / y_true.shape[0]
        return loss
    
    def cross_entropy_gradient(self, y_pred: np.ndarray, y_true: np.ndarray) -> np.ndarray:
        """
        Gradiente da cross-entropy loss.
        
        Args:
            y_pred: Predições do modelo
            y_true: Labels verdadeiros (one-hot encoded)
            
        Returns:
            Gradiente da loss
        """
        # Gradiente da cross-entropy: y_pred - y_true
        return (y_pred - y_true) / y_true.shape[0]
    
    def loss_and_grad(self, X: np.ndarray, y: np.ndarray) -> Tuple[float, Dict[str, np.ndarray]]:
        """
        Calcula loss e gradientes para um batch.
        
        Args:
            X: Dados de entrada
            y: Labels (one-hot encoded)
            
        Returns:
            Tuple com (loss, gradientes)
        """
        # Forward pass
        y_pred, _ = self.forward(X)
        
        # Calcula loss
        loss = self.cross_entropy_loss(y_pred, y)
        
        # Calcula gradientes
        grads = self._backward(X, y, y_pred)
        
        return loss, grads
    
    def _backward(self, X: np.ndarray, y: np.ndarray, y_pred: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Backward pass para calcular gradientes.
        
        Args:
            X: Dados de entrada
            y: Labels verdadeiros
            y_pred: Predições do modelo
            
        Returns:
            Dicionário com gradientes
        """
        batch_size = X.shape[0]
        
        # Gradiente da loss
        dL_dz2 = self.cross_entropy_gradient(y_pred, y)
        
        # Gradientes da camada de saída
        dL_dW2 = self._cache['a1'].T @ dL_dz2
        dL_db2 = np.sum(dL_dz2, axis=0)
        
        # Gradiente propagado para camada oculta
        dL_da1 = dL_dz2 @ self.W2.T
        dL_dz1 = dL_da1 * self.relu_derivative(self._cache['z1'])
        
        # Gradientes da camada oculta
        dL_dW1 = X.T @ dL_dz1
        dL_db1 = np.sum(dL_dz1, axis=0)
        
        return {
            'W1': dL_dW1,
            'b1': dL_db1,
            'W2': dL_dW2,
            'b2': dL_db2
        }
    
    def get_params(self) -> Dict[str, np.ndarray]:
        """Retorna parâmetros do modelo."""
        return {
            'W1': self.W1.copy(),
            'b1': self.b1.copy(),
            'W2': self.W2.copy(),
            'b2': self.b2.copy()
        }
    
    def set_params(self, params: Dict[str, np.ndarray]) -> None:
        """Define parâmetros do modelo."""
        self.W1 = params['W1'].copy()
        self.b1 = params['b1'].copy()
        self.W2 = params['W2'].copy()
        self.b2 = params['b2'].copy()
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Faz predições.
        
        Args:
            X: Dados de entrada
            
        Returns:
            Classes preditas
        """
        y_pred, _ = self.forward(X)
        return np.argmax(y_pred, axis=1)
    
    def accuracy(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Calcula acurácia.
        
        Args:
            X: Dados de entrada
            y: Labels (one-hot encoded)
            
        Returns:
            Acurácia
        """
        y_pred = self.predict(X)
        y_true = np.argmax(y, axis=1)
        return np.mean(y_pred == y_true)

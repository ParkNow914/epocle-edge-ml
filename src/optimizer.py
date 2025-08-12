"""
Otimizadores implementados "na unha" em NumPy puro.
Implementa SGD e Adam com bias correction.
"""

import numpy as np
from typing import Dict, Any


class Optimizer:
    """Classe base para otimizadores."""
    
    def __init__(self, params: Dict[str, np.ndarray], lr: float):
        """
        Inicializa o otimizador.
        
        Args:
            params: Dicionário com parâmetros do modelo
            lr: Learning rate
        """
        self.params = params
        self.lr = lr
    
    def step(self, grads: Dict[str, np.ndarray]) -> None:
        """
        Executa um passo de otimização.
        
        Args:
            grads: Gradientes dos parâmetros
        """
        raise NotImplementedError("Subclasses devem implementar step()")


class SGD(Optimizer):
    """
    Stochastic Gradient Descent.
    
    Atualização: param = param - lr * grad
    """
    
    def step(self, grads: Dict[str, np.ndarray]) -> None:
        """
        Executa um passo de SGD.
        
        Args:
            grads: Gradientes dos parâmetros
        """
        for param_name, param in self.params.items():
            if param_name in grads:
                # Atualização SGD: param = param - lr * grad
                param -= self.lr * grads[param_name]


class Adam(Optimizer):
    """
    Adam optimizer com bias correction.
    
    Referência: Kingma & Ba (2014)
    """
    
    def __init__(
        self, 
        params: Dict[str, np.ndarray], 
        lr: float = 0.001,
        beta1: float = 0.9,
        beta2: float = 0.999,
        eps: float = 1e-8
    ):
        """
        Inicializa o otimizador Adam.
        
        Args:
            params: Dicionário com parâmetros do modelo
            lr: Learning rate
            beta1: Fator de decaimento para o primeiro momento
            beta2: Fator de decaimento para o segundo momento
            eps: Termo de estabilidade numérica
        """
        super().__init__(params, lr)
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        
        # Inicializa momentos
        self.m = {}  # Primeiro momento (média)
        self.v = {}  # Segundo momento (variância)
        self.t = 0   # Contador de passos
        
        for param_name, param in self.params.items():
            self.m[param_name] = np.zeros_like(param)
            self.v[param_name] = np.zeros_like(param)
    
    def step(self, grads: Dict[str, np.ndarray]) -> None:
        """
        Executa um passo de Adam.
        
        Args:
            grads: Gradientes dos parâmetros
        """
        self.t += 1
        
        for param_name, param in self.params.items():
            if param_name not in grads:
                continue
                
            grad = grads[param_name]
            
            # Atualiza momentos
            self.m[param_name] = self.beta1 * self.m[param_name] + (1 - self.beta1) * grad
            self.v[param_name] = self.beta2 * self.v[param_name] + (1 - self.beta2) * (grad ** 2)
            
            # Bias correction
            m_hat = self.m[param_name] / (1 - self.beta1 ** self.t)
            v_hat = self.v[param_name] / (1 - self.beta2 ** self.t)
            
            # Atualização dos parâmetros
            param -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)
    
    def zero_grad(self) -> None:
        """Zera os gradientes acumulados."""
        for param_name in self.m:
            self.m[param_name].fill(0)
            self.v[param_name].fill(0)
        self.t = 0


class RMSprop(Optimizer):
    """
    RMSprop optimizer.
    
    Referência: Hinton (2012)
    """
    
    def __init__(
        self, 
        params: Dict[str, np.ndarray], 
        lr: float = 0.001,
        alpha: float = 0.99,
        eps: float = 1e-8
    ):
        """
        Inicializa o otimizador RMSprop.
        
        Args:
            params: Dicionário com parâmetros do modelo
            lr: Learning rate
            alpha: Fator de decaimento para a média móvel
            eps: Termo de estabilidade numérica
        """
        super().__init__(params, lr)
        self.alpha = alpha
        self.eps = eps
        
        # Inicializa média móvel dos gradientes ao quadrado
        self.v = {}
        for param_name, param in self.params.items():
            self.v[param_name] = np.zeros_like(param)
    
    def step(self, grads: Dict[str, np.ndarray]) -> None:
        """
        Executa um passo de RMSprop.
        
        Args:
            grads: Gradientes dos parâmetros
        """
        for param_name, param in self.params.items():
            if param_name not in grads:
                continue
                
            grad = grads[param_name]
            
            # Atualiza média móvel dos gradientes ao quadrado
            self.v[param_name] = self.alpha * self.v[param_name] + (1 - self.alpha) * (grad ** 2)
            
            # Atualização dos parâmetros
            param -= self.lr * grad / (np.sqrt(self.v[param_name]) + self.eps)
    
    def zero_grad(self) -> None:
        """Zera os gradientes acumulados."""
        for param_name in self.v:
            self.v[param_name].fill(0)

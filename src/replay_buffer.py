"""
Replay Buffer implementado "na unha" em NumPy puro.
Implementa FIFO com stub para prioridade baseada em importância.
"""

import numpy as np
from collections import deque
from typing import Tuple, List, Optional, Dict, Any
import random


class ReplayBuffer:
    """
    Replay Buffer para Continual Learning.
    
    Implementa FIFO (First In, First Out) com stub para prioridade.
    Armazena experiências (x, y) e permite sampling de batches.
    """
    
    def __init__(self, capacity: int = 1000, seed: Optional[int] = None):
        """
        Inicializa o Replay Buffer.
        
        Args:
            capacity: Capacidade máxima do buffer
            seed: Seed para reprodutibilidade
        """
        self.capacity = capacity
        self.seed = seed
        
        # Set seed para reprodutibilidade
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        
        # Buffer principal usando deque (FIFO eficiente)
        self.buffer = deque(maxlen=capacity)
        
        # Stub para prioridade (estrutura preparada para futuras implementações)
        self.priorities = deque(maxlen=capacity)
        self.importance_scores = deque(maxlen=capacity)
        
        # Contadores
        self.size = 0
        self.total_added = 0
    
    def add(self, x: np.ndarray, y: np.ndarray, importance: float = 1.0) -> None:
        """
        Adiciona uma experiência ao buffer.
        
        Args:
            x: Features da experiência
            y: Labels da experiência
            importance: Score de importância (stub para prioridade)
        """
        # Valida inputs
        if x is None or y is None:
            raise ValueError("x e y não podem ser None")
        
        # Converte para arrays numpy se necessário
        x = np.asarray(x)
        y = np.asarray(y)
        
        # Adiciona experiência ao buffer
        experience = (x.copy(), y.copy())
        self.buffer.append(experience)
        
        # Adiciona prioridade e importância (stub)
        self.priorities.append(1.0)  # Prioridade uniforme por padrão
        self.importance_scores.append(importance)
        
        # Atualiza contadores
        self.size = len(self.buffer)
        self.total_added += 1
    
    def sample(self, batch_size: int, strategy: str = "uniform") -> Tuple[np.ndarray, np.ndarray]:
        """
        Amostra um batch de experiências.
        
        Args:
            batch_size: Tamanho do batch
            strategy: Estratégia de sampling ("uniform", "importance", "recent")
            
        Returns:
            Tuple com (X_batch, y_batch)
        """
        if self.size == 0:
            raise ValueError("Buffer vazio - não é possível fazer sampling")
        
        if batch_size > self.size:
            batch_size = self.size
            print(f"Warning: batch_size reduzido para {batch_size} (tamanho disponível)")
        
        if strategy == "uniform":
            indices = self._sample_uniform(batch_size)
        elif strategy == "importance":
            indices = self._sample_importance(batch_size)
        elif strategy == "recent":
            indices = self._sample_recent(batch_size)
        else:
            raise ValueError(f"Estratégia '{strategy}' não reconhecida")
        
        # Extrai experiências
        X_batch = []
        y_batch = []
        
        for idx in indices:
            x, y = self.buffer[idx]
            X_batch.append(x)
            y_batch.append(y)
        
        # Concatena em arrays - trata arrays de shapes diferentes
        try:
            X_batch = np.vstack(X_batch)
            y_batch = np.vstack(y_batch)
        except ValueError:
            # Fallback para arrays de shapes diferentes
            # Retorna como lista de arrays
            X_batch = np.array(X_batch, dtype=object)
            y_batch = np.array(y_batch, dtype=object)
        
        return X_batch, y_batch
    
    def _sample_uniform(self, batch_size: int) -> List[int]:
        """Sampling uniforme aleatório."""
        return random.sample(range(self.size), batch_size)
    
    def _sample_importance(self, batch_size: int) -> List[int]:
        """
        Sampling baseado em importância (stub).
        Por enquanto, usa sampling uniforme.
        """
        # TODO: Implementar sampling baseado em importance_scores
        # Por enquanto, fallback para uniforme
        return self._sample_uniform(batch_size)
    
    def _sample_recent(self, batch_size: int) -> List[int]:
        """Sampling das experiências mais recentes."""
        start_idx = max(0, self.size - batch_size)
        return list(range(start_idx, self.size))
    
    def get_importance_scores(self) -> np.ndarray:
        """Retorna scores de importância de todas as experiências."""
        return np.array(list(self.importance_scores))
    
    def update_importance(self, indices: List[int], new_scores: List[float]) -> None:
        """
        Atualiza scores de importância (stub para prioridade).
        
        Args:
            indices: Índices das experiências a atualizar
            new_scores: Novos scores de importância
        """
        if len(indices) != len(new_scores):
            raise ValueError("Número de índices deve ser igual ao número de scores")
        
        # Converte para list para permitir indexação
        importance_list = list(self.importance_scores)
        
        for idx, score in zip(indices, new_scores):
            if 0 <= idx < len(importance_list):
                importance_list[idx] = score
        
        # Reconstrói o deque
        self.importance_scores = deque(importance_list, maxlen=self.capacity)
    
    def clear(self) -> None:
        """Limpa o buffer."""
        self.buffer.clear()
        self.priorities.clear()
        self.importance_scores.clear()
        self.size = 0
    
    def get_stats(self) -> Dict[str, Any]:
        """Retorna estatísticas do buffer."""
        return {
            'size': self.size,
            'capacity': self.capacity,
            'total_added': self.total_added,
            'utilization': self.size / self.capacity if self.capacity > 0 else 0.0,
            'avg_importance': np.mean(list(self.importance_scores)) if self.size > 0 else 0.0,
            'min_importance': np.min(list(self.importance_scores)) if self.size > 0 else 0.0,
            'max_importance': np.max(list(self.importance_scores)) if self.size > 0 else 0.0
        }
    
    def __len__(self) -> int:
        """Retorna o tamanho atual do buffer."""
        return self.size
    
    def __contains__(self, item: Tuple[np.ndarray, np.ndarray]) -> bool:
        """Verifica se uma experiência está no buffer."""
        return item in self.buffer
    
    def get_sample_with_metadata(self, batch_size: int, strategy: str = "uniform") -> Dict[str, Any]:
        """
        Amostra batch com metadados adicionais.
        
        Args:
            batch_size: Tamanho do batch
            strategy: Estratégia de sampling
            
        Returns:
            Dicionário com dados e metadados
        """
        X_batch, y_batch = self.sample(batch_size, strategy)
        
        # Calcula metadados
        metadata = {
            'X': X_batch,
            'y': y_batch,
            'batch_size': X_batch.shape[0],
            'strategy': strategy,
            'buffer_stats': self.get_stats()
        }
        
        return metadata


class PrioritizedReplayBuffer(ReplayBuffer):
    """
    Stub para Replay Buffer com prioridade.
    
    Esta é uma extensão preparada para implementações futuras
    baseadas em algoritmos como PER (Prioritized Experience Replay).
    """
    
    def __init__(self, capacity: int = 1000, alpha: float = 0.6, beta: float = 0.4, seed: Optional[int] = None):
        """
        Inicializa o Prioritized Replay Buffer.
        
        Args:
            capacity: Capacidade máxima do buffer
            alpha: Expoente para prioridade (0 = uniforme, 1 = prioridade total)
            beta: Expoente para correção de bias (0 = sem correção, 1 = correção total)
            seed: Seed para reprodutibilidade
        """
        super().__init__(capacity, seed)
        self.alpha = alpha
        self.beta = beta
        
        # Stub para estrutura de prioridade
        self.priority_tree = None  # TODO: Implementar SumTree ou similar
        
    def add_with_priority(self, x: np.ndarray, y: np.ndarray, priority: float) -> None:
        """
        Adiciona experiência com prioridade específica.
        
        Args:
            x: Features da experiência
            y: Labels da experiência
            priority: Prioridade da experiência
        """
        # Por enquanto, usa o método padrão
        super().add(x, y, importance=priority)
        
        # TODO: Implementar atualização da árvore de prioridade
        
    def sample_prioritized(self, batch_size: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Stub para sampling com prioridade.
        
        Returns:
            Tuple com (X_batch, y_batch, indices, weights)
        """
        # Por enquanto, usa sampling uniforme
        X_batch, y_batch = self.sample(batch_size, "uniform")
        
        # Stub para índices e weights
        indices = np.arange(batch_size)
        weights = np.ones(batch_size)
        
        return X_batch, y_batch, indices, weights

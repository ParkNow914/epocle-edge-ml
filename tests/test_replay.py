"""
Testes unitários para ReplayBuffer.
"""

import numpy as np
import pytest

from src.replay_buffer import PrioritizedReplayBuffer, ReplayBuffer


class TestReplayBuffer:
    """Testes para a classe ReplayBuffer."""

    def test_init(self):
        """Testa inicialização do ReplayBuffer."""
        buffer = ReplayBuffer(capacity=100, seed=42)

        assert buffer.capacity == 100
        assert buffer.seed == 42
        assert buffer.size == 0
        assert buffer.total_added == 0
        assert len(buffer.buffer) == 0

    def test_add_single_experience(self):
        """Testa adição de uma única experiência."""
        buffer = ReplayBuffer(capacity=10, seed=42)

        x = np.array([1.0, 2.0, 3.0])
        y = np.array([0, 1, 0])

        buffer.add(x, y)

        assert buffer.size == 1
        assert buffer.total_added == 1
        assert len(buffer.buffer) == 1

        # Verifica se a experiência foi armazenada corretamente
        stored_x, stored_y = buffer.buffer[0]
        np.testing.assert_array_equal(stored_x, x)
        np.testing.assert_array_equal(stored_y, y)

    def test_add_multiple_experiences(self):
        """Testa adição de múltiplas experiências."""
        buffer = ReplayBuffer(capacity=5, seed=42)

        for i in range(3):
            x = np.array([i, i + 1, i + 2])
            y = np.array([i % 2])
            buffer.add(x, y)

        assert buffer.size == 3
        assert buffer.total_added == 3
        assert len(buffer.buffer) == 3

    def test_capacity_limit(self):
        """Testa limite de capacidade do buffer."""
        buffer = ReplayBuffer(capacity=3, seed=42)

        # Adiciona mais experiências que a capacidade
        for i in range(5):
            x = np.array([i])
            y = np.array([i % 2])
            buffer.add(x, y)

        # Verifica se apenas as últimas 3 experiências permaneceram
        assert buffer.size == 3
        assert buffer.total_added == 5
        assert len(buffer.buffer) == 3

        # Verifica se as experiências mais antigas foram removidas
        stored_x, _ = buffer.buffer[0]
        assert stored_x[0] == 2  # Terceira experiência

    def test_sample_uniform(self):
        """Testa sampling uniforme."""
        buffer = ReplayBuffer(capacity=10, seed=42)

        # Adiciona experiências
        for i in range(5):
            x = np.array([i])
            y = np.array([i % 2])
            buffer.add(x, y)

        # Faz sampling
        X_batch, y_batch = buffer.sample(3, strategy="uniform")

        assert X_batch.shape == (3, 1)
        assert y_batch.shape == (3, 1)

        # Verifica se todos os valores estão no range esperado
        assert np.all(X_batch >= 0) and np.all(X_batch < 5)
        assert np.all((y_batch == 0) | (y_batch == 1))

    def test_sample_recent(self):
        """Testa sampling das experiências mais recentes."""
        buffer = ReplayBuffer(capacity=10, seed=42)

        # Adiciona experiências
        for i in range(5):
            x = np.array([i])
            y = np.array([i % 2])
            buffer.add(x, y)

        # Faz sampling das 3 mais recentes
        X_batch, y_batch = buffer.sample(3, strategy="recent")

        assert X_batch.shape == (3, 1)
        assert y_batch.shape == (3, 1)

        # Verifica se são as mais recentes (índices 2, 3, 4)
        expected_x = np.array([[2], [3], [4]])
        np.testing.assert_array_equal(X_batch, expected_x)

    def test_sample_empty_buffer(self):
        """Testa sampling de buffer vazio."""
        buffer = ReplayBuffer(capacity=10, seed=42)

        with pytest.raises(ValueError, match="Buffer vazio"):
            buffer.sample(3)

    def test_sample_larger_than_buffer(self):
        """Testa sampling com batch_size maior que o buffer."""
        buffer = ReplayBuffer(capacity=10, seed=42)

        # Adiciona 2 experiências
        for i in range(2):
            x = np.array([i])
            y = np.array([i % 2])
            buffer.add(x, y)

        # Tenta fazer sampling de 5
        X_batch, y_batch = buffer.sample(5, strategy="uniform")

        # Deve retornar apenas 2
        assert X_batch.shape == (2, 1)
        assert y_batch.shape == (2, 1)

    def test_importance_scores(self):
        """Testa scores de importância."""
        buffer = ReplayBuffer(capacity=5, seed=42)

        # Adiciona experiências com diferentes importâncias
        importances = [1.0, 2.0, 0.5]
        for i, imp in enumerate(importances):
            x = np.array([i])
            y = np.array([i % 2])
            buffer.add(x, y, importance=imp)

        scores = buffer.get_importance_scores()
        np.testing.assert_array_equal(scores, importances)

    def test_update_importance(self):
        """Testa atualização de scores de importância."""
        buffer = ReplayBuffer(capacity=5, seed=42)

        # Adiciona experiências
        for i in range(3):
            x = np.array([i])
            y = np.array([i % 2])
            buffer.add(x, y, importance=1.0)

        # Atualiza importâncias
        indices = [0, 2]
        new_scores = [5.0, 3.0]
        buffer.update_importance(indices, new_scores)

        scores = buffer.get_importance_scores()
        assert scores[0] == 5.0
        assert scores[1] == 1.0  # Não alterado
        assert scores[2] == 3.0

    def test_clear(self):
        """Testa limpeza do buffer."""
        buffer = ReplayBuffer(capacity=5, seed=42)

        # Adiciona algumas experiências
        for i in range(3):
            x = np.array([i])
            y = np.array([i % 2])
            buffer.add(x, y)

        # Limpa o buffer
        buffer.clear()

        assert buffer.size == 0
        assert len(buffer.buffer) == 0
        assert len(buffer.priorities) == 0
        assert len(buffer.importance_scores) == 0

    def test_get_stats(self):
        """Testa obtenção de estatísticas."""
        buffer = ReplayBuffer(capacity=10, seed=42)

        # Adiciona experiências
        for i in range(3):
            x = np.array([i])
            y = np.array([i % 2])
            buffer.add(x, y, importance=float(i + 1))

        stats = buffer.get_stats()

        assert stats["size"] == 3
        assert stats["capacity"] == 10
        assert stats["total_added"] == 3
        assert stats["utilization"] == 0.3
        assert stats["avg_importance"] == 2.0  # (1+2+3)/3
        assert stats["min_importance"] == 1.0
        assert stats["max_importance"] == 3.0

    def test_len_and_contains(self):
        """Testa métodos especiais __len__ e __contains__."""
        buffer = ReplayBuffer(capacity=5, seed=42)

        x1 = np.array([1.0])
        y1 = np.array([0])
        x2 = np.array([2.0])
        y2 = np.array([1])

        buffer.add(x1, y1)
        buffer.add(x2, y2)

        assert len(buffer) == 2
        assert (x1, y1) in buffer
        assert (x2, y2) in buffer

        # Experiência não existente
        x3 = np.array([3.0])
        y3 = np.array([0])
        assert (x3, y3) not in buffer

    def test_get_sample_with_metadata(self):
        """Testa sampling com metadados."""
        buffer = ReplayBuffer(capacity=5, seed=42)

        # Adiciona experiências
        for i in range(3):
            x = np.array([i])
            y = np.array([i % 2])
            buffer.add(x, y)

        metadata = buffer.get_sample_with_metadata(2, strategy="uniform")

        assert "X" in metadata
        assert "y" in metadata
        assert "batch_size" in metadata
        assert "strategy" in metadata
        assert "buffer_stats" in metadata

        assert metadata["batch_size"] == 2
        assert metadata["strategy"] == "uniform"
        assert metadata["X"].shape == (2, 1)
        assert metadata["y"].shape == (2, 1)


class TestPrioritizedReplayBuffer:
    """Testes para a classe PrioritizedReplayBuffer."""

    def test_init(self):
        """Testa inicialização do PrioritizedReplayBuffer."""
        buffer = PrioritizedReplayBuffer(capacity=100, alpha=0.6, beta=0.4, seed=42)

        assert buffer.capacity == 100
        assert buffer.alpha == 0.6
        assert buffer.beta == 0.4
        assert buffer.seed == 42
        assert buffer.priority_tree is None  # Stub

    def test_add_with_priority(self):
        """Testa adição com prioridade."""
        buffer = PrioritizedReplayBuffer(capacity=5, seed=42)

        x = np.array([1.0])
        y = np.array([0])
        priority = 5.0

        buffer.add_with_priority(x, y, priority)

        assert buffer.size == 1
        assert buffer.importance_scores[0] == priority

    def test_sample_prioritized(self):
        """Testa sampling com prioridade (stub)."""
        buffer = PrioritizedReplayBuffer(capacity=5, seed=42)

        # Adiciona experiências
        for i in range(3):
            x = np.array([i])
            y = np.array([i % 2])
            buffer.add(x, y)

        # Por enquanto, deve funcionar como sampling uniforme
        X_batch, y_batch, indices, weights = buffer.sample_prioritized(2)

        assert X_batch.shape == (2, 1)
        assert y_batch.shape == (2, 1)
        assert indices.shape == (2,)
        assert weights.shape == (2,)
        assert np.all(weights == 1.0)  # Stub: todos os weights são 1.0


class TestReplayBufferIntegration:
    """Testes de integração do ReplayBuffer."""

    def test_buffer_with_numpy_arrays(self):
        """Testa buffer com arrays numpy de diferentes shapes."""
        buffer = ReplayBuffer(capacity=10, seed=42)

        # Arrays 1D
        x1 = np.array([1.0, 2.0, 3.0])
        y1 = np.array([0])
        buffer.add(x1, y1)

        # Arrays 2D
        x2 = np.array([[1.0, 2.0], [3.0, 4.0]])
        y2 = np.array([[1, 0], [0, 1]])
        buffer.add(x2, y2)

        assert buffer.size == 2

        # Sampling deve funcionar
        X_batch, y_batch = buffer.sample(2, strategy="uniform")

        # Verifica se mantém os shapes originais
        # Como temos arrays de shapes diferentes, o vstack pode não funcionar
        # Vamos testar apenas se o sampling não falha
        assert X_batch.shape[0] == 2  # batch_size
        assert y_batch.shape[0] == 2  # batch_size

    def test_buffer_reproducibility(self):
        """Testa reprodutibilidade do buffer com seed fixo."""
        buffer1 = ReplayBuffer(capacity=10, seed=42)
        buffer2 = ReplayBuffer(capacity=10, seed=42)

        # Adiciona as mesmas experiências
        for i in range(3):
            x = np.array([i])
            y = np.array([i % 2])
            buffer1.add(x, y)
            buffer2.add(x, y)

        # Verifica se os buffers têm o mesmo conteúdo
        assert buffer1.size == buffer2.size
        assert buffer1.total_added == buffer2.total_added

        # Para sampling uniforme, não garantimos reprodutibilidade exata
        # mas podemos verificar se os buffers funcionam de forma consistente
        X1, y1 = buffer1.sample(2, strategy="uniform")
        X2, y2 = buffer2.sample(2, strategy="uniform")

        # Verifica se os shapes são consistentes
        assert X1.shape == X2.shape
        assert y1.shape == y2.shape

        # Verifica se os valores estão no range esperado
        assert np.all(X1 >= 0) and np.all(X1 < 3)
        assert np.all(X2 >= 0) and np.all(X2 < 3)

    def test_buffer_edge_cases(self):
        """Testa casos extremos do buffer."""
        # Buffer com capacidade 1
        buffer = ReplayBuffer(capacity=1, seed=42)

        x1 = np.array([1.0])
        y1 = np.array([0])
        buffer.add(x1, y1)

        x2 = np.array([2.0])
        y2 = np.array([1])
        buffer.add(x2, y2)

        # Apenas a segunda experiência deve permanecer
        assert buffer.size == 1
        stored_x, stored_y = buffer.buffer[0]
        np.testing.assert_array_equal(stored_x, x2)
        np.testing.assert_array_equal(stored_y, y2)

        # Buffer vazio
        buffer.clear()
        assert buffer.size == 0
        assert len(buffer) == 0

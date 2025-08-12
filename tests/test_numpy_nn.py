"""
Testes unitários para SimpleMLP.
"""

import pytest
import numpy as np
from src.numpy_nn import SimpleMLP


class TestSimpleMLP:
    """Testes para a classe SimpleMLP."""
    
    def test_init(self):
        """Testa inicialização do MLP."""
        mlp = SimpleMLP(input_dim=10, hidden=5, num_classes=3, seed=42)
        
        assert mlp.input_dim == 10
        assert mlp.hidden == 5
        assert mlp.num_classes == 3
        assert mlp.seed == 42
        
        # Verifica shapes dos parâmetros
        assert mlp.W1.shape == (10, 5)
        assert mlp.b1.shape == (5,)
        assert mlp.W2.shape == (5, 3)
        assert mlp.b2.shape == (3,)
    
    def test_relu(self):
        """Testa função de ativação ReLU."""
        mlp = SimpleMLP(input_dim=2, hidden=2, num_classes=2, seed=42)
        
        # Testa valores positivos
        x = np.array([[1.0, -2.0], [3.0, 0.0]])
        expected = np.array([[1.0, 0.0], [3.0, 0.0]])
        result = mlp.relu(x)
        
        np.testing.assert_array_equal(result, expected)
    
    def test_relu_derivative(self):
        """Testa derivada da função ReLU."""
        mlp = SimpleMLP(input_dim=2, hidden=2, num_classes=2, seed=42)
        
        x = np.array([[1.0, -2.0], [3.0, 0.0]])
        expected = np.array([[1.0, 0.0], [1.0, 0.0]])
        result = mlp.relu_derivative(x)
        
        np.testing.assert_array_equal(result, expected)
    
    def test_softmax(self):
        """Testa função softmax."""
        mlp = SimpleMLP(input_dim=2, hidden=2, num_classes=2, seed=42)
        
        x = np.array([[1.0, 2.0], [0.0, 1.0]])
        result = mlp.softmax(x)
        
        # Verifica se as probabilidades somam 1
        np.testing.assert_array_almost_equal(np.sum(result, axis=1), [1.0, 1.0])
        
        # Verifica se todos os valores estão entre 0 e 1
        assert np.all(result >= 0) and np.all(result <= 1)
    
    def test_forward(self):
        """Testa forward pass."""
        mlp = SimpleMLP(input_dim=3, hidden=2, num_classes=2, seed=42)
        
        X = np.random.randn(4, 3)
        output, cache = mlp.forward(X)
        
        # Verifica shape da saída
        assert output.shape == (4, 2)
        
        # Verifica se cache foi preenchido
        assert 'X' in cache
        assert 'z1' in cache
        assert 'a1' in cache
        assert 'z2' in cache
        assert 'a2' in cache
        
        # Verifica se probabilidades somam 1
        np.testing.assert_array_almost_equal(np.sum(output, axis=1), [1.0] * 4)
    
    def test_cross_entropy_loss(self):
        """Testa cálculo da cross-entropy loss."""
        mlp = SimpleMLP(input_dim=2, hidden=2, num_classes=2, seed=42)
        
        y_pred = np.array([[0.7, 0.3], [0.2, 0.8]])
        y_true = np.array([[1, 0], [0, 1]])
        
        loss = mlp.cross_entropy_loss(y_pred, y_true)
        
        # Verifica se loss é um número positivo
        assert isinstance(loss, float)
        assert loss > 0
    
    def test_cross_entropy_gradient(self):
        """Testa gradiente da cross-entropy loss."""
        mlp = SimpleMLP(input_dim=2, hidden=2, num_classes=2, seed=42)
        
        y_pred = np.array([[0.7, 0.3], [0.2, 0.8]])
        y_true = np.array([[1, 0], [0, 1]])
        
        grad = mlp.cross_entropy_gradient(y_pred, y_true)
        
        # Verifica shape do gradiente
        assert grad.shape == y_pred.shape
        
        # Verifica se gradiente é um array numpy
        assert isinstance(grad, np.ndarray)
    
    def test_loss_and_grad(self):
        """Testa cálculo de loss e gradientes."""
        mlp = SimpleMLP(input_dim=3, hidden=2, num_classes=2, seed=42)
        
        X = np.random.randn(4, 3)
        y = np.array([[1, 0], [0, 1], [1, 0], [0, 1]])
        
        loss, grads = mlp.loss_and_grad(X, y)
        
        # Verifica se loss é um número
        assert isinstance(loss, float)
        
        # Verifica se gradientes têm as chaves corretas
        expected_keys = {'W1', 'b1', 'W2', 'b2'}
        assert set(grads.keys()) == expected_keys
        
        # Verifica shapes dos gradientes
        assert grads['W1'].shape == mlp.W1.shape
        assert grads['b1'].shape == mlp.b1.shape
        assert grads['W2'].shape == mlp.W2.shape
        assert grads['b2'].shape == mlp.b2.shape
    
    def test_get_set_params(self):
        """Testa get_params e set_params."""
        mlp = SimpleMLP(input_dim=2, hidden=2, num_classes=2, seed=42)
        
        # Salva parâmetros originais
        original_params = mlp.get_params()
        
        # Modifica parâmetros
        new_params = {
            'W1': np.random.randn(2, 2),
            'b1': np.random.randn(2),
            'W2': np.random.randn(2, 2),
            'b2': np.random.randn(2)
        }
        
        mlp.set_params(new_params)
        
        # Verifica se parâmetros foram alterados
        current_params = mlp.get_params()
        for key in new_params:
            np.testing.assert_array_equal(current_params[key], new_params[key])
        
        # Restaura parâmetros originais
        mlp.set_params(original_params)
        restored_params = mlp.get_params()
        for key in original_params:
            np.testing.assert_array_equal(restored_params[key], original_params[key])
    
    def test_predict(self):
        """Testa predições."""
        mlp = SimpleMLP(input_dim=2, hidden=2, num_classes=3, seed=42)
        
        X = np.random.randn(5, 2)
        predictions = mlp.predict(X)
        
        # Verifica se predições são inteiros
        assert np.issubdtype(predictions.dtype, np.integer)
        
        # Verifica se predições estão no range correto
        assert np.all(predictions >= 0) and np.all(predictions < 3)
        
        # Verifica shape
        assert predictions.shape == (5,)
    
    def test_accuracy(self):
        """Testa cálculo de acurácia."""
        mlp = SimpleMLP(input_dim=2, hidden=2, num_classes=2, seed=42)
        
        X = np.random.randn(4, 2)
        y = np.array([[1, 0], [0, 1], [1, 0], [0, 1]])
        
        accuracy = mlp.accuracy(X, y)
        
        # Verifica se acurácia está entre 0 e 1
        assert 0 <= accuracy <= 1
        
        # Verifica se é um float
        assert isinstance(accuracy, float)
    
    def test_reproducibility(self):
        """Testa reprodutibilidade com seed fixo."""
        mlp1 = SimpleMLP(input_dim=2, hidden=2, num_classes=2, seed=42)
        mlp2 = SimpleMLP(input_dim=2, hidden=2, num_classes=2, seed=42)
        
        # Verifica se parâmetros são idênticos
        params1 = mlp1.get_params()
        params2 = mlp2.get_params()
        
        for key in params1:
            np.testing.assert_array_equal(params1[key], params2[key])
    
    def test_different_seeds(self):
        """Testa se seeds diferentes produzem parâmetros diferentes."""
        mlp1 = SimpleMLP(input_dim=2, hidden=2, num_classes=2, seed=42)
        mlp2 = SimpleMLP(input_dim=2, hidden=2, num_classes=2, seed=123)
        
        params1 = mlp1.get_params()
        params2 = mlp2.get_params()
        
        # Verifica se pelo menos um parâmetro é diferente
        different = False
        for key in params1:
            if not np.array_equal(params1[key], params2[key]):
                different = True
                break
        
        assert different, "Parâmetros devem ser diferentes com seeds diferentes"

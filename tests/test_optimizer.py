"""
Testes unitários para otimizadores.
"""

import pytest
import numpy as np
from src.optimizer import SGD, Adam, RMSprop


class TestSGD:
    """Testes para o otimizador SGD."""
    
    def test_init(self):
        """Testa inicialização do SGD."""
        params = {'W': np.random.randn(2, 2), 'b': np.random.randn(2)}
        optimizer = SGD(params, lr=0.01)
        
        assert optimizer.params == params
        assert optimizer.lr == 0.01
    
    def test_step(self):
        """Testa passo de otimização do SGD."""
        params = {'W': np.random.randn(2, 2), 'b': np.random.randn(2)}
        original_params = {k: v.copy() for k, v in params.items()}
        
        optimizer = SGD(params, lr=0.01)
        grads = {'W': np.random.randn(2, 2), 'b': np.random.randn(2)}
        
        optimizer.step(grads)
        
        # Verifica se parâmetros foram atualizados
        for key in params:
            expected = original_params[key] - 0.01 * grads[key]
            np.testing.assert_array_almost_equal(params[key], expected)
    
    def test_step_partial_grads(self):
        """Testa SGD com gradientes parciais."""
        params = {'W': np.random.randn(2, 2), 'b': np.random.randn(2)}
        original_params = {k: v.copy() for k, v in params.items()}
        
        optimizer = SGD(params, lr=0.01)
        grads = {'W': np.random.randn(2, 2)}  # Sem gradiente para 'b'
        
        optimizer.step(grads)
        
        # Verifica se apenas 'W' foi atualizado
        expected_W = original_params['W'] - 0.01 * grads['W']
        np.testing.assert_array_almost_equal(params['W'], expected_W)
        
        # Verifica se 'b' não foi alterado
        np.testing.assert_array_equal(params['b'], original_params['b'])


class TestAdam:
    """Testes para o otimizador Adam."""
    
    def test_init(self):
        """Testa inicialização do Adam."""
        params = {'W': np.random.randn(2, 2), 'b': np.random.randn(2)}
        optimizer = Adam(params, lr=0.001, beta1=0.9, beta2=0.999, eps=1e-8)
        
        assert optimizer.params == params
        assert optimizer.lr == 0.001
        assert optimizer.beta1 == 0.9
        assert optimizer.beta2 == 0.999
        assert optimizer.eps == 1e-8
        assert optimizer.t == 0
        
        # Verifica se momentos foram inicializados
        for param_name in params:
            assert param_name in optimizer.m
            assert param_name in optimizer.v
            assert optimizer.m[param_name].shape == params[param_name].shape
            assert optimizer.v[param_name].shape == params[param_name].shape
    
    def test_step(self):
        """Testa passo de otimização do Adam."""
        params = {'W': np.random.randn(2, 2), 'b': np.random.randn(2)}
        original_params = {k: v.copy() for k, v in params.items()}
        
        optimizer = Adam(params, lr=0.001)
        grads = {'W': np.random.randn(2, 2), 'b': np.random.randn(2)}
        
        optimizer.step(grads)
        
        # Verifica se contador foi incrementado
        assert optimizer.t == 1
        
        # Verifica se parâmetros foram atualizados
        for key in params:
            assert not np.array_equal(params[key], original_params[key])
    
    def test_bias_correction(self):
        """Testa bias correction do Adam."""
        params = {'W': np.random.randn(2, 2)}
        optimizer = Adam(params, lr=0.001, beta1=0.9, beta2=0.999)
        grads = {'W': np.random.randn(2, 2)}
        
        # Primeiro passo
        optimizer.step(grads)
        assert optimizer.t == 1
        
        # Segundo passo
        optimizer.step(grads)
        assert optimizer.t == 2
        
        # Verifica se bias correction está funcionando
        m_hat = optimizer.m['W'] / (1 - optimizer.beta1 ** optimizer.t)
        v_hat = optimizer.v['W'] / (1 - optimizer.beta2 ** optimizer.t)
        
        # Verifica se bias correction é maior que os momentos originais
        assert np.all(np.abs(m_hat) >= np.abs(optimizer.m['W']))
        assert np.all(v_hat >= optimizer.v['W'])
    
    def test_zero_grad(self):
        """Testa zero_grad do Adam."""
        params = {'W': np.random.randn(2, 2)}
        optimizer = Adam(params, lr=0.001)
        grads = {'W': np.random.randn(2, 2)}
        
        # Executa alguns passos
        for _ in range(3):
            optimizer.step(grads)
        
        assert optimizer.t == 3
        
        # Zera gradientes
        optimizer.zero_grad()
        
        # Verifica se momentos foram zerados
        for param_name in optimizer.m:
            assert np.all(optimizer.m[param_name] == 0)
            assert np.all(optimizer.v[param_name] == 0)
        
        # Verifica se contador foi resetado
        assert optimizer.t == 0


class TestRMSprop:
    """Testes para o otimizador RMSprop."""
    
    def test_init(self):
        """Testa inicialização do RMSprop."""
        params = {'W': np.random.randn(2, 2), 'b': np.random.randn(2)}
        optimizer = RMSprop(params, lr=0.001, alpha=0.99, eps=1e-8)
        
        assert optimizer.params == params
        assert optimizer.lr == 0.001
        assert optimizer.alpha == 0.99
        assert optimizer.eps == 1e-8
        
        # Verifica se v foi inicializado
        for param_name in params:
            assert param_name in optimizer.v
            assert optimizer.v[param_name].shape == params[param_name].shape
    
    def test_step(self):
        """Testa passo de otimização do RMSprop."""
        params = {'W': np.random.randn(2, 2), 'b': np.random.randn(2)}
        original_params = {k: v.copy() for k, v in params.items()}
        
        optimizer = RMSprop(params, lr=0.001)
        grads = {'W': np.random.randn(2, 2), 'b': np.random.randn(2)}
        
        optimizer.step(grads)
        
        # Verifica se parâmetros foram atualizados
        for key in params:
            assert not np.array_equal(params[key], original_params[key])
    
    def test_zero_grad(self):
        """Testa zero_grad do RMSprop."""
        params = {'W': np.random.randn(2, 2)}
        optimizer = RMSprop(params, lr=0.001)
        grads = {'W': np.random.randn(2, 2)}
        
        # Executa alguns passos
        for _ in range(3):
            optimizer.step(grads)
        
        # Zera gradientes
        optimizer.zero_grad()
        
        # Verifica se v foi zerado
        for param_name in optimizer.v:
            assert np.all(optimizer.v[param_name] == 0)


class TestOptimizerIntegration:
    """Testes de integração dos otimizadores."""
    
    def test_sgd_convergence(self):
        """Testa convergência do SGD em problema simples."""
        # Problema: encontrar mínimo de f(x) = x^2
        params = {'x': np.array([10.0])}
        optimizer = SGD(params, lr=0.1)
        
        for _ in range(100):
            # Gradiente de f(x) = x^2 é 2x
            grads = {'x': np.array([2 * params['x'][0]])}
            optimizer.step(grads)
        
        # Verifica se convergiu para próximo de 0
        assert abs(params['x'][0]) < 0.1
    
    def test_adam_convergence(self):
        """Testa convergência do Adam em problema simples."""
        # Problema: encontrar mínimo de f(x) = x^2
        params = {'x': np.array([10.0])}
        optimizer = Adam(params, lr=0.1)
        
        for _ in range(50):
            # Gradiente de f(x) = x^2 é 2x
            grads = {'x': np.array([2 * params['x'][0]])}
            optimizer.step(grads)
        
        # Verifica se convergiu para próximo de 0
        assert abs(params['x'][0]) < 0.1
    
    def test_optimizer_comparison(self):
        """Compara comportamento dos otimizadores."""
        # Problema: encontrar mínimo de f(x) = x^2 + y^2
        params_sgd = {'x': np.array([5.0]), 'y': np.array([5.0])}
        params_adam = {'x': np.array([5.0]), 'y': np.array([5.0])}
        
        optimizer_sgd = SGD(params_sgd, lr=0.1)
        optimizer_adam = Adam(params_adam, lr=0.1)
        
        for _ in range(50):
            # Gradientes
            grads = {
                'x': np.array([2 * params_sgd['x'][0]]),
                'y': np.array([2 * params_sgd['y'][0]])
            }
            
            optimizer_sgd.step(grads)
            optimizer_adam.step(grads)
        
        # Verifica se ambos convergiram
        sgd_norm = np.sqrt(params_sgd['x'][0]**2 + params_sgd['y'][0]**2)
        adam_norm = np.sqrt(params_adam['x'][0]**2 + params_adam['y'][0]**2)
        
        assert sgd_norm < 0.5
        assert adam_norm < 0.5

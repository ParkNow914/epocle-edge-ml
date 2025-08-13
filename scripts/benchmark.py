#!/usr/bin/env python3
"""
Script de benchmark para Epocle Edge ML
Compara performance de diferentes componentes
"""

import os
import sys
import time
import numpy as np

# Adicionar src ao path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src import SimpleMLP, SGD, Adam, RMSprop
from examples.synthetic_data import generate_synthetic_data


def benchmark_optimizers():
    """Benchmark de diferentes otimizadores"""
    print("⚡ Benchmark: Otimizadores")
    print("=" * 50)
    
    # Criar dados
    X, y = generate_synthetic_data(1000, 20, 3)
    print(f"Dados: {X.shape[0]} amostras, {X.shape[1]} features")
    
    # Testar diferentes otimizadores
    optimizers = {
        'SGD': SGD,
        'Adam': Adam,
        'RMSprop': RMSprop
    }
    
    results = {}
    
    for name, optimizer_class in optimizers.items():
        print(f"\n🔧 Testando {name}...")
        
        # Criar modelo e otimizador
        model = SimpleMLP(input_dim=20, hidden=64, num_classes=3)
        optimizer = optimizer_class(model.get_params(), lr=0.01)
        
        # Treinamento
        start_time = time.time()
        losses = []
        
        for epoch in range(50):
            loss, gradients = model.loss_and_grad(X, y)
            optimizer.step(gradients)
            losses.append(loss)
        
        training_time = time.time() - start_time
        
        # Resultados
        final_loss = losses[-1]
        loss_reduction = losses[0] - losses[-1]
        
        results[name] = {
            'final_loss': final_loss,
            'loss_reduction': loss_reduction,
            'training_time': training_time,
            'convergence_rate': loss_reduction / training_time
        }
        
        print(f"  Final Loss: {final_loss:.6f}")
        print(f"  Loss Reduction: {loss_reduction:.6f}")
        print(f"  Training Time: {training_time:.3f}s")
        print(f"  Convergence Rate: {loss_reduction/training_time:.3f}")
    
    return results


def benchmark_model_sizes():
    """Benchmark de diferentes tamanhos de modelo"""
    print("\n📏 Benchmark: Tamanhos de Modelo")
    print("=" * 50)
    
    # Criar dados
    X, y = generate_synthetic_data(500, 20, 3)
    
    # Diferentes configurações
    configs = [
        (20, 32, 3),   # Pequeno
        (20, 64, 3),   # Médio
        (20, 128, 3),  # Grande
        (20, 256, 3),  # Muito grande
    ]
    
    results = {}
    
    for input_dim, hidden, num_classes in configs:
        print(f"\n🔍 Testando modelo {input_dim}x{hidden}x{num_classes}...")
        
        # Criar modelo
        model = SimpleMLP(input_dim=input_dim, hidden=hidden, num_classes=num_classes)
        optimizer = SGD(model.get_params(), lr=0.01)
        
        # Contar parâmetros
        params = model.get_params()
        total_params = sum(p.size for p in params.values())
        
        # Forward pass
        start_time = time.time()
        for _ in range(100):
            _ = model.forward(X[:, :input_dim])
        forward_time = time.time() - start_time
        
        # Backward pass
        start_time = time.time()
        for _ in range(100):
            _, _ = model.loss_and_grad(X[:, :input_dim], y)
        backward_time = time.time() - start_time
        
        results[f"{input_dim}x{hidden}x{num_classes}"] = {
            'total_params': total_params,
            'forward_time': forward_time,
            'backward_time': backward_time,
            'total_time': forward_time + backward_time
        }
        
        print(f"  Parâmetros: {total_params:,}")
        print(f"  Forward (100x): {forward_time:.3f}s")
        print(f"  Backward (100x): {backward_time:.3f}s")
        print(f"  Total (100x): {forward_time + backward_time:.3f}s")
    
    return results


def print_summary(opt_results, model_results):
    """Imprimir resumo dos benchmarks"""
    print("\n📊 RESUMO DOS BENCHMARKS")
    print("=" * 60)
    
    # Melhor otimizador
    best_optimizer = min(opt_results.items(), key=lambda x: x[1]['final_loss'])
    print(f"🏆 Melhor Otimizador: {best_optimizer[0]}")
    print(f"   Loss Final: {best_optimizer[1]['final_loss']:.6f}")
    
    # Modelo mais rápido
    fastest_model = min(model_results.items(), key=lambda x: x[1]['total_time'])
    print(f"⚡ Modelo Mais Rápido: {fastest_model[0]}")
    print(f"   Tempo Total: {fastest_model[1]['total_time']:.3f}s")
    
    # Modelo com mais parâmetros
    largest_model = max(model_results.items(), key=lambda x: x[1]['total_params'])
    print(f"🐘 Maior Modelo: {largest_model[0]}")
    print(f"   Parâmetros: {largest_model[1]['total_params']:,}")


def main():
    """Função principal"""
    print("🚀 EPOCLE EDGE ML - BENCHMARK COMPLETO")
    print("=" * 60)
    
    try:
        # Executar benchmarks
        opt_results = benchmark_optimizers()
        model_results = benchmark_model_sizes()
        
        # Imprimir resumo
        print_summary(opt_results, model_results)
        
        print("\n🎉 BENCHMARK CONCLUÍDO COM SUCESSO!")
        
    except Exception as e:
        print(f"\n❌ Erro durante benchmark: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())

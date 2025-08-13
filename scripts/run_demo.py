#!/usr/bin/env python3
"""
Script principal de demonstração para Epocle Edge ML
Executa todos os componentes principais do sistema
"""

import os
import sys
import time
import numpy as np

# Adicionar src ao path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src import SimpleMLP, SGD, Adam, EWC
from src.replay_buffer import ReplayBuffer
from examples.synthetic_data import generate_synthetic_data

def demo_numpy_mlp():
    """Demonstração do MLP NumPy"""
    print("🔬 Demonstração: MLP NumPy")
    print("=" * 50)
    
    # Criar dados
    X, y = generate_synthetic_data(1000, 20, 3)
    print(f"Dados: X shape {X.shape}, y shape {y.shape}")
    
    # Criar modelo
    model = SimpleMLP(input_dim=20, hidden=64, num_classes=3)
    optimizer = SGD(model.get_params(), lr=0.01)
    
    # Treinamento
    print("\n📚 Treinando modelo...")
    losses = []
    for epoch in range(30):
        loss, gradients = model.loss_and_grad(X, y)
        optimizer.step(gradients)
        losses.append(loss)
        
        if epoch % 10 == 0:
            print(f"  Epoch {epoch}: Loss = {loss:.4f}")
    
    # Avaliação
    predictions = model.predict(X)
    accuracy = np.mean(np.argmax(predictions, axis=1) == np.argmax(y, axis=1))
    print(f"\n✅ Acurácia final: {accuracy:.4f}")
    print(f"📉 Loss final: {losses[-1]:.4f}")
    
    return model, losses

def demo_replay_buffer():
    """Demonstração do Replay Buffer"""
    print("\n🔄 Demonstração: Replay Buffer")
    print("=" * 50)
    
    # Criar buffer
    buffer = ReplayBuffer(capacity=1000)
    
    # Adicionar experiências
    for i in range(100):
        X = np.random.randn(10, 20)
        y = np.eye(3)[np.random.randint(0, 3, 10)]
        buffer.add(X, y, importance=i/100)
    
    print(f"Buffer preenchido com {len(buffer)} experiências")
    
    # Amostrar
    X_batch, y_batch = buffer.sample(batch_size=32)
    print(f"Amostra: X shape {X_batch.shape}, y shape {y_batch.shape}")
    
    # Estatísticas
    stats = buffer.get_stats()
    print(f"Estatísticas: {stats}")
    
    return buffer

def demo_ewc():
    """Demonstração do EWC"""
    print("\n🛡️ Demonstração: Elastic Weight Consolidation (EWC)")
    print("=" * 50)
    
    # Criar modelo e dados
    X, y = generate_synthetic_data(500, 20, 3)
    model = SimpleMLP(input_dim=20, hidden=32, num_classes=3)
    
    # Registrar modelo no EWC
    ewc = EWC(model, lambda_=1000.0)
    ewc.register(X, y, num_samples=100)
    
    print(f"EWC registrado com {len(ewc.anchor_params)} parâmetros")
    
    # Calcular penalty
    penalty, gradients = ewc.penalty()
    print(f"Penalty EWC: {penalty:.6f}")
    
    return ewc

def main():
    """Função principal"""
    print("🚀 EPOCLE EDGE ML - DEMONSTRAÇÃO COMPLETA")
    print("=" * 60)
    print("Este script demonstra todos os componentes principais")
    print("do sistema de Continual Learning implementado em NumPy")
    print("=" * 60)
    
    try:
        # Executar demonstrações
        model, losses = demo_numpy_mlp()
        buffer = demo_replay_buffer()
        ewc = demo_ewc()
        
        print("\n🎉 TODAS AS DEMONSTRAÇÕES CONCLUÍDAS COM SUCESSO!")
        print("\n📊 Resumo:")
        print(f"  • MLP NumPy: {len(losses)} epochs treinados")
        print(f"  • Replay Buffer: {len(buffer)} experiências")
        print(f"  • EWC: {len(ewc.anchor_params)} parâmetros protegidos")
        
    except Exception as e:
        print(f"\n❌ Erro durante demonstração: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())

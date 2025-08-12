#!/usr/bin/env python3
"""
Pipeline de treino online para Continual Learning.
Implementa treino incremental com Replay Buffer e EWC.
"""

import argparse
import numpy as np
import os
import json
import time
from datetime import datetime
from typing import Dict, List, Tuple, Optional

from src.numpy_nn import SimpleMLP
from src.optimizer import SGD, Adam, RMSprop
from src.replay_buffer import ReplayBuffer
from src.ewc import EWC, compute_ewc_loss


def create_timestamp_dir(base_dir: str = "artifacts") -> str:
    """
    Cria diretório com timestamp para artifacts.
    
    Args:
        base_dir: Diretório base
        
    Returns:
        Caminho do diretório criado
    """
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    artifacts_dir = os.path.join(base_dir, timestamp)
    os.makedirs(artifacts_dir, exist_ok=True)
    return artifacts_dir


def load_data(data_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Carrega dados do arquivo .npz.
    
    Args:
        data_path: Caminho para o arquivo de dados
        
    Returns:
        Tuple com (X, y)
    """
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Arquivo de dados não encontrado: {data_path}")
    
    data = np.load(data_path)
    X = data['X']
    y = data['y']
    
    print(f"Dados carregados: {X.shape[0]} amostras, {X.shape[1]} features, {y.shape[1]} classes")
    return X, y


def create_model(input_dim: int, hidden: int, num_classes: int, seed: int) -> SimpleMLP:
    """
    Cria modelo MLP.
    
    Args:
        input_dim: Dimensão da entrada
        hidden: Número de neurônios na camada oculta
        num_classes: Número de classes
        seed: Seed para reprodutibilidade
        
    Returns:
        Modelo MLP inicializado
    """
    model = SimpleMLP(input_dim=input_dim, hidden=hidden, num_classes=num_classes, seed=seed)
    print(f"Modelo criado: {input_dim} -> {hidden} -> {num_classes}")
    return model


def create_optimizer(optimizer_name: str, params: Dict, lr: float) -> object:
    """
    Cria otimizador.
    
    Args:
        optimizer_name: Nome do otimizador
        params: Parâmetros do modelo
        lr: Learning rate
        
    Returns:
        Otimizador
    """
    if optimizer_name.lower() == "sgd":
        optimizer = SGD(params, lr=lr)
    elif optimizer_name.lower() == "adam":
        optimizer = Adam(params, lr=lr)
    elif optimizer_name.lower() == "rmsprop":
        optimizer = RMSprop(params, lr=lr)
    else:
        raise ValueError(f"Otimizador não reconhecido: {optimizer_name}")
    
    print(f"Otimizador criado: {optimizer_name} com lr={lr}")
    return optimizer


def train_epoch(
    model: SimpleMLP,
    optimizer: object,
    X: np.ndarray,
    y: np.ndarray,
    batch_size: int,
    replay_buffer: Optional[ReplayBuffer] = None,
    ewc: Optional[EWC] = None,
    dp_sigma: float = 0.0
) -> Tuple[float, float]:
    """
    Treina uma época.
    
    Args:
        model: Modelo MLP
        optimizer: Otimizador
        X: Dados de entrada
        y: Labels
        batch_size: Tamanho do batch
        replay_buffer: Buffer de replay (opcional)
        ewc: EWC (opcional)
        dp_sigma: Sigma para differential privacy
        
    Returns:
        Tuple com (loss_epoch, accuracy_epoch)
    """
    num_samples = X.shape[0]
    num_batches = (num_samples + batch_size - 1) // batch_size
    
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    
    # Embaralha dados
    indices = np.random.permutation(num_samples)
    
    for batch_idx in range(num_batches):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, num_samples)
        
        batch_indices = indices[start_idx:end_idx]
        X_batch = X[batch_indices]
        y_batch = y[batch_indices]
        
        # Adiciona ruído para differential privacy
        if dp_sigma > 0:
            noise = np.random.normal(0, dp_sigma, X_batch.shape)
            X_batch = X_batch + noise
        
        # Computa loss e gradientes
        if ewc is not None and ewc.is_registered():
            loss, grads = compute_ewc_loss(model, X_batch, y_batch, ewc)
        else:
            loss, grads = model.loss_and_grad(X_batch, y_batch)
        
        # Atualiza parâmetros
        optimizer.step(grads)
        
        # Atualiza EWC online se aplicável
        if ewc is not None and hasattr(ewc, 'update_fisher_online'):
            ewc.update_fisher_online(grads)
        
        # Adiciona ao replay buffer
        if replay_buffer is not None:
            # Probabilidade de armazenar no buffer
            if np.random.random() < 0.5:  # p_store
                replay_buffer.add(X_batch, y_batch)
        
        # Estatísticas
        total_loss += loss * (end_idx - start_idx)
        predictions = model.predict(X_batch)
        y_true = np.argmax(y_batch, axis=1)
        total_correct += np.sum(predictions == y_true)
        total_samples += (end_idx - start_idx)
    
    # Média da época
    avg_loss = total_loss / total_samples
    avg_accuracy = total_correct / total_samples
    
    return avg_loss, avg_accuracy


def train_with_replay(
    model: SimpleMLP,
    optimizer: object,
    X: np.ndarray,
    y: np.ndarray,
    replay_buffer: ReplayBuffer,
    replay_batch_size: int,
    num_replay_steps: int = 1
) -> Tuple[float, float]:
    """
    Treina com dados do replay buffer.
    
    Args:
        model: Modelo MLP
        optimizer: Otimizador
        X: Dados atuais
        y: Labels atuais
        replay_buffer: Buffer de replay
        replay_batch_size: Tamanho do batch de replay
        num_replay_steps: Número de passos de replay
        
    Returns:
        Tuple com (loss_replay, accuracy_replay)
    """
    if len(replay_buffer) == 0:
        return 0.0, 0.0
    
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    
    for _ in range(num_replay_steps):
        if len(replay_buffer) >= replay_batch_size:
            X_replay, y_replay = replay_buffer.sample(replay_batch_size, strategy="uniform")
            
            # Treina com dados de replay
            loss, grads = model.loss_and_grad(X_replay, y_replay)
            optimizer.step(grads)
            
            # Estatísticas
            total_loss += loss * replay_batch_size
            predictions = model.predict(X_replay)
            y_true = np.argmax(y_replay, axis=1)
            total_correct += np.sum(predictions == y_true)
            total_samples += replay_batch_size
    
    if total_samples > 0:
        avg_loss = total_loss / total_samples
        avg_accuracy = total_correct / total_samples
        return avg_loss, avg_accuracy
    else:
        return 0.0, 0.0


def save_checkpoint(
    model: SimpleMLP,
    optimizer: object,
    replay_buffer: Optional[ReplayBuffer],
    ewc: Optional[EWC],
    epoch: int,
    artifacts_dir: str
) -> None:
    """
    Salva checkpoint do modelo.
    
    Args:
        model: Modelo MLP
        optimizer: Otimizador
        replay_buffer: Buffer de replay
        ewc: EWC
        epoch: Época atual
        artifacts_dir: Diretório para artifacts
    """
    # Salva parâmetros do modelo
    model_params = model.get_params()
    model_path = os.path.join(artifacts_dir, f"model_epoch_{epoch}.npz")
    
    np.savez_compressed(
        model_path,
        **{f"{k}_{epoch}": v for k, v in model_params.items()}
    )
    
    # Salva estatísticas do replay buffer
    if replay_buffer is not None:
        buffer_stats = replay_buffer.get_stats()
        buffer_path = os.path.join(artifacts_dir, f"buffer_epoch_{epoch}.json")
        
        with open(buffer_path, 'w') as f:
            json.dump(buffer_stats, f, indent=2)
    
    # Salva estatísticas do EWC
    if ewc is not None and ewc.is_registered():
        ewc_stats = ewc.get_fisher_stats()
        ewc_path = os.path.join(artifacts_dir, f"ewc_epoch_{epoch}.json")
        
        with open(ewc_path, 'w') as f:
            json.dump(ewc_stats, f, indent=2)
    
    print(f"Checkpoint salvo: época {epoch}")


def main():
    """Função principal."""
    parser = argparse.ArgumentParser(
        description="Pipeline de treino online para Continual Learning"
    )
    
    # Parâmetros de dados
    parser.add_argument(
        "--data", 
        type=str, 
        required=True,
        help="Caminho para arquivo de dados (.npz)"
    )
    
    # Parâmetros do modelo
    parser.add_argument(
        "--hidden", 
        type=int, 
        default=64,
        help="Número de neurônios na camada oculta"
    )
    
    # Parâmetros de treino
    parser.add_argument(
        "--epochs", 
        type=int, 
        default=10,
        help="Número de épocas"
    )
    
    parser.add_argument(
        "--batch_size", 
        type=int, 
        default=32,
        help="Tamanho do batch"
    )
    
    parser.add_argument(
        "--lr", 
        type=float, 
        default=0.001,
        help="Learning rate"
    )
    
    parser.add_argument(
        "--optimizer", 
        type=str, 
        default="adam",
        choices=["sgd", "adam", "rmsprop"],
        help="Otimizador"
    )
    
    # Parâmetros de replay
    parser.add_argument(
        "--replay_capacity", 
        type=int, 
        default=1000,
        help="Capacidade do replay buffer"
    )
    
    parser.add_argument(
        "--replay_batch", 
        type=int, 
        default=32,
        help="Tamanho do batch de replay"
    )
    
    parser.add_argument(
        "--p_store", 
        type=float, 
        default=0.5,
        help="Probabilidade de armazenar no replay buffer"
    )
    
    # Parâmetros de EWC
    parser.add_argument(
        "--use_ewc", 
        type=str, 
        default="False",
        help="Usar EWC (True/False)"
    )
    
    parser.add_argument(
        "--ewc_lambda", 
        type=float, 
        default=100.0,
        help="Peso do penalty EWC"
    )
    
    parser.add_argument(
        "--ewc_samples", 
        type=int, 
        default=100,
        help="Número de amostras para computar Fisher"
    )
    
    # Parâmetros de differential privacy
    parser.add_argument(
        "--dp_sigma", 
        type=float, 
        default=0.0,
        help="Sigma para differential privacy (0 = sem DP)"
    )
    
    # Parâmetros gerais
    parser.add_argument(
        "--seed", 
        type=int, 
        default=42,
        help="Seed para reprodutibilidade"
    )
    
    parser.add_argument(
        "--checkpoint_freq", 
        type=int, 
        default=5,
        help="Frequência de checkpoint (épocas)"
    )
    
    parser.add_argument(
        "--verbose", 
        action="store_true",
        help="Modo verboso"
    )
    
    args = parser.parse_args()
    
    # Converte string para boolean
    use_ewc = args.use_ewc.lower() == "true"
    
    # Set seed
    np.random.seed(args.seed)
    
    # Cria diretório de artifacts
    artifacts_dir = create_timestamp_dir()
    print(f"Artifacts serão salvos em: {artifacts_dir}")
    
    # Carrega dados
    X, y = load_data(args.data)
    input_dim = X.shape[1]
    num_classes = y.shape[1]
    
    # Cria modelo
    model = create_model(input_dim, args.hidden, num_classes, args.seed)
    
    # Cria otimizador
    optimizer = create_optimizer(args.optimizer, model.get_params(), args.lr)
    
    # Cria replay buffer
    replay_buffer = ReplayBuffer(capacity=args.replay_capacity, seed=args.seed)
    print(f"Replay buffer criado com capacidade {args.replay_capacity}")
    
    # Cria EWC se solicitado
    ewc = None
    if use_ewc:
        ewc = EWC(model, lambda_=args.ewc_lambda)
        print(f"EWC criado com lambda={args.ewc_lambda}")
        
        # Registra dados para EWC
        ewc.register(X, y, num_samples=args.ewc_samples)
    
    # Histórico de treino
    history = {
        'epoch': [],
        'loss': [],
        'accuracy': [],
        'replay_loss': [],
        'replay_accuracy': [],
        'time': []
    }
    
    print(f"\n=== Iniciando Treino Online ===")
    print(f"Épocas: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.lr}")
    print(f"Otimizador: {args.optimizer}")
    print(f"Replay: {args.replay_capacity} (batch: {args.replay_batch})")
    print(f"EWC: {use_ewc}")
    print(f"DP sigma: {args.dp_sigma}")
    
    # Loop de treino
    for epoch in range(args.epochs):
        start_time = time.time()
        
        # Treina uma época
        loss, accuracy = train_epoch(
            model, optimizer, X, y, args.batch_size,
            replay_buffer, ewc, args.dp_sigma
        )
        
        # Treina com replay
        replay_loss, replay_accuracy = train_with_replay(
            model, optimizer, X, y, replay_buffer, args.replay_batch
        )
        
        epoch_time = time.time() - start_time
        
        # Salva histórico
        history['epoch'].append(epoch + 1)
        history['loss'].append(loss)
        history['accuracy'].append(accuracy)
        history['replay_loss'].append(replay_loss)
        history['replay_accuracy'].append(replay_accuracy)
        history['time'].append(epoch_time)
        
        # Imprime progresso
        if args.verbose or (epoch + 1) % 5 == 0:
            print(f"Época {epoch + 1:2d}/{args.epochs}: "
                  f"Loss={loss:.4f}, Acc={accuracy:.4f}, "
                  f"Replay Loss={replay_loss:.4f}, Replay Acc={replay_accuracy:.4f}, "
                  f"Tempo={epoch_time:.2f}s")
        
        # Checkpoint
        if (epoch + 1) % args.checkpoint_freq == 0:
            save_checkpoint(model, optimizer, replay_buffer, ewc, epoch + 1, artifacts_dir)
    
    # Salva modelo final
    final_model_path = os.path.join(artifacts_dir, "model_final.npz")
    final_params = model.get_params()
    np.savez_compressed(final_model_path, **final_params)
    
    # Salva histórico
    history_path = os.path.join(artifacts_dir, "training_history.json")
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    
    # Salva configuração
    config = {
        'input_dim': input_dim,
        'hidden': args.hidden,
        'num_classes': num_classes,
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'lr': args.lr,
        'optimizer': args.optimizer,
        'replay_capacity': args.replay_capacity,
        'replay_batch': args.replay_batch,
        'use_ewc': use_ewc,
        'ewc_lambda': args.ewc_lambda,
        'dp_sigma': args.dp_sigma,
        'seed': args.seed
    }
    
    config_path = os.path.join(artifacts_dir, "config.json")
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    # Estatísticas finais
    print(f"\n=== Treino Concluído ===")
    print(f"Modelo final salvo em: {final_model_path}")
    print(f"Histórico salvo em: {history_path}")
    print(f"Configuração salva em: {config_path}")
    
    final_acc = history['accuracy'][-1]
    final_replay_acc = history['replay_accuracy'][-1]
    total_time = sum(history['time'])
    
    print(f"Acurácia final: {final_acc:.4f}")
    print(f"Acurácia replay final: {final_replay_acc:.4f}")
    print(f"Tempo total: {total_time:.2f}s")
    print(f"Tempo médio por época: {total_time/args.epochs:.2f}s")
    
    # Estatísticas do replay buffer
    buffer_stats = replay_buffer.get_stats()
    print(f"\nReplay Buffer:")
    print(f"  Tamanho: {buffer_stats['size']}/{buffer_stats['capacity']}")
    print(f"  Utilização: {buffer_stats['utilization']:.2%}")
    print(f"  Total adicionado: {buffer_stats['total_added']}")
    
    if ewc is not None and ewc.is_registered():
        ewc_stats = ewc.get_fisher_stats()
        print(f"\nEWC:")
        print(f"  Fisher diagonal médio: {ewc_stats.get('W1_mean', 0):.6f}")
        print(f"  Total parâmetros: {sum(1 for k in ewc_stats if k.endswith('_mean'))}")


if __name__ == "__main__":
    exit(main())

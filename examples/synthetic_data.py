#!/usr/bin/env python3
"""
Gerador de dados sintéticos para treino e teste.
Gera datasets multiclasse com distribuições controladas.
"""

import argparse
import os
from typing import Tuple

import numpy as np


def generate_synthetic_data(
    num_classes: int,
    samples_per_class: int,
    input_dim: int,
    seed: int = 42,
    noise_std: float = 0.1,
    class_separation: float = 2.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Gera dados sintéticos multiclasse.

    Args:
        num_classes: Número de classes
        samples_per_class: Número de amostras por classe
        input_dim: Dimensão da entrada
        seed: Seed para reprodutibilidade
        noise_std: Desvio padrão do ruído
        class_separation: Separação entre centros das classes

    Returns:
        Tuple com (X, y) onde y é one-hot encoded
    """
    np.random.seed(seed)

    total_samples = num_classes * samples_per_class

    # Gera centros das classes
    class_centers = np.random.randn(num_classes, input_dim) * class_separation

    # Gera dados para cada classe
    X = []
    y = []

    for class_idx in range(num_classes):
        # Gera amostras ao redor do centro da classe
        class_samples = np.random.randn(samples_per_class, input_dim) * noise_std
        class_samples += class_centers[class_idx]

        X.append(class_samples)

        # Cria one-hot encoding
        class_labels = np.zeros((samples_per_class, num_classes))
        class_labels[:, class_idx] = 1
        y.append(class_labels)

    # Concatena todas as classes
    X = np.vstack(X)
    y = np.vstack(y)

    # Embaralha os dados
    indices = np.random.permutation(total_samples)
    X = X[indices]
    y = y[indices]

    return X, y


def save_data(X: np.ndarray, y: np.ndarray, output_path: str) -> None:
    """
    Salva dados em formato .npz.

    Args:
        X: Features
        y: Labels (one-hot encoded)
        output_path: Caminho para salvar
    """
    # Cria diretório se não existir
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Salva dados
    np.savez_compressed(output_path, X=X, y=y)
    print(f"Dados salvos em: {output_path}")


def print_dataset_info(X: np.ndarray, y: np.ndarray) -> None:
    """
    Imprime informações sobre o dataset.

    Args:
        X: Features
        y: Labels (one-hot encoded)
    """
    print("\n=== Informações do Dataset ===")
    print(f"Shape de X: {X.shape}")
    print(f"Shape de y: {y.shape}")
    print(f"Número de classes: {y.shape[1]}")
    print(f"Amostras por classe: {X.shape[0] // y.shape[1]}")
    print(f"Dimensão da entrada: {X.shape[1]}")

    # Estatísticas básicas
    print(f"\nEstatísticas de X:")
    print(f"  Média: {X.mean():.4f}")
    print(f"  Desvio padrão: {X.std():.4f}")
    print(f"  Mínimo: {X.min():.4f}")
    print(f"  Máximo: {X.max():.4f}")

    # Distribuição das classes
    class_counts = np.sum(y, axis=0)
    print(f"\nDistribuição das classes:")
    for i, count in enumerate(class_counts):
        print(f"  Classe {i}: {int(count)} amostras")


def main():
    """Função principal."""
    parser = argparse.ArgumentParser(
        description="Gerador de dados sintéticos para treino e teste"
    )

    parser.add_argument(
        "--out",
        type=str,
        default="data/synthetic.npz",
        help="Caminho de saída para os dados (.npz)",
    )

    parser.add_argument("--classes", type=int, default=3, help="Número de classes")

    parser.add_argument(
        "--per_class", type=int, default=200, help="Número de amostras por classe"
    )

    parser.add_argument("--dim", type=int, default=20, help="Dimensão da entrada")

    parser.add_argument(
        "--seed", type=int, default=42, help="Seed para reprodutibilidade"
    )

    parser.add_argument(
        "--noise", type=float, default=0.1, help="Desvio padrão do ruído"
    )

    parser.add_argument(
        "--separation",
        type=float,
        default=2.0,
        help="Separação entre centros das classes",
    )

    parser.add_argument("--verbose", action="store_true", help="Modo verboso")

    args = parser.parse_args()

    if args.verbose:
        print("=== Gerador de Dados Sintéticos ===")
        print(f"Classes: {args.classes}")
        print(f"Amostras por classe: {args.per_class}")
        print(f"Dimensão: {args.dim}")
        print(f"Seed: {args.seed}")
        print(f"Ruído: {args.noise}")
        print(f"Separação: {args.separation}")
        print(f"Saída: {args.out}")

    try:
        # Gera dados
        X, y = generate_synthetic_data(
            num_classes=args.classes,
            samples_per_class=args.per_class,
            input_dim=args.dim,
            seed=args.seed,
            noise_std=args.noise,
            class_separation=args.separation,
        )

        # Salva dados
        save_data(X, y, args.out)

        # Imprime informações
        print_dataset_info(X, y)

        if args.verbose:
            print(f"\nDataset gerado com sucesso!")
            print(f"Total de amostras: {X.shape[0]}")
            print(f"Memória utilizada: {X.nbytes + y.nbytes} bytes")

    except Exception as e:
        print(f"Erro ao gerar dados: {e}")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())

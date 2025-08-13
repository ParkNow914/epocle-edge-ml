#!/usr/bin/env python3
"""
Reptile Meta-Learning Algorithm Implementation

This script implements the Reptile algorithm for few-shot learning using PyTorch.
Reptile is a simple meta-learning algorithm that learns to initialize neural
networks for fast adaptation to new tasks.

Author: Senior Autonomous Engineering Agent
License: MIT
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from examples.synthetic_data import generate_synthetic_data


class SimpleMLP(nn.Module):
    """Simple MLP for few-shot learning tasks"""

    def __init__(self, input_dim: int, hidden_dim: int, num_classes: int):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class ReptileTrainer:
    """Reptile meta-learning trainer"""

    def __init__(
        self,
        model: nn.Module,
        meta_lr: float = 0.01,
        inner_lr: float = 0.1,
        inner_steps: int = 5,
        device: str = "cpu",
    ):
        self.model = model.to(device)
        self.meta_lr = meta_lr
        self.inner_lr = inner_lr
        self.inner_steps = inner_steps
        self.device = device

        # Store initial parameters
        self.initial_params = self._get_params()

    def _get_params(self) -> Dict[str, torch.Tensor]:
        """Get current model parameters"""
        return {name: param.clone() for name, param in self.model.named_parameters()}

    def _set_params(self, params: Dict[str, torch.Tensor]) -> None:
        """Set model parameters"""
        for name, param in self.model.named_parameters():
            if name in params:
                param.data = params[name].data.clone()

    def _inner_loop(
        self, X: torch.Tensor, y: torch.Tensor
    ) -> Tuple[float, Dict[str, torch.Tensor]]:
        """Inner loop optimization for a single task"""
        # Create a copy of the model for this task
        task_model = type(self.model)(
            self.model.layers[0].in_features,
            self.model.layers[2].out_features,
            self.model.layers[-1].out_features,
        ).to(self.device)

        # Copy initial parameters
        for name, param in task_model.named_parameters():
            if name in self.initial_params:
                param.data = self.initial_params[name].data.clone()

        # Inner loop optimization
        optimizer = optim.SGD(task_model.parameters(), lr=self.inner_lr)
        criterion = nn.CrossEntropyLoss()

        for _ in range(self.inner_steps):
            optimizer.zero_grad()
            outputs = task_model(X)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()

        # Return final loss and updated parameters
        final_loss = criterion(task_model(X), y).item()
        final_params = {
            name: param.clone() for name, param in task_model.named_parameters()
        }

        return final_loss, final_params

    def meta_step(
        self, tasks: List[Tuple[torch.Tensor, torch.Tensor]]
    ) -> Tuple[float, Dict[str, float]]:
        """Perform one meta-learning step"""
        meta_loss = 0.0
        task_losses = []

        # Store initial parameters
        initial_params = self._get_params()

        # Process each task
        for X, y in tasks:
            loss, final_params = self._inner_loop(X, y)
            meta_loss += loss
            task_losses.append(loss)

            # Reptile update: move towards task-adapted parameters
            for name, param in self.model.named_parameters():
                if name in final_params:
                    param.data += self.meta_lr * (
                        final_params[name] - initial_params[name]
                    )

        # Average meta loss
        meta_loss /= len(tasks)

        # Calculate statistics
        stats = {
            "meta_loss": meta_loss,
            "avg_task_loss": np.mean(task_losses),
            "std_task_loss": np.std(task_losses),
            "min_task_loss": np.min(task_losses),
            "max_task_loss": np.max(task_losses),
        }

        return meta_loss, stats


def generate_few_shot_tasks(
    X: np.ndarray,
    y: np.ndarray,
    num_tasks: int,
    shots_per_class: int,
    query_per_class: int,
    num_classes: int,
    seed: int = 42,
) -> List[Tuple[torch.Tensor, torch.Tensor]]:
    """Generate few-shot learning tasks"""
    np.random.seed(seed)
    torch.manual_seed(seed)

    tasks = []

    for _ in range(num_tasks):
        # Randomly select classes for this task
        task_classes = np.random.choice(num_classes, size=num_classes, replace=False)

        # Collect support and query samples
        support_X, support_y = [], []
        query_X, query_y = [], []

        for class_idx in task_classes:
            # Get samples for this class
            class_mask = y == class_idx
            class_samples = X[class_mask]

            if len(class_samples) < shots_per_class + query_per_class:
                continue

            # Randomly select support and query samples
            indices = np.random.choice(
                len(class_samples),
                size=shots_per_class + query_per_class,
                replace=False,
            )

            # Support set (for training)
            support_indices = indices[:shots_per_class]
            support_X.append(class_samples[support_indices])
            support_y.extend([class_idx] * shots_per_class)

            # Query set (for evaluation)
            query_indices = indices[shots_per_class:]
            query_X.append(class_samples[query_indices])
            query_y.extend([class_idx] * query_per_class)

        if len(support_X) > 0:
            # Combine support samples
            support_X = np.vstack(support_X)
            support_y = np.array(support_y)

            # Combine query samples
            query_X = np.vstack(query_X)
            query_y = np.array(query_y)

            # Convert to tensors
            support_tensor = (torch.FloatTensor(support_X), torch.LongTensor(support_y))
            query_tensor = (torch.FloatTensor(query_X), torch.LongTensor(query_y))

            tasks.append((support_tensor, query_tensor))

    return tasks


def evaluate_model(
    model: nn.Module,
    tasks: List[
        Tuple[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]
    ],
    inner_steps: int = 5,
    inner_lr: float = 0.1,
    device: str = "cpu",
) -> Dict[str, float]:
    """Evaluate model on few-shot tasks"""
    model.eval()
    accuracies = []

    for (support_X, support_y), (query_X, query_y) in tasks:
        # Create a copy for adaptation
        task_model = type(model)(
            model.layers[0].in_features,
            model.layers[2].out_features,
            model.layers[-1].out_features,
        ).to(device)

        # Copy initial parameters
        for name, param in task_model.named_parameters():
            if name in dict(model.named_parameters()):
                param.data = dict(model.named_parameters())[name].data.clone()

        # Adapt to support set
        optimizer = optim.SGD(task_model.parameters(), lr=inner_lr)
        criterion = nn.CrossEntropyLoss()

        for _ in range(inner_steps):
            optimizer.zero_grad()
            outputs = task_model(support_X)
            loss = criterion(outputs, support_y)
            loss.backward()
            optimizer.step()

        # Evaluate on query set
        with torch.no_grad():
            outputs = task_model(query_X)
            _, predicted = torch.max(outputs, 1)
            accuracy = (predicted == query_y).float().mean().item()
            accuracies.append(accuracy)

    return {
        "mean_accuracy": np.mean(accuracies),
        "std_accuracy": np.std(accuracies),
        "min_accuracy": np.min(accuracies),
        "max_accuracy": np.max(accuracies),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Reptile Meta-Learning para Few-Shot Learning"
    )
    parser.add_argument(
        "--data",
        type=str,
        default="data/synthetic.npz",
        help="Caminho para os dados (.npz)",
    )
    parser.add_argument(
        "--meta_epochs", type=int, default=100, help="Número de épocas meta-learning"
    )
    parser.add_argument(
        "--tasks_per_epoch", type=int, default=10, help="Número de tarefas por época"
    )
    parser.add_argument(
        "--shots_per_class",
        type=int,
        default=5,
        help="Número de shots por classe (5-shot)",
    )
    parser.add_argument(
        "--query_per_class",
        type=int,
        default=15,
        help="Número de amostras de query por classe",
    )
    parser.add_argument(
        "--meta_lr", type=float, default=0.01, help="Learning rate meta-learning"
    )
    parser.add_argument(
        "--inner_lr", type=float, default=0.1, help="Learning rate inner loop"
    )
    parser.add_argument(
        "--inner_steps", type=int, default=5, help="Número de passos inner loop"
    )
    parser.add_argument(
        "--hidden_dim", type=int, default=64, help="Dimensão da camada oculta"
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Seed para reprodutibilidade"
    )
    parser.add_argument(
        "--eval_every", type=int, default=10, help="Avaliar a cada N épocas"
    )
    parser.add_argument("--verbose", action="store_true", help="Modo verboso")

    args = parser.parse_args()

    # Set seeds
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Create artifacts directory
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    artifacts_dir = Path(f"artifacts/{timestamp}")
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    print(f"🚀 Reptile Meta-Learning iniciado")
    print(f"📁 Artifacts: {artifacts_dir}")

    # Load data
    if not os.path.exists(args.data):
        print(f"❌ Dados não encontrados: {args.data}")
        print("💡 Execute primeiro: python examples/synthetic_data.py")
        return

    data = np.load(args.data)
    X, y = data["X"], data["y"]
    # Convert one-hot to integer labels
    y = np.argmax(y, axis=1)
    num_classes = len(np.unique(y))
    input_dim = X.shape[1]

    print(f"📊 Dados carregados: {X.shape}, {num_classes} classes")

    # Create model
    model = SimpleMLP(input_dim, args.hidden_dim, num_classes)
    trainer = ReptileTrainer(
        model=model,
        meta_lr=args.meta_lr,
        inner_lr=args.inner_lr,
        inner_steps=args.inner_steps,
    )

    print(f"🧠 Modelo criado: {sum(p.numel() for p in model.parameters())} parâmetros")

    # Training loop
    meta_history = []
    eval_history = []

    print(f"\n🔄 Iniciando treino meta-learning...")
    start_time = time.time()

    for epoch in range(args.meta_epochs):
        # Generate tasks for this epoch
        tasks = generate_few_shot_tasks(
            X,
            y,
            args.tasks_per_epoch,
            args.shots_per_class,
            args.query_per_class,
            num_classes,
            args.seed + epoch,
        )

        # Meta-learning step
        meta_loss, stats = trainer.meta_step([task[0] for task in tasks])
        meta_history.append({"epoch": epoch, "meta_loss": meta_loss, **stats})

        # Evaluation
        if (epoch + 1) % args.eval_every == 0:
            eval_stats = evaluate_model(model, tasks, args.inner_steps, args.inner_lr)
            eval_history.append({"epoch": epoch, **eval_stats})

            if args.verbose:
                print(
                    f"Epoch {epoch+1:3d}/{args.meta_epochs} | "
                    f"Meta Loss: {meta_loss:.4f} | "
                    f"Accuracy: {eval_stats['mean_accuracy']:.3f} ± {eval_stats['std_accuracy']:.3f}"
                )

        # Progress indicator
        if (epoch + 1) % 10 == 0:
            print(
                f"✅ Epoch {epoch+1:3d}/{args.meta_epochs} | Meta Loss: {meta_loss:.4f}"
            )

    training_time = time.time() - start_time
    print(f"\n🎯 Treino concluído em {training_time:.1f}s")

    # Final evaluation
    final_tasks = generate_few_shot_tasks(
        X,
        y,
        50,
        args.shots_per_class,
        args.query_per_class,
        num_classes,
        args.seed + 999,
    )
    final_eval = evaluate_model(model, final_tasks, args.inner_steps, args.inner_lr)

    print(f"\n📈 Resultados Finais:")
    print(
        f"   Accuracy: {final_eval['mean_accuracy']:.3f} ± {final_eval['std_accuracy']:.3f}"
    )
    print(
        f"   Range: [{final_eval['min_accuracy']:.3f}, {final_eval['max_accuracy']:.3f}]"
    )

    # Save artifacts
    print(f"\n💾 Salvando artifacts...")

    # Save model
    torch.save(model.state_dict(), artifacts_dir / "reptile_model.pth")

    # Save training history
    with open(artifacts_dir / "meta_history.json", "w") as f:
        json.dump(meta_history, f, indent=2, default=str)

    # Save evaluation history
    with open(artifacts_dir / "eval_history.json", "w") as f:
        json.dump(eval_history, f, indent=2, default=str)

    # Save final evaluation
    with open(artifacts_dir / "final_eval.json", "w") as f:
        json.dump(final_eval, f, indent=2, default=str)

    # Save configuration
    config = {
        "meta_epochs": args.meta_epochs,
        "tasks_per_epoch": args.tasks_per_epoch,
        "shots_per_class": args.shots_per_class,
        "query_per_class": args.query_per_class,
        "meta_lr": args.meta_lr,
        "inner_lr": args.inner_lr,
        "inner_steps": args.inner_steps,
        "hidden_dim": args.hidden_dim,
        "seed": args.seed,
        "training_time": training_time,
        "final_accuracy": final_eval["mean_accuracy"],
    }

    with open(artifacts_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    print(f"✅ Artifacts salvos em: {artifacts_dir}")
    print(f"🎉 Reptile Meta-Learning concluído com sucesso!")


if __name__ == "__main__":
    main()

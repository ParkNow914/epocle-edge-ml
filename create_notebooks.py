#!/usr/bin/env python3
"""
Script para criar notebooks funcionais para o projeto Epocle Edge ML
"""

import json
import os

def create_notebook_01():
    """Criar notebook de treinamento NumPy"""
    notebook = {
        "cells": [
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "# Notebook 01: Treinamento com NumPy MLP\n\n",
                    "Este notebook demonstra o uso do MLP implementado em NumPy com otimizadores e EWC."
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "import numpy as np\n",
                    "import matplotlib.pyplot as plt\n",
                    "from src import SimpleMLP, SGD, EWC\n\n",
                    "# Configurar seed para reprodutibilidade\n",
                    "np.random.seed(42)"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# Criar dados sintéticos\n",
                    "X = np.random.randn(1000, 20)\n",
                    "y = np.eye(3)[np.random.randint(0, 3, 1000)]\n\n",
                    "print(f\"Dados criados: X shape {X.shape}, y shape {y.shape}\")"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# Criar modelo\n",
                    "model = SimpleMLP(input_dim=20, hidden=64, num_classes=3)\n",
                    "optimizer = SGD(model.get_params(), lr=0.01)\n\n",
                    "print(\"Modelo criado com sucesso!\")"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# Treinamento básico\n",
                    "losses = []\n",
                    "for epoch in range(50):\n",
                    "    loss, gradients = model.loss_and_grad(X, y)\n",
                    "    optimizer.step(gradients)\n",
                    "    losses.append(loss)\n",
                    "    \n",
                    "    if epoch % 10 == 0:\n",
                    "        print(f\"Epoch {epoch}: Loss = {loss:.4f}\")\n\n",
                    "print(f\"Treinamento concluído! Loss final: {losses[-1]:.4f}\")"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# Plotar curva de loss\n",
                    "plt.figure(figsize=(10, 6))\n",
                    "plt.plot(losses)\n",
                    "plt.title('Curva de Loss durante Treinamento')\n",
                    "plt.xlabel('Epoch')\n",
                    "plt.ylabel('Loss')\n",
                    "plt.grid(True)\n",
                    "plt.show()"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# Avaliar modelo\n",
                    "predictions = model.predict(X)\n",
                    "accuracy = np.mean(np.argmax(predictions, axis=1) == np.argmax(y, axis=1))\n",
                    "print(f\"Acurácia no conjunto de treinamento: {accuracy:.4f}\")"
                ]
            }
        ],
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "codemirror_mode": {"name": "ipython", "version": 3},
                "file_extension": ".py",
                "mimetype": "text/x-python",
                "name": "python",
                "nbconvert_exporter": "python",
                "pygments_lexer": "ipython3",
                "version": "3.9.0"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 4
    }
    
    with open('notebooks/01_numpy_training.ipynb', 'w', encoding='utf-8') as f:
        json.dump(notebook, f, indent=1, ensure_ascii=False)
    
    print("✅ Notebook 01 criado com sucesso!")

def create_notebook_02():
    """Criar notebook de meta-learning com Reptile"""
    notebook = {
        "cells": [
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "# Notebook 02: Meta-Learning com Reptile\n\n",
                    "Este notebook demonstra o uso do algoritmo Reptile para few-shot learning."
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "import torch\n",
                    "import torch.nn as nn\n",
                    "import numpy as np\n",
                    "import matplotlib.pyplot as plt\n",
                    "from examples.reptile_pytorch import ReptileTrainer, generate_few_shot_tasks\n\n",
                    "# Configurar seed para reprodutibilidade\n",
                    "torch.manual_seed(42)\n",
                    "np.random.seed(42)"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# Gerar tasks de few-shot\n",
                    "num_tasks = 50\n",
                    "shots = 5\n",
                    "ways = 3\n",
                    "input_dim = 20\n\n",
                    "tasks = generate_few_shot_tasks(num_tasks, shots, ways, input_dim)\n",
                    "print(f\"Geradas {len(tasks)} tasks de {ways}-way {shots}-shot\")"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# Criar trainer Reptile\n",
                    "trainer = ReptileTrainer(\n",
                    "    model_type=\"mlp\",\n",
                    "    input_dim=input_dim,\n",
                    "    hidden_dim=64,\n",
                    "    num_classes=ways,\n",
                    "    inner_lr=0.01,\n",
                    "    outer_lr=0.001\n",
                    ")\n\n",
                    "print(\"Trainer Reptile criado com sucesso!\")"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# Treinamento meta-learning\n",
                    "meta_losses = []\n",
                    "for epoch in range(100):\n",
                    "    meta_loss = trainer.train_epoch(tasks[:10])  # Usar apenas 10 tasks por epoch\n",
                    "    meta_losses.append(meta_loss)\n",
                    "    \n",
                    "    if epoch % 20 == 0:\n",
                    "        print(f\"Meta-epoch {epoch}: Loss = {meta_loss:.4f}\")\n\n",
                    "print(f\"Meta-treinamento concluído! Loss final: {meta_losses[-1]:.4f}\")"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# Plotar curva de meta-loss\n",
                    "plt.figure(figsize=(10, 6))\n",
                    "plt.plot(meta_losses)\n",
                    "plt.title('Curva de Meta-Loss durante Treinamento')\n",
                    "plt.xlabel('Meta-Epoch')\n",
                    "plt.ylabel('Meta-Loss')\n",
                    "plt.grid(True)\n",
                    "plt.show()"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# Avaliar em novas tasks\n",
                    "test_tasks = generate_few_shot_tasks(10, shots, ways, input_dim)\n",
                    "accuracies = []\n\n",
                    "for task in test_tasks:\n",
                    "    X_support, y_support, X_query, y_query = task\n",
                    "    accuracy = trainer.evaluate_task(X_support, y_support, X_query, y_query)\n",
                    "    accuracies.append(accuracy)\n\n",
                    "mean_accuracy = np.mean(accuracies)\n",
                    "print(f\"Acurácia média em novas tasks: {mean_accuracy:.4f}\")\n",
                    "print(f\"Desvio padrão: {np.std(accuracies):.4f}\")"
                ]
            }
        ],
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "codemirror_mode": {"name": "ipython", "version": 3},
                "file_extension": ".py",
                "mimetype": "text/x-python",
                "name": "python",
                "nbconvert_exporter": "python",
                "pygments_lexer": "ipython3",
                "version": "3.9.0"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 4
    }
    
    with open('notebooks/02_reptile_fewshot.ipynb', 'w', encoding='utf-8') as f:
        json.dump(notebook, f, indent=1, ensure_ascii=False)
    
    print("✅ Notebook 02 criado com sucesso!")

def main():
    """Função principal"""
    print("🚀 Criando notebooks funcionais para Epocle Edge ML...")
    
    # Criar diretório notebooks se não existir
    os.makedirs('notebooks', exist_ok=True)
    
    # Criar notebooks
    create_notebook_01()
    create_notebook_02()
    
    print("🎉 Todos os notebooks foram criados com sucesso!")
    print("📁 Localização: ./notebooks/")

if __name__ == "__main__":
    main()

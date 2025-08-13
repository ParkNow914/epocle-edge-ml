#!/usr/bin/env python3
"""
Secure Privacy Demo Pipeline

This script demonstrates the integration of Secure Aggregation and Differential Privacy
in a federated learning scenario with multiple clients.

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
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from src.dp_utils import (NoiseMechanism, PrivacyPreservingTraining,
                          calculate_sensitivity_for_gradients,
                          create_privacy_budget)
from src.secure_agg import (FederatedLearningSecurity,
                            generate_test_model_update,
                            verify_secure_aggregation)


class SecurePrivacyDemo:
    """Demo class for Secure Aggregation and Differential Privacy"""

    def __init__(self, num_clients: int, privacy_epsilon: float, privacy_delta: float):
        self.num_clients = num_clients
        self.privacy_epsilon = privacy_epsilon
        self.privacy_delta = privacy_delta

        # Initialize security components
        self.security = FederatedLearningSecurity()
        self.security.setup_secure_aggregation(num_clients)

        # Initialize privacy components
        self.privacy_budget = create_privacy_budget(
            epsilon=privacy_epsilon,
            delta=privacy_delta,
            mechanism=NoiseMechanism.LAPLACE,
        )
        self.privacy_training = PrivacyPreservingTraining(self.privacy_budget)

        # Client sessions
        self.client_sessions = {}
        self.client_updates = {}

    def setup_clients(self) -> Dict[str, str]:
        """Setup client sessions for federated learning"""
        print(f"🔐 Configurando {self.num_clients} clientes seguros...")

        for i in range(self.num_clients):
            client_id = f"client_{i:02d}"
            session_token = self.security.register_federated_client(client_id)
            self.client_sessions[client_id] = session_token

            print(
                f"   ✅ Cliente {client_id} registrado com token: {session_token[:8]}..."
            )

        return self.client_sessions

    def generate_client_updates(
        self, input_dim: int = 20, hidden_dim: int = 64
    ) -> Dict[str, Dict[str, np.ndarray]]:
        """Generate synthetic model updates for each client"""
        print(f"📊 Gerando atualizações de modelo para {self.num_clients} clientes...")

        for client_id in self.client_sessions.keys():
            # Generate different update for each client
            np.random.seed(hash(client_id) % 1000)
            update = generate_test_model_update(input_dim, hidden_dim)
            self.client_updates[client_id] = update

            print(
                f"   ✅ {client_id}: update gerado com {sum(v.size for v in update.values())} parâmetros"
            )

        return self.client_updates

    def apply_differential_privacy(
        self, clip_norm: float = 1.0
    ) -> Dict[str, Dict[str, np.ndarray]]:
        """Apply differential privacy to client updates"""
        print(
            f"🔒 Aplicando Differential Privacy (ε={self.privacy_epsilon}, δ={self.privacy_delta})..."
        )

        private_updates = {}

        for client_id, update in self.client_updates.items():
            # Clip gradients to bound sensitivity
            clipped_update = self.privacy_training.clip_gradients(update, clip_norm)

            # Calculate sensitivity
            sensitivity = calculate_sensitivity_for_gradients(clipped_update, clip_norm)

            # Add noise for differential privacy
            private_update = self.privacy_training.add_noise_to_model_update(
                clipped_update, sensitivity
            )

            private_updates[client_id] = private_update

            print(
                f"   ✅ {client_id}: privacidade aplicada (sensitivity={sensitivity:.3f})"
            )

        return private_updates

    def submit_secure_updates(
        self, private_updates: Dict[str, Dict[str, np.ndarray]]
    ) -> bool:
        """Submit private updates to secure aggregation"""
        print(f"🔐 Submetendo atualizações para agregação segura...")

        success_count = 0

        for client_id, private_update in private_updates.items():
            session_token = self.client_sessions[client_id]

            success = self.security.submit_federated_update(
                client_id, session_token, private_update
            )

            if success:
                success_count += 1
                print(f"   ✅ {client_id}: update submetido com sucesso")
            else:
                print(f"   ❌ {client_id}: falha na submissão")

        print(f"📊 {success_count}/{self.num_clients} updates submetidos com sucesso")
        return success_count == self.num_clients

    def perform_secure_aggregation(self) -> Optional[Dict[str, np.ndarray]]:
        """Perform secure aggregation of all updates"""
        print(f"🔄 Executando agregação segura...")

        # Check status
        status = self.security.get_federated_status()
        print(
            f"   📊 Status: {status['submitted_updates']}/{status['threshold']} updates prontos"
        )

        if not status["ready_for_aggregation"]:
            print(f"   ⚠️ Agregação não possível: threshold não atingido")
            return None

        # Perform aggregation
        start_time = time.time()
        aggregated_update = self.security.aggregate_federated_updates()
        aggregation_time = time.time() - start_time

        if aggregated_update:
            print(f"   ✅ Agregação concluída em {aggregation_time:.3f}s")
            print(
                f"   📊 Resultado: {sum(v.size for v in aggregated_update.values())} parâmetros"
            )
        else:
            print(f"   ❌ Falha na agregação")

        return aggregated_update

    def verify_results(
        self, aggregated_update: Dict[str, np.ndarray]
    ) -> Dict[str, Any]:
        """Verify the results of secure aggregation"""
        print(f"🔍 Verificando resultados da agregação segura...")

        # Convert updates to list format for verification
        original_updates = list(self.client_updates.values())

        # Verify secure aggregation
        is_correct = verify_secure_aggregation(original_updates, aggregated_update)

        # Calculate privacy metrics
        privacy_status = self.privacy_training.get_privacy_status()

        # Calculate utility metrics
        utility_metrics = self._calculate_utility_metrics(
            original_updates, aggregated_update
        )

        results = {
            "aggregation_correct": is_correct,
            "privacy_status": privacy_status,
            "utility_metrics": utility_metrics,
            "num_clients": self.num_clients,
            "privacy_epsilon": self.privacy_epsilon,
            "privacy_delta": self.privacy_delta,
        }

        print(f"   ✅ Verificação concluída:")
        print(f"      Agregação correta: {is_correct}")
        print(f"      Queries de privacidade: {privacy_status['query_count']}")
        print(f"      Epsilon restante: {privacy_status['remaining_epsilon']:.3f}")

        return results

    def _calculate_utility_metrics(
        self,
        original_updates: List[Dict[str, np.ndarray]],
        aggregated_update: Dict[str, np.ndarray],
    ) -> Dict[str, float]:
        """Calculate utility metrics for the aggregated update"""
        metrics = {}

        # Calculate expected aggregation (simple average)
        expected_aggregated = {}
        for key in original_updates[0].keys():
            expected_aggregated[key] = np.mean(
                [update[key] for update in original_updates], axis=0
            )

        # Calculate MSE between expected and actual
        total_mse = 0.0
        total_params = 0

        for key in expected_aggregated.keys():
            expected = expected_aggregated[key]
            actual = aggregated_update[key]

            mse = np.mean((expected - actual) ** 2)
            total_mse += mse * expected.size
            total_params += expected.size

        metrics["average_mse"] = total_mse / total_params if total_params > 0 else 0.0

        # Calculate correlation
        try:
            expected_flat = np.concatenate(
                [v.flatten() for v in expected_aggregated.values()]
            )
            actual_flat = np.concatenate(
                [v.flatten() for v in aggregated_update.values()]
            )

            correlation = np.corrcoef(expected_flat, actual_flat)[0, 1]
            metrics["correlation"] = correlation if not np.isnan(correlation) else 0.0
        except:
            metrics["correlation"] = 0.0

        return metrics

    def run_demo(
        self, input_dim: int = 20, hidden_dim: int = 64, clip_norm: float = 1.0
    ) -> Dict[str, Any]:
        """Run the complete secure privacy demo"""
        print(f"🚀 Iniciando Demo de Segurança e Privacidade")
        print(f"📊 Clientes: {self.num_clients}")
        print(f"🔒 Privacidade: ε={self.privacy_epsilon}, δ={self.privacy_delta}")
        print(f"📏 Dimensões: {input_dim} → {hidden_dim} → 3")
        print(f"✂️ Clip norm: {clip_norm}")
        print("-" * 60)

        start_time = time.time()

        # Setup clients
        self.setup_clients()

        # Generate updates
        self.generate_client_updates(input_dim, hidden_dim)

        # Apply differential privacy
        private_updates = self.apply_differential_privacy(clip_norm)

        # Submit updates
        submission_success = self.submit_secure_updates(private_updates)

        if not submission_success:
            print("❌ Falha na submissão dos updates")
            return {}

        # Perform aggregation
        aggregated_update = self.perform_secure_aggregation()

        if aggregated_update is None:
            print("❌ Falha na agregação")
            return {}

        # Verify results
        results = self.verify_results(aggregated_update)

        total_time = time.time() - start_time

        print("-" * 60)
        print(f"🎉 Demo concluído em {total_time:.2f}s")

        return results


def main():
    parser = argparse.ArgumentParser(
        description="Demo de Segurança e Privacidade para Federated Learning"
    )
    parser.add_argument(
        "--clients", type=int, default=5, help="Número de clientes federados"
    )
    parser.add_argument(
        "--epsilon", type=float, default=1.0, help="Parâmetro de privacidade epsilon"
    )
    parser.add_argument(
        "--delta", type=float, default=1e-5, help="Parâmetro de privacidade delta"
    )
    parser.add_argument("--input_dim", type=int, default=20, help="Dimensão da entrada")
    parser.add_argument(
        "--hidden_dim", type=int, default=64, help="Dimensão da camada oculta"
    )
    parser.add_argument(
        "--clip_norm", type=float, default=1.0, help="Norma para clipping de gradientes"
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Seed para reprodutibilidade"
    )

    args = parser.parse_args()

    # Set seed
    np.random.seed(args.seed)

    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    output_dir = Path(f"artifacts/{timestamp}")
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"📁 Saída: {output_dir}")

    # Run demo
    demo = SecurePrivacyDemo(args.clients, args.epsilon, args.delta)
    results = demo.run_demo(args.input_dim, args.hidden_dim, args.clip_norm)

    if results:
        # Save results
        print(f"\n💾 Salvando resultados...")

        with open(output_dir / "secure_privacy_results.json", "w") as f:
            json.dump(results, f, indent=2, default=str)

        print(f"✅ Resultados salvos em: {output_dir / 'secure_privacy_results.json'}")

        # Print summary
        print(f"\n📊 Resumo dos Resultados:")
        print(f"   Clientes: {results['num_clients']}")
        print(f"   Agregação correta: {results['aggregation_correct']}")
        print(f"   MSE médio: {results['utility_metrics']['average_mse']:.6f}")
        print(f"   Correlação: {results['utility_metrics']['correlation']:.3f}")
        print(
            f"   Epsilon usado: {results['privacy_status']['total_epsilon_used']:.3f}"
        )
        print(
            f"   Epsilon restante: {results['privacy_status']['remaining_epsilon']:.3f}"
        )
    else:
        print("❌ Demo falhou - nenhum resultado para salvar")


if __name__ == "__main__":
    main()

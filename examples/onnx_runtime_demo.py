#!/usr/bin/env python3
"""
ONNX Runtime Demo Script

This script demonstrates ONNX Runtime inference capabilities, including
batch processing, performance benchmarking, and comparison with PyTorch.

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
from typing import Dict, List, Tuple, Optional, Union

import numpy as np
import torch
import torch.nn as nn
import onnxruntime as ort

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))


class SimpleMLP(nn.Module):
    """Simple MLP for comparison with ONNX"""
    
    def __init__(self, input_dim: int, hidden_dim: int, num_classes: int):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class ONNXRuntimeDemo:
    """ONNX Runtime demonstration class"""
    
    def __init__(self, onnx_path: str, device: str = "cpu"):
        self.onnx_path = onnx_path
        self.device = device
        self.session = None
        self.input_name = None
        self.output_name = None
        self.input_shape = None
        self.output_shape = None
        
        self._load_model()
    
    def _load_model(self):
        """Load ONNX model and create session"""
        if not os.path.exists(self.onnx_path):
            raise FileNotFoundError(f"ONNX model not found: {self.onnx_path}")
        
        # Create ONNX Runtime session
        providers = ['CPUExecutionProvider']
        if self.device == "gpu" and 'CUDAExecutionProvider' in ort.get_available_providers():
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        
        self.session = ort.InferenceSession(self.onnx_path, providers=providers)
        
        # Get input/output info
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name
        self.input_shape = self.session.get_inputs()[0].shape
        self.output_shape = self.session.get_outputs()[0].shape
        
        print(f"✅ ONNX model loaded: {self.onnx_path}")
        print(f"📊 Input: {self.input_name} {self.input_shape}")
        print(f"📊 Output: {self.output_name} {self.output_shape}")
        print(f"🔧 Providers: {self.session.get_providers()}")
    
    def predict(self, input_data: np.ndarray) -> np.ndarray:
        """Run inference on input data"""
        if self.session is None:
            raise RuntimeError("ONNX session not initialized")
        
        # Prepare input
        ort_inputs = {self.input_name: input_data.astype(np.float32)}
        
        # Run inference
        outputs = self.session.run(None, ort_inputs)
        
        return outputs[0]
    
    def predict_batch(self, input_batch: np.ndarray, batch_size: int = 32) -> np.ndarray:
        """Run inference on batched input data"""
        if self.session is None:
            raise RuntimeError("ONNX session not initialized")
        
        results = []
        
        for i in range(0, len(input_batch), batch_size):
            batch = input_batch[i:i + batch_size]
            batch_result = self.predict(batch)
            results.append(batch_result)
        
        return np.vstack(results)
    
    def benchmark_inference(self, input_data: np.ndarray, num_runs: int = 100) -> Dict[str, float]:
        """Benchmark inference performance"""
        if self.session is None:
            raise RuntimeError("ONNX session not initialized")
        
        # Warm up
        for _ in range(10):
            _ = self.predict(input_data)
        
        # Benchmark
        start_time = time.time()
        for _ in range(num_runs):
            _ = self.predict(input_data)
        total_time = time.time() - start_time
        
        avg_time = total_time / num_runs
        throughput = len(input_data) / avg_time
        
        return {
            'total_time': total_time,
            'avg_time_per_inference': avg_time,
            'avg_time_per_sample': avg_time / len(input_data),
            'throughput_samples_per_second': throughput,
            'num_runs': num_runs
        }


def create_synthetic_data(
    num_samples: int, 
    input_dim: int, 
    num_classes: int,
    seed: int = 42
) -> Tuple[np.ndarray, np.ndarray]:
    """Create synthetic data for testing"""
    np.random.seed(seed)
    
    # Generate random features
    X = np.random.randn(num_samples, input_dim).astype(np.float32)
    
    # Generate random labels
    y = np.random.randint(0, num_classes, num_samples)
    
    return X, y


def compare_pytorch_onnx(
    pytorch_model: nn.Module,
    onnx_demo: ONNXRuntimeDemo,
    input_data: np.ndarray,
    tolerance: float = 1e-5
) -> Dict[str, any]:
    """Compare PyTorch and ONNX outputs"""
    
    # PyTorch inference
    pytorch_model.eval()
    with torch.no_grad():
        pytorch_input = torch.FloatTensor(input_data)
        pytorch_output = pytorch_model(pytorch_input).numpy()
    
    # ONNX inference
    onnx_output = onnx_demo.predict(input_data)
    
    # Compare outputs
    diff = np.abs(pytorch_output - onnx_output)
    max_diff = np.max(diff)
    mean_diff = np.mean(diff)
    std_diff = np.std(diff)
    
    is_correct = max_diff < tolerance
    
    return {
        'is_correct': is_correct,
        'max_difference': max_diff,
        'mean_difference': mean_diff,
        'std_difference': std_diff,
        'tolerance': tolerance,
        'pytorch_output_shape': pytorch_output.shape,
        'onnx_output_shape': onnx_output.shape,
        'pytorch_output_range': [pytorch_output.min(), pytorch_output.max()],
        'onnx_output_range': [onnx_output.min(), onnx_output.max()]
    }


def benchmark_comparison(
    pytorch_model: nn.Module,
    onnx_demo: ONNXRuntimeDemo,
    input_data: np.ndarray,
    num_runs: int = 100
) -> Dict[str, any]:
    """Benchmark PyTorch vs ONNX Runtime performance"""
    
    # PyTorch benchmark
    pytorch_model.eval()
    pytorch_input = torch.FloatTensor(input_data)
    
    # Warm up
    with torch.no_grad():
        for _ in range(10):
            _ = pytorch_model(pytorch_input)
    
    # Benchmark
    start_time = time.time()
    with torch.no_grad():
        for _ in range(num_runs):
            _ = pytorch_model(pytorch_input)
    pytorch_time = time.time() - start_time
    
    # ONNX benchmark
    onnx_results = onnx_demo.benchmark_inference(input_data, num_runs)
    
    # Calculate speedup
    speedup = pytorch_time / onnx_results['total_time']
    
    return {
        'pytorch': {
            'total_time': pytorch_time,
            'avg_time_per_inference': pytorch_time / num_runs,
            'avg_time_per_sample': pytorch_time / (num_runs * len(input_data))
        },
        'onnx': onnx_results,
        'speedup': speedup,
        'num_runs': num_runs
    }


def run_interactive_demo(onnx_demo: ONNXRuntimeDemo, input_dim: int, num_classes: int):
    """Run interactive demo with user input"""
    print(f"\n🎮 Demo Interativo ONNX Runtime")
    print(f"📊 Input dimension: {input_dim}")
    print(f"🏷️ Number of classes: {num_classes}")
    
    while True:
        try:
            # Get user input
            user_input = input(f"\n🔢 Digite {input_dim} números separados por espaço (ou 'quit' para sair): ")
            
            if user_input.lower() == 'quit':
                break
            
            # Parse input
            values = [float(x) for x in user_input.split()]
            if len(values) != input_dim:
                print(f"❌ Erro: esperava {input_dim} valores, recebeu {len(values)}")
                continue
            
            # Create input tensor
            input_data = np.array(values).reshape(1, input_dim).astype(np.float32)
            
            # Run inference
            start_time = time.time()
            output = onnx_demo.predict(input_data)
            inference_time = (time.time() - start_time) * 1000
            
            # Process output
            predicted_class = np.argmax(output[0])
            confidence = np.max(output[0])
            
            print(f"🎯 Resultado:")
            print(f"   Classe prevista: {predicted_class}")
            print(f"   Confiança: {confidence:.4f}")
            print(f"   Tempo de inferência: {inference_time:.2f} ms")
            print(f"   Output completo: {output[0]}")
            
        except ValueError:
            print("❌ Erro: valores inválidos. Use números separados por espaço.")
        except KeyboardInterrupt:
            break
    
    print(f"\n👋 Demo interativo finalizado!")


def main():
    parser = argparse.ArgumentParser(description="Demo ONNX Runtime para inferência")
    parser.add_argument("--onnx_model", type=str, required=True,
                       help="Caminho para modelo ONNX")
    parser.add_argument("--input_dim", type=int, default=20,
                       help="Dimensão da entrada")
    parser.add_argument("--hidden_dim", type=int, default=64,
                       help="Dimensão da camada oculta")
    parser.add_argument("--num_classes", type=int, default=3,
                       help="Número de classes")
    parser.add_argument("--num_samples", type=int, default=1000,
                       help="Número de amostras para benchmark")
    parser.add_argument("--batch_size", type=int, default=32,
                       help="Tamanho do batch")
    parser.add_argument("--num_runs", type=int, default=100,
                       help="Número de execuções para benchmark")
    parser.add_argument("--compare", action="store_true",
                       help="Comparar com modelo PyTorch")
    parser.add_argument("--interactive", action="store_true",
                       help="Executar demo interativo")
    parser.add_argument("--seed", type=int, default=42,
                       help="Seed para reprodutibilidade")
    
    args = parser.parse_args()
    
    # Set seeds
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    output_dir = Path(f"artifacts/{timestamp}")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"🚀 ONNX Runtime Demo iniciado")
    print(f"📁 Saída: {output_dir}")
    print(f"🔧 Modelo: {args.onnx_model}")
    
    # Load ONNX model
    try:
        onnx_demo = ONNXRuntimeDemo(args.onnx_model)
    except Exception as e:
        print(f"❌ Erro ao carregar modelo ONNX: {e}")
        return
    
    # Generate synthetic data
    print(f"\n📊 Gerando dados sintéticos...")
    X, y = create_synthetic_data(args.num_samples, args.input_dim, args.num_classes, args.seed)
    print(f"✅ Dados gerados: {X.shape}, {len(np.unique(y))} classes")
    
    # Basic inference test
    print(f"\n🧪 Teste básico de inferência...")
    sample_input = X[:5]  # First 5 samples
    sample_output = onnx_demo.predict(sample_input)
    print(f"✅ Inferência bem-sucedida: {sample_output.shape}")
    
    # Batch processing test
    print(f"\n📦 Teste de processamento em batch...")
    batch_output = onnx_demo.predict_batch(X, args.batch_size)
    print(f"✅ Batch processing: {batch_output.shape}")
    
    # Performance benchmark
    print(f"\n⚡ Benchmark de performance...")
    benchmark_results = onnx_demo.benchmark_inference(X, args.num_runs)
    
    print(f"📊 Resultados do Benchmark:")
    print(f"   Tempo total: {benchmark_results['total_time']:.3f}s")
    print(f"   Tempo médio por inferência: {benchmark_results['avg_time_per_inference']*1000:.3f} ms")
    print(f"   Throughput: {benchmark_results['throughput_samples_per_second']:.0f} amostras/s")
    
    # Comparison with PyTorch if requested
    comparison_results = None
    if args.compare:
        print(f"\n🔍 Comparando com PyTorch...")
        
        # Create PyTorch model
        pytorch_model = SimpleMLP(args.input_dim, args.hidden_dim, args.num_classes)
        
        # Compare outputs
        comparison_results = compare_pytorch_onnx(pytorch_model, onnx_demo, X[:100])
        
        if comparison_results['is_correct']:
            print(f"✅ Comparação bem-sucedida!")
            print(f"📊 Diferença máxima: {comparison_results['max_difference']:.2e}")
        else:
            print(f"⚠️ Diferenças detectadas na comparação")
            print(f"📊 Diferença máxima: {comparison_results['max_difference']:.2e}")
        
        # Performance comparison
        perf_comparison = benchmark_comparison(pytorch_model, onnx_demo, X[:100], args.num_runs)
        
        print(f"\n📊 Comparação de Performance:")
        print(f"   PyTorch: {perf_comparison['pytorch']['avg_time_per_inference']*1000:.3f} ms")
        print(f"   ONNX:    {perf_comparison['onnx']['avg_time_per_inference']*1000:.3f} ms")
        print(f"   Speedup: {perf_comparison['speedup']:.2f}x")
    
    # Interactive demo if requested
    if args.interactive:
        run_interactive_demo(onnx_demo, args.input_dim, args.num_classes)
    
    # Save results
    print(f"\n💾 Salvando resultados...")
    
    results = {
        'timestamp': timestamp,
        'onnx_model_path': args.onnx_model,
        'input_dim': args.input_dim,
        'hidden_dim': args.hidden_dim,
        'num_classes': args.num_classes,
        'num_samples': args.num_samples,
        'batch_size': args.batch_size,
        'num_runs': args.num_runs,
        'benchmark_results': benchmark_results,
        'model_info': {
            'input_shape': onnx_demo.input_shape,
            'output_shape': onnx_demo.output_shape,
            'providers': onnx_demo.session.get_providers() if onnx_demo.session else []
        }
    }
    
    if comparison_results:
        results['comparison_results'] = comparison_results
    
    with open(output_dir / "runtime_demo_results.json", 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"✅ Resultados salvos em: {output_dir / 'runtime_demo_results.json'}")
    print(f"🎉 ONNX Runtime Demo concluído com sucesso!")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
ONNX Export Script for PyTorch Models

This script exports PyTorch models to ONNX format for deployment and inference.
Supports both the Reptile meta-learning model and simple MLP architectures.

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
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import onnx
import onnxruntime as ort
import torch
import torch.nn as nn
import torch.onnx

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))


class SimpleMLP(nn.Module):
    """Simple MLP for ONNX export"""

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


class ReptileMLP(nn.Module):
    """Reptile-trained MLP for ONNX export"""

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


def create_dummy_input(
    input_shape: Tuple[int, ...], dtype: torch.dtype = torch.float32
) -> torch.Tensor:
    """Create dummy input tensor for ONNX export"""
    return torch.randn(input_shape, dtype=dtype)


def export_model_to_onnx(
    model: nn.Module,
    dummy_input: torch.Tensor,
    output_path: str,
    input_names: List[str] = None,
    output_names: List[str] = None,
    dynamic_axes: Dict[str, Dict[int, str]] = None,
    opset_version: int = 11,
    verbose: bool = False,
) -> Dict[str, any]:
    """Export PyTorch model to ONNX format"""

    if input_names is None:
        input_names = ["input"]

    if output_names is None:
        output_names = ["output"]

    if dynamic_axes is None:
        dynamic_axes = {"input": {0: "batch_size"}, "output": {0: "batch_size"}}

    # Set model to eval mode
    model.eval()

    # Export to ONNX
    start_time = time.time()

    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=opset_version,
        do_constant_folding=True,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        verbose=verbose,
    )

    export_time = time.time() - start_time

    # Validate ONNX model
    onnx_model = onnx.load(output_path)
    onnx.checker.check_model(onnx_model)

    # Get model info
    model_info = {
        "input_shape": list(dummy_input.shape),
        "output_shape": list(model(dummy_input).shape),
        "input_names": input_names,
        "output_names": output_names,
        "opset_version": opset_version,
        "export_time": export_time,
        "model_size_mb": os.path.getsize(output_path) / (1024 * 1024),
        "onnx_ir_version": onnx_model.ir_version,
        "producer_name": onnx_model.producer_name,
        "producer_version": onnx_model.producer_version,
    }

    return model_info


def create_simple_mlp_model(
    input_dim: int, hidden_dim: int, num_classes: int
) -> SimpleMLP:
    """Create and initialize a simple MLP model"""
    model = SimpleMLP(input_dim, hidden_dim, num_classes)

    # Initialize weights with Xavier/Glorot initialization
    for module in model.modules():
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            nn.init.zeros_(module.bias)

    return model


def load_reptile_model(
    model_path: str, input_dim: int, hidden_dim: int, num_classes: int
) -> ReptileMLP:
    """Load a pre-trained Reptile model"""
    model = ReptileMLP(input_dim, hidden_dim, num_classes)

    if os.path.exists(model_path):
        checkpoint = torch.load(model_path, map_location="cpu")
        model.load_state_dict(checkpoint)
        print(f"✅ Modelo Reptile carregado de: {model_path}")
    else:
        print(f"⚠️ Modelo não encontrado: {model_path}")
        print("💡 Criando modelo com inicialização aleatória")

    return model


def benchmark_models(
    pytorch_model: nn.Module,
    onnx_path: str,
    input_tensor: torch.Tensor,
    num_runs: int = 100,
) -> Dict[str, float]:
    """Benchmark PyTorch vs ONNX Runtime performance"""

    # Warm up
    for _ in range(10):
        _ = pytorch_model(input_tensor)

    # PyTorch benchmark
    start_time = time.time()
    for _ in range(num_runs):
        _ = pytorch_model(input_tensor)
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    pytorch_time = (time.time() - start_time) / num_runs

    # ONNX Runtime benchmark
    ort_session = ort.InferenceSession(onnx_path)
    ort_inputs = {ort_session.get_inputs()[0].name: input_tensor.numpy()}

    # Warm up
    for _ in range(10):
        _ = ort_session.run(None, ort_inputs)

    start_time = time.time()
    for _ in range(num_runs):
        _ = ort_session.run(None, ort_inputs)
    onnx_time = (time.time() - start_time) / num_runs

    return {
        "pytorch_time_ms": pytorch_time * 1000,
        "onnx_time_ms": onnx_time * 1000,
        "speedup": pytorch_time / onnx_time,
        "num_runs": num_runs,
    }


def verify_onnx_export(
    pytorch_model: nn.Module,
    onnx_path: str,
    input_tensor: torch.Tensor,
    tolerance: float = 1e-5,
) -> Dict[str, any]:
    """Verify ONNX export correctness"""

    # PyTorch output
    pytorch_model.eval()
    with torch.no_grad():
        pytorch_output = pytorch_model(input_tensor).numpy()

    # ONNX Runtime output
    ort_session = ort.InferenceSession(onnx_path)
    ort_inputs = {ort_session.get_inputs()[0].name: input_tensor.numpy()}
    onnx_output = ort_session.run(None, ort_inputs)[0]

    # Compare outputs
    diff = np.abs(pytorch_output - onnx_output)
    max_diff = np.max(diff)
    mean_diff = np.mean(diff)

    is_correct = max_diff < tolerance

    return {
        "is_correct": is_correct,
        "max_difference": max_diff,
        "mean_difference": mean_diff,
        "tolerance": tolerance,
        "pytorch_output_shape": pytorch_output.shape,
        "onnx_output_shape": onnx_output.shape,
    }


def main():
    parser = argparse.ArgumentParser(description="Exporta modelos PyTorch para ONNX")
    parser.add_argument(
        "--model_type",
        type=str,
        default="simple",
        choices=["simple", "reptile"],
        help="Tipo de modelo para exportar",
    )
    parser.add_argument(
        "--reptile_path",
        type=str,
        default="artifacts/latest/reptile_model.pth",
        help="Caminho para modelo Reptile treinado",
    )
    parser.add_argument("--input_dim", type=int, default=20, help="Dimensão da entrada")
    parser.add_argument(
        "--hidden_dim", type=int, default=64, help="Dimensão da camada oculta"
    )
    parser.add_argument("--num_classes", type=int, default=3, help="Número de classes")
    parser.add_argument(
        "--batch_size", type=int, default=32, help="Tamanho do batch para dummy input"
    )
    parser.add_argument("--opset", type=int, default=11, help="Versão do ONNX opset")
    parser.add_argument(
        "--out_dir",
        type=str,
        default="models/onnx",
        help="Diretório de saída para modelos ONNX",
    )
    parser.add_argument(
        "--benchmark", action="store_true", help="Executar benchmark de performance"
    )
    parser.add_argument(
        "--verify", action="store_true", help="Verificar correção da exportação"
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Seed para reprodutibilidade"
    )

    args = parser.parse_args()

    # Set seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    output_dir = Path(args.out_dir) / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"🚀 ONNX Export iniciado")
    print(f"📁 Saída: {output_dir}")
    print(f"🔧 Modelo: {args.model_type}")

    # Create model
    if args.model_type == "simple":
        model = create_simple_mlp_model(
            args.input_dim, args.hidden_dim, args.num_classes
        )
        model_name = "simple_mlp"
        print(f"✅ Modelo MLP simples criado")
    else:
        model = load_reptile_model(
            args.reptile_path, args.input_dim, args.hidden_dim, args.num_classes
        )
        model_name = "reptile_mlp"
        print(f"✅ Modelo Reptile carregado")

    print(f"🧠 Parâmetros: {sum(p.numel() for p in model.parameters()):,}")

    # Create dummy input
    dummy_input = create_dummy_input((args.batch_size, args.input_dim))
    print(f"📊 Input shape: {dummy_input.shape}")

    # Export to ONNX
    onnx_path = output_dir / f"{model_name}.onnx"
    print(f"\n🔄 Exportando para ONNX...")

    model_info = export_model_to_onnx(
        model=model,
        dummy_input=dummy_input,
        output_path=str(onnx_path),
        opset_version=args.opset,
        verbose=False,
    )

    print(f"✅ Exportação concluída em {model_info['export_time']:.2f}s")
    print(f"📁 Modelo salvo: {onnx_path}")
    print(f"📊 Tamanho: {model_info['model_size_mb']:.2f} MB")

    # Verify export if requested
    if args.verify:
        print(f"\n🔍 Verificando exportação...")
        verification = verify_onnx_export(model, str(onnx_path), dummy_input)

        if verification["is_correct"]:
            print(f"✅ Exportação verificada com sucesso!")
            print(f"📊 Diferença máxima: {verification['max_difference']:.2e}")
        else:
            print(f"❌ Falha na verificação!")
            print(f"📊 Diferença máxima: {verification['max_difference']:.2e}")

    # Benchmark if requested
    if args.benchmark:
        print(f"\n⚡ Executando benchmark...")
        benchmark_results = benchmark_models(model, str(onnx_path), dummy_input)

        print(f"📊 Resultados do Benchmark:")
        print(f"   PyTorch: {benchmark_results['pytorch_time_ms']:.3f} ms")
        print(f"   ONNX:    {benchmark_results['onnx_time_ms']:.3f} ms")
        print(f"   Speedup: {benchmark_results['speedup']:.2f}x")

    # Save export info
    export_summary = {
        "model_type": args.model_type,
        "model_name": model_name,
        "export_timestamp": timestamp,
        "model_info": model_info,
        "input_shape": list(dummy_input.shape),
        "args": vars(args),
    }

    if args.verify:
        export_summary["verification"] = verification

    if args.benchmark:
        export_summary["benchmark"] = benchmark_results

    with open(output_dir / "export_info.json", "w") as f:
        json.dump(export_summary, f, indent=2, default=str)

    print(f"\n💾 Informações salvas em: {output_dir / 'export_info.json'}")
    print(f"🎉 Exportação ONNX concluída com sucesso!")


if __name__ == "__main__":
    main()

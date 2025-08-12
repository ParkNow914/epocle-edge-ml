# Epocle Edge ML

Continual Learning pipeline for edge devices with NumPy implementation.

## Instalação

```bash
python -m venv .venv
source .venv/bin/activate   # Linux/Mac
.venv\Scripts\activate      # Windows
pip install -r requirements.txt
```

## Uso Rápido

### Gerar dados sintéticos
```bash
python examples/synthetic_data.py --out data/synthetic.npz --classes 3 --per_class 200 --dim 20 --seed 42
```

### Treino online
```bash
python examples/train_online.py --data data/synthetic.npz --epochs 3 --lr 1e-2 \
  --replay_capacity 500 --replay_batch 32 --p_store 0.5 --use_ewc True \
  --dp_sigma 0.0 --seed 42
```

### Meta-learning Reptile
```bash
python examples/reptile_pytorch.py --meta_iters 100 --k_shot 5 --inner_steps 5 --seed 42
```

### Export ONNX
```bash
python examples/export_onnx.py --out artifacts/model.onnx
```

### Testes
```bash
pytest -q
```

## Estrutura

- `src/` - Implementações NumPy (MLP, otimizadores, EWC, etc.)
- `examples/` - Scripts executáveis
- `notebooks/` - Jupyter notebooks
- `tests/` - Testes unitários
- `scripts/` - Scripts de automação

## Licença

MIT

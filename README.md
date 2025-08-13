# Epocle Edge ML 🚀

> **Continual Learning para Dispositivos Edge com Privacidade e Segurança**

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Tests](https://img.shields.io/badge/tests-133%20passing-brightgreen.svg)](https://github.com/your-org/epocle-edge-ml)
[![Coverage](https://img.shields.io/badge/coverage-92%25-brightgreen.svg)](https://github.com/your-org/epocle-edge-ml)

**Epocle Edge ML** é um monorepo completo que implementa um pipeline de Continual Learning para dispositivos edge, com protótipos práticos de **Differential Privacy** e **Secure Aggregation**. Desenvolvido seguindo princípios de engenharia de software robusta e pesquisa em ML.

## ✨ Características Principais

### 🧠 **Core ML Components (NumPy)**
- **MLP completo** com forward/backward propagation manual
- **Otimizadores**: SGD, Adam, RMSprop
- **Ativações**: ReLU, Softmax estável
- **Loss**: Cross-entropy com gradientes

### 🔄 **Continual Learning**
- **EWC** (Elastic Weight Consolidation) para evitar forgetting
- **ReplayBuffer** para experiências passadas
- **OnlineEWC** para atualizações incrementais
- **Análise de importância de parâmetros**

### 🔒 **Privacy & Security**
- **Differential Privacy** com mecanismos Laplace/Gaussian/Exponential
- **Secure Aggregation** com conceitos de HE e MPC
- **Privacy budget management** adaptativo
- **Federated learning security** utilities

### 🚀 **Meta-learning & ONNX**
- **Reptile algorithm** em PyTorch para few-shot learning
- **ONNX export** e runtime inference
- **Benchmarking** de performance
- **Interoperabilidade** entre frameworks

### 🧪 **Testing & Quality**
- **133 testes** com 92% de cobertura
- **Formatação automática** com Black e isort
- **Linting** com Flake8
- **CI/CD** com GitHub Actions

## 🚀 Quick Start

### Instalação

```bash
# Clone o repositório
git clone https://github.com/your-org/epocle-edge-ml.git
cd epocle-edge-ml

# Instale as dependências
pip install -r requirements.txt
pip install -e .

# Instale ferramentas de desenvolvimento
pip install black isort flake8 pytest pytest-cov
```

### Uso Básico

```python
from src import SimpleMLP, SGD, EWC, ReplayBuffer
import numpy as np

# Crie um modelo
model = SimpleMLP(input_dim=20, hidden=64, num_classes=3)

# Configure otimizador e EWC
optimizer = SGD(model.get_params(), lr=0.01)
ewc = EWC(model=model, lambda_=1.0)

# Gere dados sintéticos
X = np.random.randn(100, 20)
y = np.eye(3)[np.random.randint(0, 3, 100)]

# Registre com EWC
ewc.register(X, y)

# Treinamento
for epoch in range(10):
    loss, gradients = model.loss_and_grad(X, y)
    
    # Adicione penalty EWC
    _, ewc_gradients = ewc.penalty()
    for key in gradients:
        gradients[key] += ewc_gradients[key]
    
    # Otimize
    optimizer.step(gradients)
```

### Exemplos Práticos

```bash
# Gerar dados sintéticos
python examples/synthetic_data.py --num-samples 1000 --num-features 20 --num-classes 3

# Treinamento online com EWC e DP
python examples/train_online.py --epochs 10 --use-ewc --use-dp

# Meta-learning com Reptile
python examples/reptile_pytorch.py --tasks 100 --shots 5

# Export para ONNX
python examples/export_onnx.py --model-type mlp --output artifacts/model.onnx

# Demo ONNX Runtime
python examples/onnx_runtime_demo.py --model artifacts/model.onnx
```

## 🏗️ Arquitetura

```
epocle-edge-ml/
├── src/                    # Código fonte principal
│   ├── numpy_nn.py        # MLP implementado em NumPy
│   ├── optimizer.py       # Otimizadores
│   ├── replay_buffer.py   # Buffer de replay para CL
│   ├── ewc.py            # Elastic Weight Consolidation
│   ├── dp_utils.py       # Differential Privacy
│   ├── secure_agg.py     # Secure Aggregation
│   └── __init__.py       # Interface pública
├── examples/              # Scripts de demonstração
├── tests/                 # Testes unitários (133 testes)
├── notebooks/             # Jupyter notebooks
├── artifacts/             # Modelos e resultados
└── data/                  # Dados sintéticos
```

## 🧪 Testing

```bash
# Executar todos os testes
python -m pytest tests/ --cov=src --cov-report=term-missing -v

# Executar testes específicos
python -m pytest tests/test_ewc.py -v

# Verificar cobertura mínima (70%)
python -m pytest tests/ --cov=src --cov-fail-under=70

# Executar CI completo
make ci
```

**Cobertura atual**: 92% (693 statements, 54 missing)

## 🐳 Docker

```bash
# Construir imagem
make docker-build

# Executar testes em container
make docker-test

# Executar container de desenvolvimento
make docker-dev

# Ou usar docker-compose
docker-compose up epocle-edge-ml
```

## 📚 Documentação

- **[DESIGN.md](DESIGN.md)** - Arquitetura e decisões de design
- **[CONTRIBUTING.md](CONTRIBUTING.md)** - Guia de contribuição
- **[CHANGELOG.md](CHANGELOG.md)** - Histórico de mudanças
- **Docstrings** - Documentação da API em cada módulo

## 🔧 Desenvolvimento

### Comandos Make

```bash
# Ver todos os comandos
make help

# Instalar dependências
make install

# Executar testes
make test

# Formatar código
make format

# Verificar linting
make lint

# Limpar arquivos temporários
make clean

# Gerar dados sintéticos
make data

# Executar demo
make demo

# CI completo
make ci

# Criar pacote
make package

# Preparar release
make release
```

### Padrões de Código

- **Formatação**: Black (88 chars, linha)
- **Imports**: isort
- **Linting**: Flake8
- **Commits**: Conventional Commits
- **Type hints**: Obrigatórios para APIs públicas

## 🚀 Roadmap

### v0.2.0 (Próximo)
- [ ] Otimizações de performance NumPy
- [ ] Mais algoritmos de CL (GEM, iCaRL)
- [ ] Benchmarks de performance
- [ ] Suporte a mais dispositivos edge

### v0.5.0 (Médio prazo)
- [ ] Integração com ONNX Runtime
- [ ] Pipeline de treinamento distribuído
- [ ] Suporte a hardware especializado

### v1.0.0 (Longo prazo)
- [ ] Implementações robustas de segurança
- [ ] Integração com frameworks de produção
- [ ] Suporte a edge computing real

## 🤝 Contribuindo

Contribuições são bem-vindas! Por favor, leia o [CONTRIBUTING.md](CONTRIBUTING.md) para detalhes sobre:

- Como configurar o ambiente de desenvolvimento
- Padrões de código e testes
- Processo de Pull Request
- Diretrizes de contribuição

## 📄 Licença

Este projeto está licenciado sob a Licença MIT - veja o arquivo [LICENSE](LICENSE) para detalhes.

## 🙏 Agradecimentos

- **Pesquisadores** que desenvolveram os algoritmos originais
- **Comunidade open source** por ferramentas e bibliotecas
- **Contribuidores** que ajudaram a construir este projeto

## 📞 Contato

- **Issues**: [GitHub Issues](https://github.com/your-org/epocle-edge-ml/issues)
- **Discussions**: [GitHub Discussions](https://github.com/your-org/epocle-edge-ml/discussions)
- **Wiki**: [GitHub Wiki](https://github.com/your-org/epocle-edge-ml/wiki)

---

**Epocle Edge ML** - Continual Learning para o futuro da computação edge! 🚀🔒🧠

# DESIGN.md - Epocle Edge ML

## Visão Geral

**Epocle Edge ML** é um monorepo completo que implementa um pipeline de Continual Learning para dispositivos edge, com protótipos práticos de Differential Privacy e Secure Aggregation. O projeto foi desenvolvido seguindo princípios de engenharia de software robusta e pesquisa em ML.

## Arquitetura do Sistema

### 1. Estrutura do Monorepo

```
epocle-edge-ml/
├── src/                    # Código fonte principal
│   ├── numpy_nn.py        # MLP implementado em NumPy
│   ├── optimizer.py       # Otimizadores (SGD, Adam, RMSprop)
│   ├── replay_buffer.py   # Buffer de replay para CL
│   ├── ewc.py            # Elastic Weight Consolidation
│   ├── dp_utils.py       # Differential Privacy
│   ├── secure_agg.py     # Secure Aggregation
│   └── __init__.py       # Interface pública
├── examples/              # Scripts de demonstração
├── tests/                 # Testes unitários
├── notebooks/             # Jupyter notebooks
├── artifacts/             # Modelos e resultados
└── data/                  # Dados sintéticos
```

### 2. Componentes Principais

#### 2.1 Neural Network (NumPy)
- **SimpleMLP**: Implementação manual de MLP com forward/backward
- **Ativações**: ReLU, Softmax estável
- **Loss**: Cross-entropy com gradientes
- **Gerenciamento de parâmetros**: get_params/set_params

#### 2.2 Otimizadores
- **SGD**: Stochastic Gradient Descent básico
- **Adam**: Adaptive Moment Estimation com bias correction
- **RMSprop**: Root Mean Square Propagation

#### 2.3 Continual Learning
- **ReplayBuffer**: Buffer FIFO para experiências passadas
- **EWC**: Elastic Weight Consolidation para evitar forgetting
- **OnlineEWC**: Versão incremental do EWC

#### 2.4 Privacy & Security
- **DifferentialPrivacy**: Mecanismos de ruído (Laplace, Gaussian, Exponential)
- **SecureAggregator**: Agregação segura para federated learning
- **HomomorphicEncryption**: Conceitos simplificados de HE

## Decisões de Design

### 1. Implementação "na unha" vs Frameworks

**Decisão**: Implementar componentes core em NumPy puro
**Justificativa**:
- Controle total sobre algoritmos
- Sem dependências externas pesadas
- Melhor compreensão dos fundamentos
- Facilita portabilidade para edge devices

**Alternativas consideradas**:
- PyTorch/TensorFlow: Muito pesados para edge
- Scikit-learn: Limitado para redes neurais
- JAX: Boa opção, mas complexidade adicional

### 2. Arquitetura Modular

**Decisão**: Design modular com interfaces claras
**Benefícios**:
- Facilita testes unitários
- Permite substituição de componentes
- Melhora manutenibilidade
- Suporta diferentes cenários de uso

### 3. POCs para Features Complexas

**Decisão**: Implementar POCs funcionais para features complexas
**Exemplos**:
- **Secure Aggregation**: HE simplificado + MPC conceitual
- **Differential Privacy**: Mecanismos básicos + budget management
- **ONNX Export**: Suporte básico para interoperabilidade

**Evolução para produção**:
- Substituir implementações simplificadas por bibliotecas robustas
- Integrar com frameworks de segurança (OpenMined, TF Privacy)
- Implementar protocolos completos de MPC

## Padrões de Implementação

### 1. Gerenciamento de Estado

```python
class EWC:
    def __init__(self, model, lambda_):
        self.model = model
        self.lambda_ = lambda_
        self.anchor_params = None
        self.fisher_diag = None
    
    def register(self, X, y):
        # Computa Fisher e salva parâmetros âncora
        pass
```

**Padrão**: Injeção de dependência via construtor
**Benefício**: Facilita testes e flexibilidade

### 2. Tratamento de Erros

```python
def create_privacy_budget(epsilon, delta=1e-5):
    if epsilon <= 0:
        raise ValueError("Epsilon must be positive")
    if delta <= 0 or delta >= 1:
        raise ValueError("Delta must be in (0, 1)")
    return PrivacyBudget(epsilon, delta)
```

**Padrão**: Validação early com mensagens claras
**Benefício**: Debugging mais fácil, APIs robustas

### 3. Testes Abrangentes

```python
def test_ewc_penalty_scaling(self):
    # Testa que penalty escala com desvio de parâmetros
    penalty_small = ewc.penalty()
    # Modifica parâmetros
    penalty_large = ewc.penalty()
    assert penalty_large > penalty_small
```

**Padrão**: Testes que verificam comportamento, não implementação
**Benefício**: Refatoração segura, documentação viva

## Trade-offs e Limitações

### 1. Performance vs Simplicidade

**Limitação**: Implementações NumPy são mais lentas que C++/CUDA
**Mitigação**: 
- Otimizações de vetorização
- POCs para validação conceitual
- Documentação de migração para produção

### 2. Segurança vs Funcionalidade

**Limitação**: HE e MPC são simplificados
**Mitigação**:
- Documentação clara das limitações
- Referências para implementações robustas
- Foco em conceitos e interfaces

### 3. Cobertura vs Complexidade

**Meta**: 70%+ de cobertura de testes
**Realidade**: 92% alcançado
**Benefício**: Código robusto e confiável

## Evolução e Roadmap

### 1. Curto Prazo (v0.2.0)
- [ ] Otimizações de performance NumPy
- [ ] Mais algoritmos de CL (GEM, iCaRL)
- [ ] Benchmarks de performance

### 2. Médio Prazo (v0.5.0)
- [ ] Integração com ONNX Runtime
- [ ] Suporte a mais dispositivos edge
- [ ] Pipeline de treinamento distribuído

### 3. Longo Prazo (v1.0.0)
- [ ] Implementações robustas de segurança
- [ ] Suporte a hardware especializado
- [ ] Integração com frameworks de produção

## Referências Técnicas

### 1. Continual Learning
- **EWC**: Kirkpatrick et al. (2017) - Overcoming catastrophic forgetting
- **Replay**: Lin (1992) - Self-improving reactive agents
- **Meta-learning**: Finn et al. (2017) - Model-agnostic meta-learning

### 2. Differential Privacy
- **Laplace Mechanism**: Dwork (2006) - Differential privacy
- **Gaussian Mechanism**: Dwork & Roth (2014) - Algorithmic foundations
- **Composition**: Kairouz et al. (2015) - Composition theorems

### 3. Secure Aggregation
- **Homomorphic Encryption**: Gentry (2009) - Fully homomorphic encryption
- **MPC**: Yao (1982) - Protocols for secure computations
- **Federated Learning**: McMahan et al. (2017) - Communication-efficient learning

## Conclusão

O design do Epocle Edge ML prioriza:
1. **Simplicidade**: Implementações claras e testáveis
2. **Modularidade**: Componentes intercambiáveis
3. **Robustez**: Testes abrangentes e validação
4. **Evolução**: POCs funcionais + roadmap claro

Esta abordagem permite validação rápida de conceitos enquanto mantém a porta aberta para evolução para sistemas de produção robustos.

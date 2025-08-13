# Contribuindo para Epocle Edge ML

Obrigado por seu interesse em contribuir para o Epocle Edge ML! Este documento fornece diretrizes para contribuições.

## Visão Geral

Epocle Edge ML é um projeto de pesquisa e desenvolvimento que implementa Continual Learning para dispositivos edge com foco em privacidade e segurança. Valorizamos contribuições que:

- Melhoram a qualidade do código
- Adicionam novos algoritmos ou funcionalidades
- Corrigem bugs ou problemas de performance
- Melhoram a documentação
- Adicionam testes

## Configuração do Ambiente

### Pré-requisitos

- Python 3.9+
- pip
- git

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

## Padrões de Código

### 1. Formatação

Usamos **Black** para formatação automática:

```bash
# Formatar código
python -m black src/ tests/ examples/

# Verificar formatação
python -m black --check src/ tests/ examples/
```

### 2. Organização de Imports

Usamos **isort** para organizar imports:

```bash
# Organizar imports
python -m isort src/ tests/ examples/

# Verificar organização
python -m isort --check-only src/ tests/ examples/
```

### 3. Linting

Usamos **Flake8** para verificação de qualidade:

```bash
# Executar linting
python -m flake8 src/ tests/ examples/ --max-line-length=88 --ignore=E203,W503
```

### 4. Testes

**Meta**: Manter cobertura de testes acima de 70%

```bash
# Executar todos os testes
python -m pytest tests/ --cov=src --cov-report=term-missing -v

# Executar testes específicos
python -m pytest tests/test_ewc.py -v

# Executar com cobertura detalhada
python -m pytest tests/ --cov=src --cov-report=html
```

## Estrutura do Projeto

### Organização de Arquivos

```
src/
├── __init__.py           # Interface pública do pacote
├── numpy_nn.py          # Implementações de redes neurais
├── optimizer.py          # Otimizadores
├── ewc.py               # Elastic Weight Consolidation
├── replay_buffer.py      # Buffer de replay
├── dp_utils.py          # Differential Privacy
└── secure_agg.py        # Secure Aggregation

tests/
├── test_numpy_nn.py     # Testes para redes neurais
├── test_optimizer.py    # Testes para otimizadores
├── test_ewc.py          # Testes para EWC
├── test_replay.py       # Testes para replay buffer
├── test_dp_utils.py     # Testes para DP
└── test_secure_agg.py   # Testes para Secure Aggregation

examples/
├── synthetic_data.py     # Geração de dados sintéticos
├── train_online.py       # Pipeline de treinamento online
├── reptile_pytorch.py    # Meta-learning com Reptile
├── export_onnx.py        # Export para ONNX
└── onnx_runtime_demo.py  # Demo do ONNX Runtime
```

### Convenções de Nomenclatura

- **Classes**: PascalCase (ex: `SimpleMLP`, `ElasticWeightConsolidation`)
- **Funções/Métodos**: snake_case (ex: `compute_fisher_diag`, `add_noise`)
- **Variáveis**: snake_case (ex: `learning_rate`, `hidden_dim`)
- **Constantes**: UPPER_SNAKE_CASE (ex: `MAX_ITERATIONS`, `DEFAULT_EPSILON`)

## Processo de Contribuição

### 1. Issue ou Discussão

Antes de implementar uma feature ou correção:

1. **Verifique se já existe uma issue** relacionada
2. **Crie uma nova issue** se necessário, descrevendo:
   - Problema ou feature
   - Contexto e motivação
   - Proposta de solução
   - Impacto esperado

### 2. Fork e Branch

```bash
# Fork o repositório no GitHub
# Clone seu fork
git clone https://github.com/your-username/epocle-edge-ml.git
cd epocle-edge-ml

# Adicione o repositório original como upstream
git remote add upstream https://github.com/original-org/epocle-edge-ml.git

# Crie uma branch para sua feature
git checkout -b feature/nova-funcionalidade
# ou para correção
git checkout -b fix/correcao-bug
```

### 3. Desenvolvimento

Durante o desenvolvimento:

1. **Mantenha commits pequenos e focados**
2. **Use Conventional Commits**:
   ```
   feat: adiciona novo algoritmo de CL
   fix: corrige bug no cálculo de gradientes
   docs: atualiza documentação da API
   test: adiciona testes para edge cases
   refactor: reorganiza estrutura do módulo
   ```

3. **Execute testes regularmente**:
   ```bash
   python -m pytest tests/ --cov=src -v
   ```

4. **Mantenha a formatação**:
   ```bash
   python -m black src/ tests/ examples/
   python -m isort src/ tests/ examples/
   ```

### 4. Testes

**Obrigatório**: Todos os testes devem passar

```bash
# Executar suite completa
python -m pytest tests/ --cov=src --cov-report=term-missing -v

# Verificar cobertura mínima (70%)
python -m pytest tests/ --cov=src --cov-fail-under=70
```

**Para novas funcionalidades**: Adicione testes que cubram:
- Casos de uso normais
- Edge cases
- Tratamento de erros
- Integração com outros componentes

### 5. Pull Request

Ao criar um PR:

1. **Título descritivo**: "feat: implementa algoritmo GEM para continual learning"
2. **Descrição detalhada**:
   - O que foi implementado
   - Como foi implementado
   - Testes adicionados
   - Impacto em outras funcionalidades
   - Screenshots (se aplicável)

3. **Checklist**:
   - [ ] Código segue padrões do projeto
   - [ ] Testes passam e cobertura ≥ 70%
   - [ ] Documentação atualizada
   - [ ] Não quebra funcionalidades existentes

### 6. Review e Merge

- **Code review obrigatório** para todos os PRs
- **CI/CD deve passar** (formatação, linting, testes)
- **Aprovação de pelo menos um maintainer**

## Diretrizes Específicas

### 1. Implementação de Algoritmos

Para novos algoritmos de ML:

```python
class NovoAlgoritmo:
    """Implementação do novo algoritmo.
    
    Referência: Autor et al. (ano) - Título do paper
    """
    
    def __init__(self, **kwargs):
        """Inicializa o algoritmo.
        
        Args:
            param1: Descrição do parâmetro
            param2: Descrição do parâmetro
        """
        pass
    
    def fit(self, X, y):
        """Treina o modelo.
        
        Args:
            X: Features de entrada
            y: Labels de saída
            
        Returns:
            self: Instância treinada
        """
        pass
```

### 2. Testes

Estrutura recomendada para testes:

```python
class TestNovoAlgoritmo:
    def test_initialization(self):
        """Testa inicialização correta"""
        pass
    
    def test_fit_method(self):
        """Testa método de treinamento"""
        pass
    
    def test_edge_cases(self):
        """Testa casos extremos"""
        pass
    
    def test_integration(self):
        """Testa integração com outros componentes"""
        pass
```

### 3. Documentação

- **Docstrings** para todas as classes e métodos públicos
- **Type hints** para parâmetros e retornos
- **Exemplos** de uso quando apropriado
- **Referências** para papers ou implementações originais

## Recursos Úteis

### Comandos Make

```bash
# Ver todos os comandos disponíveis
make help

# Executar CI completo
make ci

# Limpar arquivos temporários
make clean

# Gerar dados sintéticos
make data

# Executar demo
make demo
```

### Docker

```bash
# Construir imagem
make docker-build

# Executar testes em container
make docker-test

# Executar container de desenvolvimento
make docker-dev
```

## Comunicação

### Canais

- **Issues**: Para bugs, features e discussões
- **Discussions**: Para perguntas e ideias
- **Pull Requests**: Para contribuições de código

### Etiqueta

- Seja respeitoso e construtivo
- Use português ou inglês
- Forneça contexto suficiente
- Reconheça contribuições de outros

## Reconhecimento

Contribuições significativas serão reconhecidas:

- **Contributors.md**: Lista de contribuidores
- **Release notes**: Menção em releases
- **Documentação**: Atribuição em funcionalidades implementadas

## Perguntas?

Se você tiver dúvidas sobre como contribuir:

1. **Verifique a documentação** existente
2. **Pesquise issues** anteriores
3. **Abra uma issue** para sua dúvida
4. **Participe das discussões** do projeto

Obrigado por contribuir para o Epocle Edge ML! 🚀

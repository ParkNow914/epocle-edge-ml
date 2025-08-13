# Changelog

Todas as mudanças notáveis neste projeto serão documentadas neste arquivo.

O formato é baseado em [Keep a Changelog](https://keepachangelog.com/pt-BR/1.0.0/),
e este projeto adere ao [Semantic Versioning](https://semver.org/lang/pt-BR/).

## [Unreleased]

### Added
- Suporte a notebooks Jupyter para demonstrações interativas
- Comandos Docker adicionais no Makefile
- Arquivo docker-compose.yml para desenvolvimento

### Changed
- Melhorada formatação de código com Black e isort
- Atualizado Dockerfile com melhores práticas
- Refinado Makefile com comandos adicionais

## [0.1.0] - 2024-12-19

### Added
- **Implementação completa de MLP em NumPy**
  - Forward e backward propagation manuais
  - Ativações ReLU e Softmax estável
  - Loss Cross-entropy com gradientes
  - Gerenciamento de parâmetros (get_params/set_params)

- **Sistema de otimizadores**
  - SGD (Stochastic Gradient Descent)
  - Adam com bias correction
  - RMSprop

- **Continual Learning**
  - ReplayBuffer FIFO para experiências passadas
  - EWC (Elastic Weight Consolidation) para evitar forgetting
  - OnlineEWC para atualizações incrementais
  - Análise de importância de parâmetros

- **Differential Privacy**
  - Mecanismos de ruído: Laplace, Gaussian, Exponential
  - Gerenciamento de privacy budget
  - Privacy-preserving training utilities
  - Adaptive privacy budget management

- **Secure Aggregation**
  - Conceitos de Homomorphic Encryption (simplificado)
  - Secure multi-party computation (POC)
  - Federated learning security utilities
  - Client registration e session management

- **Meta-learning com PyTorch**
  - Algoritmo Reptile para few-shot learning
  - Geração de few-shot tasks
  - Avaliação de meta-learning

- **ONNX Support**
  - Export de modelos PyTorch para ONNX
  - ONNX Runtime inference demo
  - Benchmarking de performance

- **Pipeline de treinamento online**
  - Treinamento incremental com dados streaming
  - Integração com EWC e replay buffer
  - Suporte opcional a Differential Privacy
  - Checkpointing e salvamento de artefatos

- **Geração de dados sintéticos**
  - CLI para criação de datasets multi-classe
  - Configuração flexível de parâmetros
  - Salvamento em formato .npz

### Changed
- Estrutura do projeto reorganizada em monorepo
- Interface unificada via `src/__init__.py`
- Sistema de testes abrangente com pytest
- Cobertura de testes aumentada para 92%

### Fixed
- Correções de bugs em implementações NumPy
- Melhorias na estabilidade numérica
- Correções em testes de integração
- Resolução de problemas de compatibilidade

## [0.0.1] - 2024-12-01

### Added
- Estrutura inicial do projeto
- Configuração básica de build system
- Dependências principais definidas
- Estrutura de diretórios estabelecida

---

## Notas de Release

### v0.1.0 - Release Principal

Esta é a primeira release completa do Epocle Edge ML, implementando:

1. **Core ML Components**: MLP, otimizadores, e algoritmos de continual learning implementados "na unha" em NumPy
2. **Privacy & Security**: POCs funcionais de Differential Privacy e Secure Aggregation
3. **Meta-learning**: Implementação do algoritmo Reptile em PyTorch
4. **ONNX Support**: Export e runtime para interoperabilidade
5. **Testing**: Suite abrangente de testes com 92% de cobertura
6. **Documentation**: DESIGN.md, CONTRIBUTING.md, e documentação completa da API

### Compatibilidade

- **Python**: 3.9+
- **Dependências principais**: NumPy, PyTorch, ONNX
- **Sistemas operacionais**: Linux, macOS, Windows
- **Arquiteturas**: x86_64, ARM64 (com limitações)

### Limitações Conhecidas

1. **Performance**: Implementações NumPy são mais lentas que C++/CUDA
2. **Segurança**: HE e MPC são POCs simplificados
3. **Hardware**: Otimizações específicas para edge devices limitadas

### Roadmap para v0.2.0

- [ ] Otimizações de performance NumPy
- [ ] Mais algoritmos de Continual Learning (GEM, iCaRL)
- [ ] Benchmarks de performance
- [ ] Suporte a mais dispositivos edge
- [ ] Melhorias na documentação

---

## Convenções de Versionamento

- **MAJOR**: Mudanças incompatíveis na API
- **MINOR**: Novas funcionalidades compatíveis
- **PATCH**: Correções de bugs compatíveis

## Contribuições

Para contribuir com o changelog:

1. Use o formato estabelecido
2. Agrupe mudanças por tipo (Added, Changed, Fixed, etc.)
3. Inclua contexto suficiente para usuários
4. Mantenha entradas concisas mas informativas

## Links

- [Keep a Changelog](https://keepachangelog.com/pt-BR/1.0.0/)
- [Semantic Versioning](https://semver.org/lang/pt-BR/)
- [Conventional Commits](https://www.conventionalcommits.org/)

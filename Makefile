.PHONY: help install test lint format clean docker-build docker-run

help:
	@echo "Comandos disponíveis:"
	@echo "  install     - Instalar dependências"
	@echo "  test        - Executar testes"
	@echo "  lint        - Verificar formatação"
	@echo "  format      - Formatar código com black"
	@echo "  clean       - Limpar arquivos temporários"
	@echo "  docker-build - Construir imagem Docker"
	@echo "  docker-run   - Executar container Docker"

install:
	pip install -r requirements.txt

test:
	pytest -q

lint:
	black --check .

format:
	black .

clean:
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	rm -rf artifacts/*.npz
	rm -rf artifacts/*.onnx

docker-build:
	docker build -t epocle-edge-ml .

docker-run:
	docker run -it --rm epocle-edge-ml

data:
	python examples/synthetic_data.py --out data/synthetic.npz --classes 3 --per_class 200 --dim 20 --seed 42

demo:
	python examples/train_online.py --data data/synthetic.npz --epochs 3 --lr 1e-2 --seed 42

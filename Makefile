.PHONY: help install test lint format clean docker-build docker-run docker-test docker-dev data demo ci package release

help: ## Show this help message
	@echo "Available commands:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

install: ## Install dependencies
	pip install -e .
	pip install -r requirements.txt

test: ## Run tests with coverage
	python -m pytest tests/ --cov=src --cov-report=term-missing -v

lint: ## Run linting checks
	python -m flake8 src/ tests/ examples/ --max-line-length=88 --ignore=E203,W503

format: ## Format code with black and isort
	python -m black src/ tests/ examples/ notebooks/
	python -m isort src/ tests/ examples/ notebooks/

clean: ## Clean build artifacts
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf artifacts/*
	rm -rf data/*
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete

docker-build: ## Build Docker image
	docker build -t epocle-edge-ml .

docker-run: ## Run Docker container
	docker run -it --rm -v $(PWD)/artifacts:/app/artifacts -v $(PWD)/data:/app/data epocle-edge-ml

docker-test: ## Run tests in Docker
	docker run -it --rm -v $(PWD)/artifacts:/app/artifacts -v $(PWD)/data:/app/data epocle-edge-ml python -m pytest tests/ -v

docker-dev: ## Run development container
	docker-compose --profile dev up epocle-edge-ml-dev

data: ## Generate synthetic data
	python examples/synthetic_data.py --num-samples 1000 --num-features 20 --num-classes 3 --output data/synthetic_data.npz

demo: ## Run online training demo
	python examples/train_online.py --epochs 5 --batch-size 32 --use-dp --use-ewc

ci: ## Run CI checks (format, lint, test)
	$(MAKE) format
	$(MAKE) lint
	$(MAKE) test

package: ## Create distribution package
	python -m build

release: ## Prepare release (clean, test, package)
	$(MAKE) clean
	$(MAKE) test
	$(MAKE) package

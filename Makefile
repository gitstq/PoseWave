.PHONY: help install dev test lint format clean build run docker docs

help:  ## Show this help message
	@echo "🌊 PoseWave - WiFi CSI Human Pose Detection System"
	@echo ""
	@echo "Available commands:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2}'

install:  ## Install dependencies
	pip install -r requirements.txt

dev:  ## Install development dependencies
	pip install -r requirements.txt
	pip install -e ".[dev]"

test:  ## Run tests
	pytest tests/ -v --cov=posewave --cov-report=term-missing

test-watch:  ## Run tests with file watcher
	pytest-watch tests/ -v

lint:  ## Run linting
	flake8 posewave/ tests/ --max-line-length=100
	mypy posewave/

format:  ## Format code
	black posewave/ tests/
	isort posewave/ tests/

clean:  ## Clean build artifacts
	rm -rf build/ dist/ *.egg-info .pytest_cache .coverage .mypy_cache
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

build:  ## Build package
	python -m build

run:  ## Run API server
	uvicorn posewave.api.main:app --host 0.0.0.0 --port 8000 --reload

run-prod:  ## Run API server in production mode
	uvicorn posewave.api.main:app --host 0.0.0.0 --port 8000 --workers 4

docker-build:  ## Build Docker image
	docker build -t posewave:latest .

docker-run:  ## Run Docker container
	docker run -p 8000:8000 posewave:latest

docker-up:  ## Start all services with docker-compose
	docker-compose up -d

docker-down:  ## Stop all services
	docker-compose down

docker-logs:  ## View Docker logs
	docker-compose logs -f

docs:  ## Build documentation
	mkdocs build

docs-serve:  ## Serve documentation locally
	mkdocs serve

example:  ## Run basic usage example
	python examples/basic_usage.py

example-realtime:  ## Run real-time detection example
	python examples/realtime_detection.py

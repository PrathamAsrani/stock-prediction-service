.PHONY: help install train build run stop clean test

help:
	@echo "Stock ML Prediction Service - Available Commands:"
	@echo "  make install          - Install dependencies"
	@echo "  make build-kb         - Build knowledge base"
	@echo "  make train            - Train ML models"
	@echo "  make build            - Build Docker image"
	@echo "  make run              - Run service locally"
	@echo "  make docker-run       - Run service in Docker"
	@echo "  make stop             - Stop Docker containers"
	@echo "  make clean            - Clean temporary files"
	@echo "  make test             - Run tests"

install:
	pip install -r requirements.txt

build-kb:
	python scripts/build_knowledge_base.py

train:
	python scripts/train_models.py

build:
	docker-compose build

run:
	uvicorn app.main:app --host 0.0.0.0 --port 8001 --reload

docker-run:
	docker-compose up -d

stop:
	docker-compose down

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type f -name ".DS_Store" -delete

test:
	pytest tests/ -v
	
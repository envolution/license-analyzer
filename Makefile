.PHONY: help install install-dev test test-cov lint format clean build upload

help:
	@echo "Available commands:"
	@echo "  install     Install package"
	@echo "  install-dev Install package with development dependencies"
	@echo "  test        Run tests"
	@echo "  test-cov    Run tests with coverage"
	@echo "  lint        Run linting"
	@echo "  format      Format code"
	@echo "  clean       Clean build artifacts"
	@echo "  build       Build package"
	@echo "  upload      Upload to PyPI"

install:
	pip install -e .

install-dev:
	pip install -e .[dev]
	pip install -r requirements-dev.txt

test:
	pytest

test-cov:
	pytest --cov=license_analyzer --cov-report=term-missing --cov-report=html

lint:
	flake8 license_analyzer/ tests/
	mypy license_analyzer/

format:
	black license_analyzer/ tests/

clean:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf .coverage
	rm -rf htmlcov/
	rm -rf .pytest_cache/
	rm -rf .mypy_cache/
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

build: clean
	python -m build

upload: build
	python -m twine upload dist/*

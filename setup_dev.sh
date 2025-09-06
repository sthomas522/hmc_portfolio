#!/bin/bash
set -e

echo "Setting up Portfolio Analyzer development environment with uv..."

# Initialize uv project (if not already done)
if [ ! -f "pyproject.toml" ]; then
    uv init --python 3.11
fi

# Create and activate virtual environment
echo "Creating virtual environment..."
uv venv --python 3.11

# Activate virtual environment
source .venv/bin/activate

# Install dependencies
echo "Installing dependencies..."
uv sync

# Install development dependencies
echo "Installing development dependencies..."
uv add --dev pytest pytest-asyncio pytest-cov httpx ruff black mypy pre-commit ipython jupyter rich

# Install pre-commit hooks
echo "Setting up pre-commit hooks..."
pre-commit install

# Create .env file if it doesn't exist
if [ ! -f ".env" ]; then
    cp .env.example .env
    echo "Created .env file - please update with your settings"
fi

echo "Development environment setup complete!"
echo "Activate with: source .venv/bin/activate"
echo "Run tests with: uv run pytest"
echo "Start server with: uv run uvicorn portfolio_analyzer.main:app --reload"


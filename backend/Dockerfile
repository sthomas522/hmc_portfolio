FROM python:3.11-slim

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /usr/local/bin/

# Set environment variables
ENV UV_CACHE_DIR=/opt/uv-cache
ENV UV_LINK_MODE=copy

# Create app directory
WORKDIR /app

# Copy dependency files first (for layer caching)
COPY pyproject.toml uv.lock* ./

# Install dependencies with uv (much faster than pip)
RUN --mount=type=cache,target=/opt/uv-cache \\
    uv sync --frozen --no-dev

# Copy application code
COPY . .

# Create non-root user
RUN groupadd -r appuser && useradd -r -g appuser appuser
RUN chown -R appuser:appuser /app
USER appuser

# Expose port
EXPOSE 8000

# Run with uv
CMD ["uv", "run", "uvicorn", "portfolio_analyzer.main:app", "--host", "0.0.0.0", "--port", "8000"]
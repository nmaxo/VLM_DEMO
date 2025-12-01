FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Set environment variables
ENV HF_HOME=/root/.cache/huggingface
ENV PIP_CACHE_DIR=/root/.cache/pip

# Copy and install requirements
COPY requirements_b.txt .
RUN pip install -r requirements_b.txt

# Create necessary directories
RUN mkdir -p /app/uploads ${HF_HOME} /models

# Копируем ВЕСЬ backend (сохраняет структуру backend/app/main.py)
COPY backend /app/backend

EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Правильный модульный путь для backend/app/main.py
CMD ["uvicorn", "backend.app.main:app", "--host", "0.0.0.0", "--port", "8000"]
FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy and install requirements
COPY requirements_f.txt .
RUN pip install -r requirements_f.txt

# Copy Streamlit application
COPY frontend/app.py ./app.py

# Configure Streamlit
RUN mkdir -p ~/.streamlit && \
    echo '[client]\nshowErrorDetails = true' > ~/.streamlit/config.toml

EXPOSE 8501

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

# Start Streamlit
ENTRYPOINT ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
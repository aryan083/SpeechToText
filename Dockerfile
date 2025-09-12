# Hugging Face Spaces optimized Dockerfile
FROM python:3.11-slim

# Set environment variables for Spaces
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    ffmpeg \
    git \
    libsndfile1 \
    libsox-fmt-all \
    sox \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create necessary directories with proper permissions
RUN mkdir -p /tmp/models /tmp/logs /tmp/temp && \
    chmod 755 /tmp/models /tmp/logs /tmp/temp

# Set environment variables for Spaces
ENV APP_ENV=prod \
    GRADIO_SERVER_NAME=0.0.0.0 \
    GRADIO_SERVER_PORT=7860 \
    MODEL_CACHE_DIR=/tmp/models \
    HF_HOME=/tmp/models \
    TRANSFORMERS_CACHE=/tmp/models \
    TORCH_HOME=/tmp/models

# Expose port for Spaces
EXPOSE 7860

# Use the Spaces-optimized app
CMD ["python", "app.py"]

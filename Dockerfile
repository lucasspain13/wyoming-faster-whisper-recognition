# Use Python 3.11 slim as base
FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    ffmpeg \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first to leverage Docker cache
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY . .

# Create directory for speaker embeddings
RUN mkdir -p /data

# Set default command
CMD ["python", "-m", "wyoming_faster_whisper", \
    "--model", "nvidia/parakeet-tdt-0.6b-v2", \
    "--uri", "tcp://0.0.0.0:10300", \
    "--device", "auto", \
    "--data-dir", "./data", \
    "--download-dir", "./models", \
    "--embeddings-file", "./data/user_embeddings.pkl", "--debug"]
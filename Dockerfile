# ─────────────────────────────────────────────────────────
# Hybrid Recommender – Dockerfile
# ─────────────────────────────────────────────────────────
# Build:  docker build -t hybrid-recommender .
# Run:    docker run -p 8000:8000 hybrid-recommender
# ─────────────────────────────────────────────────────────

# Use an official slim Python image as the base
FROM python:3.11-slim

# Prevents Python from buffering stdout/stderr (good for Docker logs)
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# Set working directory inside the container
WORKDIR /app

# Install system dependencies required by faiss and implicit
# (libgomp = OpenMP for parallelism; libglib provides core utils)
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        libgomp1 \
        libglib2.0-0 && \
    rm -rf /var/lib/apt/lists/*

# Install Python dependencies first (layer-cached unless requirements change)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code and configuration
COPY src/       ./src/
COPY app.py     .
COPY config.yaml .

# Copy pre-built model artifacts
# The data/ directory contains parquet files, .npy embeddings,
# faiss_index.bin, and als_model.pkl produced by running:
#   python -m src.train
# These are mounted at runtime or baked in at build time.
COPY data/      ./data/

# Expose the API port
EXPOSE 8000

# Health check so Docker/Kubernetes knows when the service is ready
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Start the FastAPI server
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]

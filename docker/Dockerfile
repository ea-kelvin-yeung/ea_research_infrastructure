# Dockerfile for Backtest Server
# Build from project root: docker build -f docker/Dockerfile -t backtest-server .
# Run: docker run -d -p 8000:8000 -v $(pwd)/snapshots:/app/snapshots:ro backtest-server

FROM python:3.11-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (for better caching)
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Install additional server dependencies
RUN pip install --no-cache-dir \
    fastapi \
    uvicorn[standard] \
    pyyaml

# Copy application code
COPY poc/ ./poc/
COPY api/ ./api/

# Create snapshots directory (will be mounted as volume)
RUN mkdir -p /app/snapshots

# Config is included in api/ directory

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8000/health', timeout=5)" || exit 1

# Default command
CMD ["python", "-m", "api.server", "--host", "0.0.0.0", "--port", "8000"]

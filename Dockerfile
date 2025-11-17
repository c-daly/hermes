# Dockerfile for Hermes API server
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy project files
COPY pyproject.toml .
COPY src/ src/

# Install Python dependencies
RUN pip install --no-cache-dir -e .

# Expose the API port
EXPOSE 8080

# Set environment variables
ENV PYTHONUNBUFFERED=1

# Run the server
CMD ["uvicorn", "hermes.main:app", "--host", "0.0.0.0", "--port", "8080"]

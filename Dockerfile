# Production-ready Dockerfile for Hermes API server
# Optimized for FastAPI/uvicorn with stateless design
FROM python:3.11-slim

# Add metadata labels following OCI standards
LABEL org.opencontainers.image.title="Hermes API" \
      org.opencontainers.image.description="Stateless language & embedding tools for Project LOGOS" \
      org.opencontainers.image.vendor="Project LOGOS Team" \
      org.opencontainers.image.source="https://github.com/c-daly/hermes" \
      org.opencontainers.image.licenses="MIT"

# Install system dependencies for ML libraries and curl for healthcheck
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    ffmpeg \
    libsndfile1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user for security
RUN useradd -m -u 1000 -s /bin/bash hermes && \
    mkdir -p /app && \
    chown -R hermes:hermes /app

# Set working directory
WORKDIR /app

# Copy dependency + source files (needed for poetry build includes)
COPY --chown=hermes:hermes pyproject.toml README.md src/ ./

# Install Python dependencies as root for system-wide availability
# Install PyTorch CPU version first to avoid CUDA packages (~500MB vs ~2GB)
RUN pip install --no-cache-dir torch torchaudio --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir ".[ml]"

# Copy application source code
COPY --chown=hermes:hermes src/ src/

# Switch to non-root user for security
USER hermes

# Download spaCy model as non-root user
RUN python -m spacy download en_core_web_sm

# Expose the API port
EXPOSE 8080

# Allow build-time injection of Hermes LLM configuration
ARG HERMES_LLM_PROVIDER=echo
ARG HERMES_LLM_API_KEY=
ARG HERMES_LLM_MODEL=gpt-4o-mini
ARG HERMES_LLM_BASE_URL=https://api.openai.com/v1

# Set environment variables (can be overridden at runtime)
ENV PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/src \
    HERMES_LLM_PROVIDER=openai \
    HERMES_LLM_API_KEY=${OPENAI_API_KEY} \
    HERMES_LLM_MODEL=${HERMES_LLM_MODEL} \
    HERMES_LLM_BASE_URL=${HERMES_LLM_BASE_URL}

# Add healthcheck
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8080/ || exit 1

# Run the server as non-root user
CMD ["uvicorn", "hermes.main:app", "--host", "0.0.0.0", "--port", "8080"]

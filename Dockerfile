# Production-ready Dockerfile for Hermes API server
# Optimized for FastAPI/uvicorn with stateless design
FROM ghcr.io/c-daly/logos-foundry:0.1.0

# Add metadata labels following OCI standards
LABEL org.opencontainers.image.title="Hermes API" \
      org.opencontainers.image.description="Stateless language & embedding tools for Project LOGOS" \
      org.opencontainers.image.vendor="Project LOGOS Team" \
      org.opencontainers.image.source="https://github.com/c-daly/hermes" \
      org.opencontainers.image.licenses="MIT"

# Set working directory
WORKDIR /app/hermes

# Copy application code and configuration
COPY src/ ./src/
COPY pyproject.toml README.md ./

# Install CPU-only PyTorch first to avoid downloading CUDA (saves ~4GB)
# This must be done before poetry install to ensure we get the CPU version
RUN pip install --no-cache-dir \
    torch==2.0.1 \
    torchaudio==2.0.2 \
    --index-url https://download.pytorch.org/whl/cpu

# Install Hermes dependencies (including ML packages)
# Note: foundry base already has Poetry and common dependencies
# PyTorch is already installed above, so this won't re-download CUDA version
RUN poetry install --only main --extras ml --no-interaction --no-ansi

# Download spaCy model
RUN poetry run python -m spacy download en_core_web_sm

# Expose the API port
EXPOSE 8080

# Allow build-time injection of Hermes LLM configuration
ARG HERMES_LLM_PROVIDER=echo
ARG HERMES_LLM_API_KEY=
ARG HERMES_LLM_MODEL=gpt-4o-mini
ARG HERMES_LLM_BASE_URL=https://api.openai.com/v1

# Set environment variables (can be overridden at runtime)
ENV PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/hermes/src \
    HERMES_LLM_PROVIDER=${HERMES_LLM_PROVIDER} \
    HERMES_LLM_API_KEY=${HERMES_LLM_API_KEY} \
    HERMES_LLM_MODEL=${HERMES_LLM_MODEL} \
    HERMES_LLM_BASE_URL=${HERMES_LLM_BASE_URL}

# Add healthcheck
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8080/ || exit 1

# Run the server
CMD ["poetry", "run", "uvicorn", "hermes.main:app", "--host", "0.0.0.0", "--port", "8080"]

# Docker Deployment Guide for Hermes

This guide provides detailed instructions for deploying Hermes using Docker in various configurations.

## Quick Start

```bash
# Start production service
docker-compose up -d

# Check health
curl http://localhost:8080/

# View logs
docker-compose logs -f hermes
```

## Table of Contents

- [Production Deployment](#production-deployment)
- [Development Deployment](#development-deployment)
- [Configuration](#configuration)
- [Security Features](#security-features)
- [Monitoring and Health Checks](#monitoring-and-health-checks)
- [Troubleshooting](#troubleshooting)

## Production Deployment

### Using Docker Compose (Recommended)

The production `docker-compose.yml` includes all best practices:

```bash
# Start the service
docker-compose up -d

# Stop the service
docker-compose down

# Restart the service
docker-compose restart hermes

# View logs
docker-compose logs -f hermes

# Check resource usage
docker stats hermes
```

### Using Docker CLI

For manual control:

```bash
# Build the production image
docker build -t hermes:latest .

# Run the container
docker run -d \
  --name hermes \
  -p 8080:8080 \
  --restart unless-stopped \
  --security-opt no-new-privileges:true \
  --read-only \
  --tmpfs /tmp \
  --tmpfs /home/hermes/.cache \
  --cpus="2.0" \
  --memory="4g" \
  hermes:latest

# Check health
docker inspect --format='{{.State.Health.Status}}' hermes

# View logs
docker logs -f hermes

# Stop and remove
docker stop hermes
docker rm hermes
```

## Development Deployment

For rapid development with hot-reload:

```bash
# Using development docker-compose
docker-compose -f docker-compose.dev.yml up

# Or build and run manually
docker build -f Dockerfile.dev -t hermes:dev .
docker run -d \
  --name hermes-dev \
  -p 8080:8080 \
  -v $(pwd)/src:/app/src:ro \
  hermes:dev
```

**Development features:**
- No ML dependencies (faster build ~1 minute vs ~10 minutes)
- Source code volume mounting for hot-reload
- Auto-reload on code changes
- Smaller image size (~200MB vs ~2GB)

## Configuration

### Environment Variables

Set via `docker-compose.yml` or `-e` flag:

```yaml
environment:
  - PYTHONUNBUFFERED=1          # Enable real-time logging
  - LOG_LEVEL=INFO              # Set logging level
```

### Resource Limits

Configured in `docker-compose.yml`:

```yaml
deploy:
  resources:
    limits:
      cpus: '2.0'                # Max 2 CPU cores
      memory: 4G                 # Max 4GB RAM
    reservations:
      cpus: '0.5'                # Reserve 0.5 cores
      memory: 1G                 # Reserve 1GB RAM
```

### Port Mapping

Default: Host `8080` → Container `8080`

To change host port:
```bash
# Map to port 9000 on host
docker run -p 9000:8080 hermes:latest
```

Or in `docker-compose.yml`:
```yaml
ports:
  - "9000:8080"
```

## Security Features

### Non-Root User

The container runs as user `hermes` (UID 1000):

```dockerfile
USER hermes
```

### Read-Only Filesystem

Configured in `docker-compose.yml`:

```yaml
read_only: true
tmpfs:
  - /tmp                        # Writable temp directory
  - /home/hermes/.cache         # Model cache directory
```

### No Privilege Escalation

```yaml
security_opt:
  - no-new-privileges:true
```

### Network Security

- Exposes only port 8080
- No outbound restrictions (needed for model downloads)
- Stateless design (no data persistence)

## Monitoring and Health Checks

### Built-in Health Check

The Dockerfile includes a health check:

```dockerfile
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8080/ || exit 1
```

**Parameters:**
- `interval`: Check every 30 seconds
- `timeout`: Wait 10 seconds for response
- `start-period`: Wait 40 seconds before first check (model loading)
- `retries`: Mark unhealthy after 3 consecutive failures

### Check Health Status

```bash
# Using Docker
docker inspect --format='{{.State.Health.Status}}' hermes

# Using Docker Compose
docker-compose ps
```

**Possible states:**
- `starting` - Container is starting up
- `healthy` - Passing health checks
- `unhealthy` - Failed health checks

### View Health Check Logs

```bash
docker inspect hermes | jq '.[0].State.Health'
```

### Manual Health Check

```bash
# Check root endpoint
curl http://localhost:8080/

# Expected response:
# {
#   "name": "Hermes API",
#   "version": "0.1.0",
#   "description": "Stateless language & embedding tools for Project LOGOS",
#   "endpoints": ["/stt", "/tts", "/simple_nlp", "/embed_text"]
# }
```

## Troubleshooting

### Container won't start

**Check logs:**
```bash
docker logs hermes
# or
docker-compose logs hermes
```

**Common issues:**
- Port 8080 already in use: Change port mapping
- Insufficient memory: Increase memory limit
- Permission issues: Check volume mounts

### Health check failing

**Check container logs:**
```bash
docker logs hermes | tail -50
```

**Possible causes:**
- Model loading timeout (increase `start_period`)
- Application crashed (check logs)
- Port not accessible

### Slow startup

**Normal for production image:**
- First run: ~2-5 minutes (model downloads)
- Subsequent runs: ~30-60 seconds (model loading)

**Monitor startup:**
```bash
docker logs -f hermes
```

**Look for:**
```
INFO:     Loading Whisper model (base)...
INFO:     Whisper model loaded successfully
INFO:     Loading TTS model...
INFO:     TTS model loaded successfully
INFO:     Loading spaCy model...
INFO:     spaCy model loaded successfully
```

### Out of memory errors

**Increase memory limit:**

In `docker-compose.yml`:
```yaml
deploy:
  resources:
    limits:
      memory: 6G  # Increase from 4G
```

Or with Docker CLI:
```bash
docker run --memory="6g" hermes:latest
```

### Image too large

**Use development image:**
```bash
docker build -f Dockerfile.dev -t hermes:dev .
```

**Image sizes:**
- Production: ~2GB (with ML models)
- Development: ~200MB (no ML models)

### Permission errors with volumes

**Ensure correct ownership:**
```bash
# On host
chown -R 1000:1000 ./src
```

### Network issues during build

**SSL certificate errors:**

Add to Dockerfile (temporary):
```dockerfile
RUN pip install --trusted-host pypi.org --trusted-host files.pythonhosted.org ...
```

**DNS issues:**

Configure Docker DNS:
```json
{
  "dns": ["8.8.8.8", "8.8.4.4"]
}
```

## Advanced Usage

### Custom Configuration

Create a custom `docker-compose.override.yml`:

```yaml
version: '3.8'

services:
  hermes:
    environment:
      - LOG_LEVEL=DEBUG
    deploy:
      resources:
        limits:
          memory: 6G
```

Run with:
```bash
docker-compose up -d
```

### Multi-Container Setup

Example with reverse proxy:

```yaml
version: '3.8'

services:
  hermes:
    image: hermes:latest
    expose:
      - "8080"
    networks:
      - logos-network

  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf:ro
    depends_on:
      - hermes
    networks:
      - logos-network

networks:
  logos-network:
    driver: bridge
```

### Logging to External System

Configure Docker logging driver:

```yaml
logging:
  driver: syslog
  options:
    syslog-address: "tcp://192.168.0.1:514"
    tag: "hermes"
```

## Best Practices

1. **Use Docker Compose** for consistent deployments
2. **Pin image versions** in production (`hermes:0.1.0` not `hermes:latest`)
3. **Monitor resource usage** with `docker stats`
4. **Regular updates** to base image for security patches
5. **Backup configurations** (compose files, env files)
6. **Use secrets** for sensitive data (not needed for Hermes)
7. **Test health checks** before deploying to production
8. **Set resource limits** to prevent resource exhaustion
9. **Enable logging rotation** to manage disk space
10. **Run security scans** on images regularly

## Integration with LOGOS Ecosystem

Hermes is designed to integrate with other Project LOGOS components:

```yaml
version: '3.8'

services:
  hermes:
    image: hermes:latest
    networks:
      - logos

  sophia:
    image: sophia:latest
    depends_on:
      - hermes
    environment:
      - HERMES_URL=http://hermes:8080
    networks:
      - logos

  apollo:
    image: apollo:latest
    depends_on:
      - hermes
      - sophia
    environment:
      - HERMES_URL=http://hermes:8080
      - SOPHIA_URL=http://sophia:8000
    ports:
      - "3000:3000"
    networks:
      - logos

networks:
  logos:
    driver: bridge
```

## Additional Resources

- [Hermes GitHub Repository](https://github.com/c-daly/hermes)
- [Project LOGOS Documentation](https://github.com/c-daly/logos)
- [Docker Best Practices](https://docs.docker.com/develop/dev-best-practices/)
- [Docker Compose Reference](https://docs.docker.com/compose/compose-file/)

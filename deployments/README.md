# Hermes Deployment Configurations

This directory contains deployment configurations and tools for deploying Hermes API in various environments.

## Quick Links

- [Kubernetes Deployment](kubernetes/) - Production-ready Kubernetes manifests
- [Docker Swarm Deployment](swarm/) - Docker Swarm stack configuration
- [Integration Guide](../examples/INTEGRATION.md) - Integrating with Sophia and Apollo
- [Deployment Validation](validate.py) - Automated deployment verification script

## Deployment Options

### 1. Docker Compose (Recommended for Development)

The simplest way to run Hermes locally or in development:

```bash
# Production with ML models
docker-compose up -d

# Development without ML models (faster)
docker-compose -f docker-compose.dev.yml up
```

See [DOCKER.md](../DOCKER.md) for detailed Docker deployment instructions.

### 2. Kubernetes (Recommended for Production)

Production-ready Kubernetes deployment with autoscaling and health checks:

```bash
# Deploy to Kubernetes
kubectl apply -f deployments/kubernetes/deployment.yaml

# Optional: Deploy ingress
kubectl apply -f deployments/kubernetes/ingress.yaml
```

See [kubernetes/README.md](kubernetes/README.md) for detailed instructions.

### 3. Docker Swarm

Container orchestration with Docker Swarm:

```bash
# Deploy to Swarm
docker stack deploy -c deployments/swarm/stack.yml hermes
```

See [swarm/README.md](swarm/README.md) for detailed instructions.

## Configuration

### Environment Variables

Create a `.env` file based on [`.env.example`](../.env.example):

```bash
# Copy example file
cp .env.example .env

# Edit configuration
nano .env
```

Key configuration options:
- `HOST`: Server host address (default: 0.0.0.0)
- `PORT`: Server port (default: 8080)
- `LOG_LEVEL`: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
- `PYTHONUNBUFFERED`: Enable real-time logging (default: 1)

### Resource Requirements

**Minimum (without ML models):**
- CPU: 0.5 cores
- Memory: 512MB
- Storage: 500MB

**Recommended (with ML models):**
- CPU: 2 cores
- Memory: 4GB
- Storage: 5GB (includes model downloads)

**Production (high availability):**
- CPU: 2-4 cores per replica
- Memory: 4-8GB per replica
- Storage: 10GB
- Replicas: 2+ for redundancy

## Validation

Use the included validation script to verify your deployment:

```bash
# Validate local deployment
python deployments/validate.py

# Validate remote deployment
python deployments/validate.py --url https://hermes.example.com

# Wait for service and validate
python deployments/validate.py --wait --verbose
```

The script checks:
- ✓ Root endpoint availability
- ✓ Health endpoint functionality
- ✓ API documentation accessibility

## Health Checks

Hermes provides two endpoints for monitoring:

### GET / (Basic Info)
Returns API name, version, and available endpoints.

```bash
curl http://localhost:8080/
```

Response:
```json
{
  "name": "Hermes API",
  "version": "0.1.0",
  "description": "Stateless language & embedding tools for Project LOGOS",
  "endpoints": ["/stt", "/tts", "/simple_nlp", "/embed_text"]
}
```

### GET /health (Detailed Status)
Returns detailed health status including ML service availability.

```bash
curl http://localhost:8080/health
```

Response:
```json
{
  "status": "healthy",
  "version": "0.1.0",
  "services": {
    "stt": "available",
    "tts": "available",
    "nlp": "available",
    "embeddings": "available"
  }
}
```

Status values:
- `healthy`: All services available
- `degraded`: Some services unavailable (ML dependencies not installed)

## Integration with Project LOGOS

Hermes is designed to integrate seamlessly with other LOGOS components:

### Sophia Integration
```yaml
# docker-compose.yml or Kubernetes config
environment:
  - HERMES_URL=http://hermes:8080
```

### Apollo Integration
```yaml
# docker-compose.yml or Kubernetes config
environment:
  - HERMES_URL=http://hermes:8080
  - SOPHIA_URL=http://sophia:8000
```

See [INTEGRATION.md](../examples/INTEGRATION.md) for detailed integration examples and code snippets.

## Security Considerations

### Production Checklist

- [ ] Run containers as non-root user (✓ default in our images)
- [ ] Use read-only filesystem where possible (✓ configured in compose/k8s)
- [ ] Set resource limits to prevent resource exhaustion
- [ ] Use secrets for sensitive configuration
- [ ] Enable TLS/HTTPS for external access
- [ ] Implement network policies to restrict access
- [ ] Regular security updates for base images
- [ ] Monitor and audit access logs
- [ ] Use private container registry for production images
- [ ] Scan images for vulnerabilities

### Network Security

**Docker Compose:**
```yaml
networks:
  logos:
    driver: bridge
    internal: true  # Prevent external access
```

**Kubernetes:**
```yaml
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: hermes-policy
spec:
  podSelector:
    matchLabels:
      app: hermes
  ingress:
  - from:
    - podSelector:
        matchLabels:
          app: sophia
```

## Monitoring

### Logging

View logs in different environments:

**Docker Compose:**
```bash
docker-compose logs -f hermes
```

**Kubernetes:**
```bash
kubectl logs -n logos -l app=hermes -f
```

**Docker Swarm:**
```bash
docker service logs -f hermes_hermes
```

### Metrics

For production deployments, integrate with:
- **Prometheus**: Metrics collection
- **Grafana**: Metrics visualization
- **ELK Stack**: Log aggregation
- **Jaeger**: Distributed tracing

## Troubleshooting

### Common Issues

**Port already in use:**
```bash
# Change port in docker-compose.yml or use:
docker-compose -p hermes-alt up -d
```

**Out of memory:**
```bash
# Increase memory limits
# Docker Compose: deploy.resources.limits.memory
# Kubernetes: resources.limits.memory
```

**Models not loading:**
```bash
# Check logs for download errors
docker logs hermes

# Verify sufficient storage
df -h
```

**Service not responding:**
```bash
# Check container status
docker ps -a

# Check health
docker inspect --format='{{.State.Health.Status}}' hermes

# Test manually
curl http://localhost:8080/health
```

### Debug Mode

Enable debug logging:

```yaml
environment:
  - LOG_LEVEL=DEBUG
```

## Performance Tuning

### CPU Optimization

For CPU-only deployments (default):
```yaml
environment:
  - OMP_NUM_THREADS=4
  - MKL_NUM_THREADS=4
```

### Memory Optimization

Reduce memory usage:
- Use development image without ML models
- Reduce number of workers
- Enable model offloading (when supported)

### Network Optimization

For high-throughput scenarios:
- Enable HTTP/2
- Use connection pooling
- Increase worker processes
- Deploy multiple replicas with load balancing

## Backup and Recovery

### Configuration Backup

```bash
# Backup all deployment configs
tar -czf hermes-deployment-backup.tar.gz \
  docker-compose.yml \
  .env \
  deployments/
```

### Model Cache Backup

```bash
# Backup downloaded models
docker run --rm -v hermes_cache:/cache -v $(pwd):/backup \
  alpine tar -czf /backup/models-backup.tar.gz -C /cache .
```

## Additional Resources

- [Hermes GitHub Repository](https://github.com/c-daly/hermes)
- [Docker Documentation](../DOCKER.md)
- [API Documentation](http://localhost:8080/docs) (when running)
- [Project LOGOS](https://github.com/c-daly/logos)
- [Kubernetes Guide](kubernetes/README.md)
- [Docker Swarm Guide](swarm/README.md)

## Support

For deployment issues:
1. Check this documentation
2. Review logs from your deployment
3. Run the validation script: `python deployments/validate.py`
4. Check [GitHub Issues](https://github.com/c-daly/hermes/issues)
5. Consult the [LOGOS community](https://github.com/c-daly/logos)

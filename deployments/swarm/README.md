# Docker Swarm Deployment for Hermes

This directory contains Docker Swarm stack configurations for deploying Hermes API in a Docker Swarm cluster.

## Prerequisites

- Docker Engine 19.03.0+ with Swarm mode enabled
- At least 2 nodes (manager + worker) for high availability
- Hermes Docker image built and available

## Quick Start

1. **Initialize Docker Swarm (if not already done):**
   ```bash
   # On the manager node
   docker swarm init --advertise-addr <MANAGER-IP>
   
   # On worker nodes, use the join token from above
   docker swarm join --token <TOKEN> <MANAGER-IP>:2377
   ```

2. **Build and tag the image:**
   ```bash
   docker build -t hermes:latest .
   
   # If using a registry
   docker tag hermes:latest registry.example.com/hermes:latest
   docker push registry.example.com/hermes:latest
   ```

3. **Deploy the stack:**
   ```bash
   docker stack deploy -c deployments/swarm/stack.yml hermes
   ```

4. **Verify deployment:**
   ```bash
   # List services
   docker stack services hermes
   
   # Check service status
   docker service ps hermes_hermes
   
   # View logs
   docker service logs -f hermes_hermes
   ```

## Configuration

### Stack Configuration (stack.yml)

The stack configuration includes:
- **2 replicas** for high availability
- **Rolling updates** with zero downtime
- **Automatic rollback** on deployment failure
- **Resource limits** (2 CPU, 4GB memory)
- **Health checks** for service monitoring
- **Overlay network** for service communication
- **Security hardening** (read-only filesystem, no new privileges)

### Scaling

Scale the service manually:
```bash
# Scale to 5 replicas
docker service scale hermes_hermes=5

# Or update the stack file and redeploy
docker stack deploy -c deployments/swarm/stack.yml hermes
```

### Environment Variables

Modify the `environment` section in `stack.yml`:
```yaml
environment:
  - PYTHONUNBUFFERED=1
  - LOG_LEVEL=DEBUG  # Change log level
```

Or use Docker secrets for sensitive data:
```bash
# Create a secret
echo "production" | docker secret create env_mode -

# Reference in stack.yml
secrets:
  - env_mode
```

### Resource Limits

Adjust resource constraints in `stack.yml`:
```yaml
resources:
  limits:
    cpus: '4.0'      # Increase CPU limit
    memory: 8G       # Increase memory limit
  reservations:
    cpus: '1.0'
    memory: 2G
```

## Networking

### Internal Communication
Services in the same stack can communicate using service names:
```bash
# From another service in the logos-network
curl http://hermes:8080/
```

### External Access

**Option 1: Published Ports**
Access directly via node IP:
```bash
curl http://<NODE-IP>:8080/
```

**Option 2: Load Balancer**
Use an external load balancer pointing to all nodes on port 8080.

**Option 3: Traefik (Recommended)**
Traefik labels are included in the stack for automatic routing:
```yaml
labels:
  - "traefik.enable=true"
  - "traefik.http.routers.hermes.rule=Host(`hermes.example.com`)"
```

Deploy Traefik as a reverse proxy:
```bash
docker stack deploy -c traefik-stack.yml traefik
```

## Monitoring

### Service Status
```bash
# List all services in stack
docker stack services hermes

# Detailed service info
docker service inspect hermes_hermes --pretty

# Check running tasks
docker service ps hermes_hermes

# Real-time task monitoring
watch docker service ps hermes_hermes
```

### Logs
```bash
# View all logs
docker service logs hermes_hermes

# Follow logs
docker service logs -f hermes_hermes

# Tail last 100 lines
docker service logs --tail 100 hermes_hermes

# Filter by timestamp
docker service logs --since 30m hermes_hermes
```

### Resource Usage
```bash
# Check resource usage across nodes
docker node ls

# Inspect node
docker node inspect <NODE-ID>

# View tasks on a specific node
docker node ps <NODE-ID>
```

## Updates and Rollbacks

### Rolling Update
```bash
# Update image version
docker service update --image hermes:v0.2.0 hermes_hermes

# Or update the stack file and redeploy
docker stack deploy -c deployments/swarm/stack.yml hermes
```

### Rollback
```bash
# Automatic rollback on failure is configured in stack.yml
# Manual rollback to previous version
docker service rollback hermes_hermes
```

### Zero-Downtime Updates
The stack configuration uses `order: start-first` which ensures:
1. New tasks start before old tasks stop
2. No downtime during updates
3. Automatic rollback if new tasks fail health checks

## High Availability

### Multi-Node Setup
For production, run on at least 3 manager nodes:
```bash
# Promote a worker to manager
docker node promote <NODE-ID>

# View node status
docker node ls
```

### Constraints
Place replicas across multiple availability zones:
```yaml
placement:
  constraints:
    - node.role == worker
  preferences:
    - spread: node.labels.zone
```

Label nodes:
```bash
docker node update --label-add zone=us-west-1a <NODE-ID-1>
docker node update --label-add zone=us-west-1b <NODE-ID-2>
docker node update --label-add zone=us-west-1c <NODE-ID-3>
```

## Integration with LOGOS Stack

### Complete Stack Deployment
Deploy all LOGOS components in the same overlay network:

```yaml
# logos-stack.yml
version: '3.8'

services:
  hermes:
    image: hermes:latest
    networks:
      - logos-network
    deploy:
      replicas: 2
    # ... (configuration from stack.yml)

  sophia:
    image: sophia:latest
    networks:
      - logos-network
    environment:
      - HERMES_URL=http://hermes:8080
    deploy:
      replicas: 2

  apollo:
    image: apollo:latest
    networks:
      - logos-network
    environment:
      - HERMES_URL=http://hermes:8080
      - SOPHIA_URL=http://sophia:8000
    ports:
      - "3000:3000"
    deploy:
      replicas: 2

networks:
  logos-network:
    driver: overlay
    attachable: true
```

Deploy:
```bash
docker stack deploy -c logos-stack.yml logos
```

### Service Discovery
Services can discover each other using DNS:
```bash
# From Sophia container
curl http://hermes:8080/

# From Apollo container
curl http://hermes:8080/embed_text
```

## Troubleshooting

### Service Not Starting
```bash
# Check service status
docker service ps hermes_hermes --no-trunc

# View error logs
docker service logs hermes_hermes --tail 50

# Inspect service
docker service inspect hermes_hermes
```

**Common Issues:**
- Image not found: Ensure image is available on all nodes
- Port conflict: Check if port 8080 is already in use
- Resource constraints: Verify nodes have sufficient resources

### Health Check Failures
```bash
# Check health status
docker service ps hermes_hermes

# Test health endpoint manually
docker exec $(docker ps -q -f name=hermes) curl http://localhost:8080/
```

**Solutions:**
- Increase `start_period` in health check configuration
- Check application logs for errors
- Verify memory limits are sufficient

### Network Issues
```bash
# List networks
docker network ls

# Inspect network
docker network inspect logos-network

# Test connectivity between services
docker exec -it <CONTAINER-ID> ping hermes
```

### Node Failures
Swarm automatically reschedules tasks from failed nodes:
```bash
# Check node status
docker node ls

# Drain a node (for maintenance)
docker node update --availability drain <NODE-ID>

# Reactivate node
docker node update --availability active <NODE-ID>
```

## Cleanup

Remove the stack:
```bash
# Remove stack
docker stack rm hermes

# Remove network (after all stacks using it are removed)
docker network rm logos-network

# Leave swarm (on worker nodes)
docker swarm leave

# Force leave swarm (on manager)
docker swarm leave --force
```

## Best Practices

1. **Use specific image tags** instead of `latest` in production
2. **Enable automatic rollback** to ensure service availability
3. **Set resource limits** to prevent resource exhaustion
4. **Use overlay networks** for secure service communication
5. **Enable health checks** for automatic failure detection
6. **Deploy multiple replicas** across different nodes
7. **Use secrets** for sensitive configuration
8. **Monitor logs** regularly for issues
9. **Test rolling updates** in staging before production
10. **Backup stack configurations** in version control

## Additional Resources

- [Docker Swarm Documentation](https://docs.docker.com/engine/swarm/)
- [Docker Stack Reference](https://docs.docker.com/engine/reference/commandline/stack/)
- [Hermes GitHub Repository](https://github.com/c-daly/hermes)
- [Project LOGOS](https://github.com/c-daly/logos)

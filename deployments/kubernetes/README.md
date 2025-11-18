# Kubernetes Deployment for Hermes

This directory contains Kubernetes manifests for deploying Hermes API in a Kubernetes cluster.

## Prerequisites

- Kubernetes cluster (v1.24+)
- `kubectl` configured to access your cluster
- Hermes Docker image built and available to the cluster

## Quick Start

1. **Build and push the Docker image:**
   ```bash
   # Build the image
   docker build -t hermes:latest .
   
   # Tag for your registry (if using remote registry)
   docker tag hermes:latest your-registry.com/hermes:latest
   
   # Push to registry (if needed)
   docker push your-registry.com/hermes:latest
   ```

2. **Deploy to Kubernetes:**
   ```bash
   # Create namespace and deploy all resources
   kubectl apply -f deployments/kubernetes/deployment.yaml
   
   # Optionally deploy ingress
   kubectl apply -f deployments/kubernetes/ingress.yaml
   ```

3. **Verify deployment:**
   ```bash
   # Check pods
   kubectl get pods -n logos -l app=hermes
   
   # Check service
   kubectl get svc -n logos -l app=hermes
   
   # View logs
   kubectl logs -n logos -l app=hermes --tail=50 -f
   ```

## Configuration Files

### deployment.yaml
Complete deployment manifest including:
- **Namespace**: `logos` namespace for Project LOGOS components
- **Deployment**: 2 replicas with resource limits and health checks
- **Service**: ClusterIP service exposing port 8080
- **HorizontalPodAutoscaler**: Auto-scaling based on CPU/memory (2-10 replicas)

### ingress.yaml
Ingress configuration to expose Hermes externally:
- Supports NGINX ingress controller
- Configurable host and TLS
- Large body size for audio uploads

## Resource Requirements

**Per Pod:**
- **Requests**: 500m CPU, 1Gi memory
- **Limits**: 2 CPU, 4Gi memory

**Recommended for Production:**
- Minimum 2 replicas for high availability
- Node with at least 4 CPU cores and 8Gi memory

## Customization

### Using a Different Namespace

Edit the `namespace` field in all manifests:
```yaml
metadata:
  namespace: your-namespace
```

### Adjusting Replicas

Edit the deployment:
```yaml
spec:
  replicas: 3  # Change to desired number
```

### Using ConfigMap for Environment Variables

Create a ConfigMap:
```yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: hermes-config
  namespace: logos
data:
  LOG_LEVEL: "DEBUG"
  PYTHONUNBUFFERED: "1"
```

Reference in deployment:
```yaml
envFrom:
- configMapRef:
    name: hermes-config
```

### External Image Registry

Update the image reference in deployment:
```yaml
spec:
  template:
    spec:
      containers:
      - name: hermes
        image: your-registry.com/hermes:v0.1.0
        imagePullPolicy: Always
      imagePullSecrets:
      - name: registry-credentials
```

## Health Checks

The deployment includes both liveness and readiness probes:

**Liveness Probe:**
- Initial delay: 60 seconds (allows for model loading)
- Period: 30 seconds
- Timeout: 10 seconds
- Failure threshold: 3

**Readiness Probe:**
- Initial delay: 40 seconds
- Period: 10 seconds
- Timeout: 10 seconds
- Failure threshold: 3

## Scaling

### Manual Scaling
```bash
kubectl scale deployment hermes -n logos --replicas=5
```

### Automatic Scaling
The HorizontalPodAutoscaler automatically scales based on:
- CPU utilization: 70% target
- Memory utilization: 80% target
- Range: 2-10 replicas

## Monitoring

### Check Pod Status
```bash
kubectl get pods -n logos -l app=hermes -w
```

### View Logs
```bash
# All pods
kubectl logs -n logos -l app=hermes --tail=100 -f

# Specific pod
kubectl logs -n logos hermes-xxxxxxxxxx-xxxxx -f
```

### Describe Resources
```bash
kubectl describe deployment hermes -n logos
kubectl describe pod hermes-xxxxxxxxxx-xxxxx -n logos
```

### Check Resource Usage
```bash
kubectl top pods -n logos -l app=hermes
kubectl top nodes
```

## Accessing the API

### From Within the Cluster
```bash
# Service DNS name
curl http://hermes.logos.svc.cluster.local:8080/

# Using kubectl proxy
kubectl port-forward -n logos svc/hermes 8080:8080
curl http://localhost:8080/
```

### From Outside the Cluster

**Option 1: Port Forward (Development)**
```bash
kubectl port-forward -n logos svc/hermes 8080:8080
```

**Option 2: Ingress (Production)**
```bash
# After configuring ingress.yaml with your domain
curl https://hermes.example.com/
```

**Option 3: LoadBalancer Service (Cloud)**
```bash
# Change service type to LoadBalancer
kubectl patch svc hermes -n logos -p '{"spec":{"type":"LoadBalancer"}}'

# Get external IP
kubectl get svc hermes -n logos
```

## Integration with Other LOGOS Components

### Sophia Integration
Sophia can access Hermes using the internal service DNS:
```yaml
env:
- name: HERMES_URL
  value: "http://hermes.logos.svc.cluster.local:8080"
```

### Apollo Integration
Apollo can access Hermes similarly:
```yaml
env:
- name: HERMES_URL
  value: "http://hermes.logos.svc.cluster.local:8080"
```

### Complete LOGOS Stack
Deploy all components in the same namespace:
```bash
kubectl apply -f deployments/kubernetes/deployment.yaml
# Then deploy Sophia, Apollo, and Talos to the 'logos' namespace
```

## Troubleshooting

### Pods Not Starting
```bash
# Check pod events
kubectl describe pod -n logos -l app=hermes

# Check logs
kubectl logs -n logos -l app=hermes --tail=100
```

**Common issues:**
- Insufficient resources: Check node capacity
- Image pull errors: Verify image name and registry access
- Crash loop: Check application logs for errors

### Health Check Failures
```bash
# Test health endpoint manually
kubectl exec -it -n logos hermes-xxxxxxxxxx-xxxxx -- curl http://localhost:8080/
```

**Solutions:**
- Increase `initialDelaySeconds` if models take longer to load
- Check memory limits if seeing OOM errors
- Verify network connectivity

### High Memory Usage
```bash
# Check actual usage
kubectl top pods -n logos -l app=hermes
```

**Solutions:**
- Increase memory limits in deployment
- Reduce number of replicas if cluster capacity is limited
- Consider node scaling

## Cleanup

Remove all Hermes resources:
```bash
kubectl delete -f deployments/kubernetes/deployment.yaml
kubectl delete -f deployments/kubernetes/ingress.yaml

# Or delete by label
kubectl delete all -n logos -l app=hermes
```

## Best Practices

1. **Use specific image tags** instead of `latest` in production
2. **Set resource requests and limits** to prevent resource starvation
3. **Enable autoscaling** for variable workloads
4. **Monitor pod metrics** regularly
5. **Use namespaces** to isolate environments (dev, staging, prod)
6. **Implement network policies** for security
7. **Regular health checks** ensure reliability
8. **Log aggregation** for centralized logging (e.g., ELK, Loki)

## Additional Resources

- [Kubernetes Documentation](https://kubernetes.io/docs/)
- [Hermes GitHub Repository](https://github.com/c-daly/hermes)
- [Project LOGOS](https://github.com/c-daly/logos)

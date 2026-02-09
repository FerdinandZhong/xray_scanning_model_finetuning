# Deployment Guide

This directory contains deployment configurations for the X-ray VQA inference system.

## Contents

- `Dockerfile` - Multi-stage Docker build for inference
- `docker-compose.yaml` - Docker Compose for local/dev deployment
- `kubernetes/` - Kubernetes manifests for production
- `prometheus.yml` - Prometheus monitoring configuration

## Quick Start

### Option 1: Docker (Development/Testing)

```bash
# Build image
docker build -f deployment/Dockerfile -t xray-vqa-inference:latest .

# Run with docker-compose
cd deployment
docker-compose up -d

# Check logs
docker-compose logs -f xray-vqa-api

# Test API
curl http://localhost:8080/health
```

### Option 2: Kubernetes (Production)

```bash
# Create namespace
kubectl apply -f deployment/kubernetes/namespace.yaml

# Deploy model to persistent storage
kubectl create -f deployment/kubernetes/model-pvc.yaml
kubectl cp outputs/qwen25vl_stcray_lora <pod>:/models/

# Deploy application
kubectl apply -f deployment/kubernetes/deployment.yaml

# Check status
kubectl get pods -n customs-ai
kubectl logs -f -n customs-ai deployment/xray-vqa-api

# Test API
kubectl port-forward -n customs-ai service/xray-vqa-api-service 8080:80
curl http://localhost:8080/health
```

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `CUDA_VISIBLE_DEVICES` | `0,1` | GPU devices to use |
| `TENSOR_PARALLEL_SIZE` | `2` | Number of GPUs for tensor parallelism |
| `MAX_MODEL_LEN` | `2048` | Maximum sequence length |
| `GPU_MEMORY_UTILIZATION` | `0.9` | GPU memory utilization (0.0-1.0) |

## Resource Requirements

### Minimum (Development)
- 2x NVIDIA GPU (16GB VRAM each)
- 32GB RAM
- 50GB storage

### Recommended (Production)
- 2x NVIDIA V100/A100 (32GB VRAM each)
- 64GB RAM
- 100GB SSD storage

## API Endpoints

- `GET /health` - Health check
- `GET /docs` - API documentation
- `POST /api/v1/inspect` - X-ray inspection

## Monitoring

### Prometheus Metrics

Available at `/metrics`:
- `api_requests_total` - Total API requests
- `api_request_duration_seconds` - Request latency
- `model_inference_duration_seconds` - Inference time
- `gpu_memory_usage_bytes` - GPU memory usage

### Grafana Dashboards

Import dashboards from `grafana-dashboards/`:
- API Performance
- GPU Utilization
- Model Metrics

Access Grafana at `http://localhost:3000` (default: admin/admin)

## Scaling

### Horizontal Scaling (Multiple Replicas)

```bash
# Docker Compose
docker-compose up --scale xray-vqa-api=3

# Kubernetes
kubectl scale deployment xray-vqa-api -n customs-ai --replicas=5
```

### Vertical Scaling (More GPUs)

Update `TENSOR_PARALLEL_SIZE` and GPU resource requests:

```yaml
# kubernetes/deployment.yaml
resources:
  requests:
    nvidia.com/gpu: "4"  # Increase GPU count
env:
  - name: TENSOR_PARALLEL_SIZE
    value: "4"
```

## Troubleshooting

### GPU Not Detected

```bash
# Check NVIDIA runtime
docker run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi

# Verify Kubernetes GPU plugin
kubectl get nodes -o json | jq '.items[].status.capacity'
```

### Out of Memory

Reduce batch size or GPU memory utilization:

```yaml
env:
  - name: GPU_MEMORY_UTILIZATION
    value: "0.8"  # Reduce from 0.9
```

### Slow Inference

- Increase `TENSOR_PARALLEL_SIZE` for more GPUs
- Enable bf16: `--dtype bfloat16`
- Use faster GPU (V100 -> A100)

### Connection Refused

Check service is running and ports are correct:

```bash
# Docker
docker-compose ps
docker-compose logs xray-vqa-api

# Kubernetes
kubectl get svc -n customs-ai
kubectl describe pod -n customs-ai <pod-name>
```

## Security Considerations

1. **Model Access**: Ensure model files are read-only in production
2. **API Authentication**: Add authentication layer (not included)
3. **Network Policies**: Restrict ingress/egress in Kubernetes
4. **Secrets Management**: Use Kubernetes secrets for sensitive data
5. **Image Scanning**: Scan Docker images for vulnerabilities

## Cost Optimization

1. **Auto-scaling**: Enable HPA to scale down during low traffic
2. **Spot Instances**: Use preemptible/spot instances for cost savings
3. **Mixed Precision**: Use bf16 for faster inference and lower memory
4. **Batch Processing**: Process multiple scans in parallel when possible
5. **GPU Sharing**: Use MIG (Multi-Instance GPU) for A100 GPUs

## Performance Tuning

### vLLM Configuration

```python
# inference/api_server.py
vllm_server = XrayVQAServer(
    model_path=args.model,
    tensor_parallel_size=2,
    gpu_memory_utilization=0.9,
    max_model_len=2048,
    dtype="bfloat16",  # Use bf16 for speed
    enforce_eager=False,  # Use CUDA graphs
)
```

### Kubernetes Node Affinity

```yaml
# Pin to high-performance GPU nodes
affinity:
  nodeAffinity:
    requiredDuringSchedulingIgnoredDuringExecution:
      nodeSelectorTerms:
        - matchExpressions:
            - key: gpu-type
              operator: In
              values:
                - a100
                - v100
```

## Backup and Recovery

### Model Backup

```bash
# Backup trained model
kubectl exec -n customs-ai <pod> -- tar czf /tmp/model.tar.gz /models
kubectl cp customs-ai/<pod>:/tmp/model.tar.gz ./backup/model-$(date +%Y%m%d).tar.gz
```

### Database Backup (if using persistent metrics)

```bash
# Backup Prometheus data
kubectl exec -n customs-ai <prometheus-pod> -- \
  tar czf /tmp/prometheus.tar.gz /prometheus
```

## Support

For issues and questions:
- Check logs: `kubectl logs -n customs-ai <pod>`
- Review metrics: `http://localhost:9090` (Prometheus)
- API docs: `http://localhost:8080/docs`

# EvoGuard Kubernetes Deployment

Kubernetes deployment configuration for EvoGuard adversarial attack simulation platform.

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Kubernetes Cluster                        │
│                                                              │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐      │
│  │   Ingress   │    │   Ingress   │    │  Prometheus │      │
│  │   (nginx)   │    │   (nginx)   │    │  + Grafana  │      │
│  └──────┬──────┘    └──────┬──────┘    └─────────────┘      │
│         │                  │                                 │
│  ┌──────▼──────┐    ┌──────▼──────┐                         │
│  │ api-service │───▶│ ml-service  │◀── GPU (optional)       │
│  │   (Go)      │    │  (Python)   │                         │
│  │  HPA: 2-10  │    │  replicas:1 │                         │
│  └──────┬──────┘    └──────┬──────┘                         │
│         │                  │                                 │
│  ┌──────▼──────┐    ┌──────▼──────┐                         │
│  │ PostgreSQL  │    │    Redis    │                         │
│  │  (Bitnami)  │    │  (Bitnami)  │                         │
│  │   PVC: 8Gi  │    │  PVC: 2Gi   │                         │
│  └─────────────┘    └─────────────┘                         │
└─────────────────────────────────────────────────────────────┘
```

## Prerequisites

- [minikube](https://minikube.sigs.k8s.io/docs/start/) v1.30+
- [helm](https://helm.sh/docs/intro/install/) v3.12+
- [kubectl](https://kubernetes.io/docs/tasks/tools/) v1.28+
- [Docker](https://docs.docker.com/engine/install/) 24+
- NVIDIA GPU + drivers (optional, for GPU inference)

## Quick Start

### 1. Setup Cluster

```bash
# Start minikube with GPU support (auto-detects GPU)
./scripts/setup-minikube-gpu.sh

# Or manually:
minikube start --gpus=all --memory=8192 --cpus=4
```

### 2. Deploy Services

```bash
# Deploy shared services (PostgreSQL, Redis)
cd helm/shared-services
helm dependency update
helm install shared-services . -f values-local.yaml

# Deploy api-service
cd ../api-service
helm install api-service . -f values-local.yaml

# Deploy ml-service (with GPU)
cd ../ml-service
helm install ml-service . -f values-local.yaml

# Or without GPU
helm install ml-service . -f values-local-cpu.yaml
```

### 3. Access Services

```bash
# Enable tunnel for LoadBalancer services
minikube tunnel

# Add to /etc/hosts
echo "$(minikube ip) api.evoguard.local ml.evoguard.local" | sudo tee -a /etc/hosts

# Test
curl http://api.evoguard.local/health
curl http://ml.evoguard.local/health
```

### 4. Run E2E Demo

```bash
./scripts/e2e-battle-demo.sh
```

## Helm Charts

| Chart | Description |
|-------|-------------|
| `api-service` | Go REST API for battle management |
| `ml-service` | Python ML inference service with GPU support |
| `shared-services` | PostgreSQL, Redis (Bitnami subcharts) |

## Configuration

### GPU Support

Enable GPU in `ml-service/values.yaml`:

```yaml
gpu:
  enabled: true
  count: 1
  resourceName: "nvidia.com/gpu"
```

### Resource Limits

Adjust in each chart's `values.yaml`:

```yaml
resources:
  limits:
    cpu: 2000m
    memory: 4Gi
  requests:
    cpu: 500m
    memory: 2Gi
```

### Autoscaling

Enable HPA in `api-service/values.yaml`:

```yaml
autoscaling:
  enabled: true
  minReplicas: 2
  maxReplicas: 10
  targetCPUUtilizationPercentage: 70
```

## Directory Structure

```
k8s/
├── helm/
│   ├── api-service/           # Go API service chart
│   │   ├── Chart.yaml
│   │   ├── values.yaml        # Default values
│   │   ├── values-local.yaml  # Local dev values
│   │   └── templates/
│   │
│   ├── ml-service/            # Python ML service chart
│   │   ├── Chart.yaml
│   │   ├── values.yaml
│   │   ├── values-local.yaml      # Local with GPU
│   │   ├── values-local-cpu.yaml  # Local CPU-only
│   │   └── templates/
│   │
│   └── shared-services/       # PostgreSQL, Redis
│       ├── Chart.yaml
│       ├── values.yaml
│       ├── values-local.yaml
│       └── templates/
│
├── scripts/
│   ├── setup-minikube-gpu.sh  # Cluster setup
│   ├── teardown.sh            # Cleanup
│   └── e2e-battle-demo.sh     # Demo script
│
└── README.md
```

## Useful Commands

```bash
# View pods
kubectl get pods

# View logs
kubectl logs -f deploy/ml-service

# Port forward
kubectl port-forward svc/ml-service 8000:8000

# Scale deployment
kubectl scale deploy/api-service --replicas=3

# Check GPU allocation
kubectl describe nodes | grep -A5 "nvidia.com/gpu"

# Uninstall all
./scripts/teardown.sh
```

## Troubleshooting

### GPU not detected

```bash
# Check NVIDIA plugin
kubectl get pods -n kube-system | grep nvidia

# Check GPU resources
kubectl describe node | grep nvidia.com/gpu
```

### Pods not starting

```bash
# Check events
kubectl describe pod <pod-name>

# Check logs
kubectl logs <pod-name> --previous
```

### Database connection failed

```bash
# Check PostgreSQL pod
kubectl get pods | grep postgresql

# Test connection
kubectl exec -it deploy/api-service -- nc -zv shared-services-postgresql 5432
```

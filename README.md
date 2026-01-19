# EvoGuard

> Adversarial Learning-based Self-improving Content Moderation System

EvoGuard is an MLOps project where attacker and defender models compete to evolve content moderation capabilities. The system automatically improves detection rates through adversarial training.

## Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                            EvoGuard Architecture                             │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│   ┌──────────────┐         ┌──────────────┐         ┌──────────────┐       │
│   │   Attacker   │ ──────► │  Battle API  │ ◄────── │   Defender   │       │
│   │  (Ollama +   │         │    (Go)      │         │  (BERT +     │       │
│   │  Mistral 7B) │         │              │         │   QLoRA)     │       │
│   └──────────────┘         └──────────────┘         └──────────────┘       │
│          │                        │                        │                │
│          │                        ▼                        │                │
│          │              ┌──────────────────┐               │                │
│          │              │  Model Monitoring │               │                │
│          │              │  - Drift Detection│               │                │
│          │              │  - Alert System   │               │                │
│          │              │  - Auto Retrain   │               │                │
│          │              └──────────────────┘               │                │
│          │                        │                        │                │
│          ▼                        ▼                        ▼                │
│   ┌──────────────────────────────────────────────────────────────┐         │
│   │                      Feature Store                            │         │
│   │     Offline (Parquet)  │  Online (Redis)  │  Registry (PG)   │         │
│   └──────────────────────────────────────────────────────────────┘         │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Key Features

- **Adversarial Training**: Attacker generates evasion patterns, defender learns to detect them
- **QLoRA Fine-tuning**: Memory-efficient training on consumer GPUs (8GB VRAM)
- **Model Monitoring**: Drift detection, confidence tracking, automated alerts
- **Feature Store**: Offline (Parquet) + Online (Redis) for ML features
- **A/B Testing**: Champion/Challenger model comparison
- **Auto Remediation**: Automatic retraining and rollback triggers

## Tech Stack

| Component | Technology |
|-----------|------------|
| Backend API | Go 1.21+ (Gin) |
| ML Service | Python 3.12 (FastAPI) |
| Attacker LLM | Ollama + Mistral 7B |
| Defender Model | BERT + QLoRA |
| Feature Store | Parquet + Redis + PostgreSQL |
| Experiment Tracking | MLflow |
| Monitoring | Prometheus + Grafana |
| Infrastructure | Docker, Kubernetes |

## Quick Start

### Prerequisites

- Docker & Docker Compose
- NVIDIA GPU (optional, for training)
- Go 1.21+ (for API development)
- Python 3.12+ (for ML development)

### 1. Clone and Setup

```bash
git clone https://github.com/ressKim-io/EvoGuard.git
cd EvoGuard
make setup
```

### 2. Start Infrastructure

```bash
# Start all services (PostgreSQL, Redis, MLflow, Prometheus, Grafana)
make docker-up

# Or start specific services
docker compose -f infra/docker-compose.yml up -d postgres redis
```

### 3. Run Services

```bash
# API Service (Go)
cd api-service
make run

# ML Service (Python)
cd ml-service
make run
```

### 4. Access UIs

| Service | URL |
|---------|-----|
| API Docs | http://localhost:8080/swagger |
| MLflow | http://localhost:5000 |
| Grafana | http://localhost:3000 |
| Prometheus | http://localhost:9090 |

## Project Structure

```
EvoGuard/
├── api-service/          # Go REST API (Battle management)
│   ├── cmd/              # Application entry points
│   ├── internal/         # Internal packages
│   │   ├── adapter/      # HTTP handlers, repositories
│   │   ├── domain/       # Business entities
│   │   └── usecase/      # Business logic
│   └── pkg/              # Public packages
│
├── ml-service/           # Python ML Service
│   ├── src/ml_service/
│   │   ├── api/          # FastAPI endpoints
│   │   ├── core/         # Config, exceptions, protocols
│   │   ├── feature_store/# Offline, Online, Registry
│   │   ├── models/       # Classifier implementations
│   │   ├── monitoring/   # Drift detection, alerts
│   │   ├── services/     # Inference service
│   │   └── training/     # QLoRA fine-tuning
│   └── tests/
│
├── attacker/             # Adversarial attack generator
│   ├── strategies/       # Attack strategies
│   └── prompts/          # LLM prompts
│
├── infra/                # Infrastructure configs
│   ├── docker-compose.yml
│   ├── grafana/          # Dashboards, alerting
│   └── k8s/              # Kubernetes manifests
│
└── data/                 # Data directory (DVC managed)
```

## Development

### Running Tests

```bash
# All tests
make test

# API Service tests
cd api-service && make test

# ML Service tests
cd ml-service && make test
```

### Code Quality

```bash
# Lint all
make lint

# Format code
make fmt
```

### Training a Model

```bash
cd ml-service

# Install training dependencies
uv pip install --group training

# Run training
python scripts/train.py --data data/train.csv --epochs 3 --batch-size 4
```

## Model Monitoring

The ML service includes comprehensive monitoring:

- **Data Drift**: PSI-based detection on feature distributions
- **Concept Drift**: Performance degradation tracking
- **Feature Drift**: Per-feature drift scores
- **Confidence Monitoring**: Low confidence prediction alerts
- **Automated Actions**: Retrain triggers, model rollback

### Grafana Dashboard

Access at http://localhost:3000 with default credentials (admin/admin).

Panels include:
- Model Performance (F1, Accuracy, Precision, Recall)
- Drift Detection Scores
- Prediction Distribution
- Automated Actions Timeline
- System Health

## Configuration

Environment variables are documented in `.env.example`:

```bash
# Copy and customize
cp .env.example .env
```

Key configurations:
- `ML_SERVICE_URL`: ML service endpoint
- `REDIS_URL`: Redis connection string
- `DATABASE_URL`: PostgreSQL connection string
- `MLFLOW_TRACKING_URI`: MLflow server URL

## Architecture

For detailed architecture documentation, see [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for development guidelines.

## License

MIT License - see [LICENSE](LICENSE) for details.

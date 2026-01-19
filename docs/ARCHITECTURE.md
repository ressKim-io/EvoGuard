# EvoGuard Architecture

## System Overview

```mermaid
flowchart TB
    subgraph Client["Client Layer"]
        Web[Web UI]
        CLI[CLI Tools]
    end

    subgraph API["API Layer"]
        Gateway[API Gateway]
        BattleAPI[Battle API<br/>Go/Gin]
    end

    subgraph ML["ML Layer"]
        MLService[ML Service<br/>Python/FastAPI]
        Inference[Inference Engine]
        Training[Training Pipeline<br/>QLoRA]
    end

    subgraph Attack["Attack Layer"]
        Attacker[Attacker Service]
        Ollama[Ollama<br/>Mistral 7B]
        Strategies[Attack Strategies]
    end

    subgraph Monitor["Monitoring Layer"]
        Drift[Drift Detection]
        Alerts[Alert System]
        Actions[Auto Actions]
        Metrics[Metrics Collector]
    end

    subgraph Storage["Storage Layer"]
        subgraph FeatureStore["Feature Store"]
            Offline[Offline Store<br/>Parquet]
            Online[Online Store<br/>Redis]
            Registry[Registry<br/>PostgreSQL]
        end
        MLflow[MLflow<br/>Model Registry]
        DB[(PostgreSQL)]
    end

    subgraph Observability["Observability"]
        Prometheus[Prometheus]
        Grafana[Grafana]
    end

    Client --> Gateway
    Gateway --> BattleAPI
    BattleAPI --> MLService
    BattleAPI --> Attacker

    MLService --> Inference
    MLService --> Training
    Training --> MLflow

    Attacker --> Ollama
    Attacker --> Strategies

    MLService --> Monitor
    Monitor --> Drift
    Monitor --> Alerts
    Monitor --> Actions
    Monitor --> Metrics

    MLService --> FeatureStore
    BattleAPI --> DB

    Metrics --> Prometheus
    Prometheus --> Grafana
```

## Component Details

### 1. API Service (Go)

```
api-service/
├── cmd/api/              # Entry point
├── internal/
│   ├── adapter/
│   │   ├── http/         # HTTP handlers
│   │   │   └── handler/  # Request handlers
│   │   └── repository/   # Data access
│   ├── domain/           # Business entities
│   │   └── entity/       # Battle, Round, etc.
│   └── usecase/          # Business logic
│       └── battle/       # Battle operations
└── pkg/                  # Shared packages
```

**Responsibilities:**
- Battle lifecycle management (Create, Start, Run, Complete)
- Round execution orchestration
- Coordination between Attacker and Defender
- REST API endpoints

### 2. ML Service (Python)

```
ml-service/
├── src/ml_service/
│   ├── api/              # FastAPI routes
│   ├── core/             # Config, exceptions
│   ├── feature_store/    # Feature management
│   │   ├── offline/      # Parquet storage
│   │   ├── online/       # Redis cache
│   │   └── registry/     # Feature metadata
│   ├── models/           # Classifier implementations
│   ├── monitoring/       # Model monitoring
│   │   ├── drift/        # Drift detection
│   │   ├── alerts/       # Alert system
│   │   └── actions/      # Auto remediation
│   ├── services/         # Business services
│   └── training/         # QLoRA fine-tuning
```

**Responsibilities:**
- Text classification inference
- Model monitoring and drift detection
- Feature engineering and storage
- Model training with QLoRA

### 3. Attacker Service (Python)

```
attacker/
├── strategies/           # Attack implementations
│   ├── homoglyph.py     # Character substitution
│   ├── leetspeak.py     # L33t speak transform
│   ├── unicode_evasion.py
│   ├── llm_evasion.py   # LLM-based mutation
│   └── adversarial_llm.py
├── prompts/              # LLM prompt templates
└── orchestrator.py       # Strategy coordination
```

**Attack Strategies:**
1. **Homoglyph**: Replace characters with visually similar ones
2. **Leetspeak**: Convert to l33t speak patterns
3. **Unicode Evasion**: Insert zero-width characters
4. **LLM Evasion**: Use LLM to generate semantic variations
5. **Adversarial LLM**: Learn from failed attempts

### 4. Feature Store

```mermaid
flowchart LR
    subgraph Offline["Offline Store (Parquet)"]
        Historical[Historical Features]
        Batch[Batch Processing]
    end

    subgraph Online["Online Store (Redis)"]
        Cache[Low-latency Cache]
        TTL[TTL Management]
    end

    subgraph Registry["Registry (PostgreSQL)"]
        Metadata[Feature Metadata]
        Versions[Version Control]
    end

    Offline --> |Sync| Online
    Registry --> |Schema| Offline
    Registry --> |Schema| Online
```

**Features:**
- Text features (length, word count, unicode ratio)
- Battle features (detection rate, evasion rate)
- User features (historical behavior)

### 5. Model Monitoring

```mermaid
flowchart TB
    Predictions[Predictions] --> Logger[Prediction Logger]
    Logger --> Buffer[Buffer]

    Buffer --> DataDrift[Data Drift<br/>PSI Detection]
    Buffer --> ConceptDrift[Concept Drift<br/>Performance Decay]
    Buffer --> FeatureDrift[Feature Drift<br/>Distribution Shift]
    Buffer --> Confidence[Confidence<br/>Monitoring]

    DataDrift --> AlertEngine[Alert Engine]
    ConceptDrift --> AlertEngine
    FeatureDrift --> AlertEngine
    Confidence --> AlertEngine

    AlertEngine --> |Threshold| Retrain[Retrain Trigger]
    AlertEngine --> |Critical| Rollback[Model Rollback]
    AlertEngine --> |Notify| Webhook[Webhooks]
```

**Drift Detection:**
- **Data Drift**: PSI (Population Stability Index) on input distributions
- **Concept Drift**: F1 score degradation over time
- **Feature Drift**: Per-feature distribution changes

**Automated Actions:**
- **Retrain Trigger**: Schedule retraining when drift detected
- **Model Rollback**: Revert to previous stable version
- **A/B Testing**: Compare champion vs challenger

### 6. Training Pipeline

```mermaid
flowchart LR
    Data[Training Data] --> Preprocess[Data Processor]
    Preprocess --> Tokenize[Tokenization]
    Tokenize --> Train[QLoRA Trainer]

    Train --> |4-bit| Quantize[Quantization]
    Train --> |LoRA| Adapters[LoRA Adapters]

    Train --> MLflow[MLflow Tracking]
    MLflow --> Registry[Model Registry]

    Registry --> |Challenger| ABTest[A/B Testing]
    ABTest --> |Promote| Champion[Champion Model]
```

**QLoRA Configuration:**
- 4-bit quantization (NF4)
- LoRA rank: 16, alpha: 32
- Gradient checkpointing enabled
- Optimized for 8GB VRAM

## Data Flow

### Battle Execution Flow

```mermaid
sequenceDiagram
    participant Client
    participant API as Battle API
    participant Attacker
    participant Defender as ML Service
    participant DB
    participant Monitor

    Client->>API: POST /battles
    API->>DB: Create Battle
    API-->>Client: Battle Created

    Client->>API: POST /battles/{id}/start

    loop Each Round
        API->>Attacker: Generate Attack
        Attacker->>Attacker: Apply Strategies
        Attacker-->>API: Mutated Text

        API->>Defender: POST /classify
        Defender->>Monitor: Log Prediction
        Defender-->>API: Classification Result

        API->>DB: Save Round Result
    end

    API->>DB: Update Battle Stats
    API-->>Client: Battle Complete
```

### Monitoring Flow

```mermaid
sequenceDiagram
    participant ML as ML Service
    participant Logger as Prediction Logger
    participant Drift as Drift Detectors
    participant Alert as Alert Engine
    participant Action as Auto Actions

    ML->>Logger: Log Prediction
    Logger->>Logger: Buffer Predictions

    Logger->>Drift: Check Drift (periodic)
    Drift->>Drift: Calculate Scores

    alt Drift Detected
        Drift->>Alert: Fire Alert
        Alert->>Alert: Evaluate Rules

        alt Critical Alert
            Alert->>Action: Trigger Rollback
        else Warning Alert
            Alert->>Action: Schedule Retrain
        end
    end
```

## Deployment Architecture

```mermaid
flowchart TB
    subgraph K8s["Kubernetes Cluster"]
        subgraph Services["Services"]
            API[API Service<br/>Deployment]
            ML[ML Service<br/>Deployment]
            Attacker[Attacker<br/>Deployment]
        end

        subgraph Data["Data Layer"]
            PG[(PostgreSQL<br/>StatefulSet)]
            Redis[(Redis<br/>StatefulSet)]
        end

        subgraph Monitoring["Monitoring"]
            Prom[Prometheus<br/>Deployment]
            Graf[Grafana<br/>Deployment]
        end

        subgraph ML_Infra["ML Infrastructure"]
            MLflow[MLflow<br/>Deployment]
            Ollama[Ollama<br/>DaemonSet]
        end
    end

    Ingress[Ingress Controller] --> Services
    Services --> Data
    Services --> Monitoring
    ML --> ML_Infra
```

## Security Considerations

1. **API Authentication**: JWT-based authentication
2. **Service Communication**: mTLS between services
3. **Secrets Management**: Kubernetes secrets / Vault
4. **Network Policies**: Restrict pod-to-pod communication
5. **Input Validation**: Sanitize all user inputs

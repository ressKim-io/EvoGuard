# 📡 06. API 명세

> Battle API, Model API, Inference API의 상세 스펙

---

## 🌐 API 개요

### 서비스 구조

```
┌─────────────────────────────────────────────────────────────────┐
│                         CLIENTS                                 │
│              (Dashboard, CLI, External Apps)                    │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    API GATEWAY (Go/Gin)                         │
│                     http://localhost:8080                       │
│                                                                 │
│  ┌────────────────┐  ┌────────────────┐  ┌────────────────┐   │
│  │  /api/v1/      │  │  /api/v1/      │  │  /api/v1/      │   │
│  │  battles/*     │  │  models/*      │  │  metrics/*     │   │
│  └────────────────┘  └────────────────┘  └────────────────┘   │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
                              │
          ┌───────────────────┼───────────────────┐
          ▼                   ▼                   ▼
┌──────────────────┐ ┌──────────────────┐ ┌──────────────────┐
│  Inference API   │ │    Ollama        │ │    MLflow        │
│  (FastAPI)       │ │  (로컬 LLM)      │ │  (Model Registry)│
│  :8001           │ │  :11434          │ │  :5000           │
└──────────────────┘ └──────────────────┘ └──────────────────┘
```

### 공통 응답 형식

```json
// 성공 응답
{
  "success": true,
  "data": { ... },
  "meta": {
    "timestamp": "2025-01-17T12:00:00Z",
    "request_id": "req_abc123"
  }
}

// 에러 응답
{
  "success": false,
  "error": {
    "code": "BATTLE_NOT_FOUND",
    "message": "Battle with id 'xxx' not found",
    "details": { ... }
  },
  "meta": {
    "timestamp": "2025-01-17T12:00:00Z",
    "request_id": "req_abc123"
  }
}
```

### 공통 에러 코드

| HTTP Status | Code | 설명 |
|-------------|------|------|
| 400 | `INVALID_REQUEST` | 잘못된 요청 형식 |
| 400 | `VALIDATION_ERROR` | 유효성 검증 실패 |
| 401 | `UNAUTHORIZED` | 인증 필요 |
| 404 | `NOT_FOUND` | 리소스 없음 |
| 409 | `CONFLICT` | 리소스 충돌 |
| 429 | `RATE_LIMITED` | 요청 한도 초과 |
| 500 | `INTERNAL_ERROR` | 서버 내부 오류 |
| 503 | `SERVICE_UNAVAILABLE` | 서비스 이용 불가 |

---

## ⚔️ Battle API

### 1. 배틀 생성

새로운 배틀 세션을 생성합니다.

```
POST /api/v1/battles
```

**Request Body:**

```json
{
  "rounds": 100,                    // 라운드 수 (1-1000)
  "attack_strategy": "mixed",       // 공격 전략
  "attack_config": {                // 전략별 설정 (선택)
    "llm_weight": 0.5,              // LLM 전략 비중
    "unicode_weight": 0.3,
    "homoglyph_weight": 0.2
  },
  "defender_alias": "champion",     // 방어 모델 alias
  "dataset": "toxic_samples",       // 공격에 사용할 데이터셋
  "async": true                     // 비동기 실행 여부
}
```

**공격 전략 옵션:**

| Strategy | 설명 |
|----------|------|
| `unicode_evasion` | 유니코드 문자 변형 |
| `homoglyph` | 동형 문자 치환 |
| `leetspeak` | 리트스피크 변형 |
| `llm_evasion` | LLM 기반 창의적 우회 |
| `adversarial_llm` | 적대적 학습 기반 |
| `mixed` | 전략 혼합 (가중치 적용) |

**Response (201 Created):**

```json
{
  "success": true,
  "data": {
    "battle_id": "bat_abc123def456",
    "status": "pending",
    "config": {
      "rounds": 100,
      "attack_strategy": "mixed",
      "defender_alias": "champion"
    },
    "created_at": "2025-01-17T12:00:00Z",
    "estimated_duration_seconds": 300
  }
}
```

### 2. 배틀 상태 조회

```
GET /api/v1/battles/{battle_id}
```

**Response (200 OK):**

```json
{
  "success": true,
  "data": {
    "battle_id": "bat_abc123def456",
    "status": "running",            // pending, running, completed, failed
    "progress": {
      "completed_rounds": 45,
      "total_rounds": 100,
      "percentage": 45.0
    },
    "current_stats": {
      "detection_count": 32,
      "evasion_count": 13,
      "detection_rate": 0.711,
      "evasion_rate": 0.289
    },
    "config": { ... },
    "created_at": "2025-01-17T12:00:00Z",
    "started_at": "2025-01-17T12:00:05Z",
    "updated_at": "2025-01-17T12:02:30Z"
  }
}
```

### 3. 배틀 라운드 목록 조회

```
GET /api/v1/battles/{battle_id}/rounds
```

**Query Parameters:**

| Parameter | Type | Default | 설명 |
|-----------|------|---------|------|
| `page` | int | 1 | 페이지 번호 |
| `page_size` | int | 20 | 페이지 크기 (max: 100) |
| `detected` | bool | - | 탐지 여부 필터 |
| `strategy` | string | - | 전략 필터 |

**Response (200 OK):**

```json
{
  "success": true,
  "data": {
    "rounds": [
      {
        "round_number": 1,
        "original_text": "이 쓰레기 같은 놈아",
        "evasion_text": "이 쓰ㄹㅔ기 같ㅇㅡㄴ 놈아",
        "attack_strategy": "unicode_evasion",
        "classification": {
          "toxic_score": 0.85,
          "is_detected": true,
          "confidence": 0.85,
          "model_version": "v3"
        },
        "created_at": "2025-01-17T12:00:10Z"
      },
      {
        "round_number": 2,
        "original_text": "바보 멍청이",
        "evasion_text": "바ㅂㅗ 멍ㅊㅓng이",
        "attack_strategy": "llm_evasion",
        "classification": {
          "toxic_score": 0.42,
          "is_detected": false,
          "confidence": 0.58,
          "model_version": "v3"
        },
        "created_at": "2025-01-17T12:00:15Z"
      }
    ],
    "pagination": {
      "page": 1,
      "page_size": 20,
      "total_items": 100,
      "total_pages": 5
    }
  }
}
```

### 4. 배틀 중지

```
POST /api/v1/battles/{battle_id}/stop
```

**Response (200 OK):**

```json
{
  "success": true,
  "data": {
    "battle_id": "bat_abc123def456",
    "status": "completed",
    "final_stats": {
      "completed_rounds": 45,
      "detection_count": 32,
      "evasion_count": 13,
      "detection_rate": 0.711
    },
    "stopped_at": "2025-01-17T12:03:00Z"
  }
}
```

### 5. 배틀 목록 조회

```
GET /api/v1/battles
```

**Query Parameters:**

| Parameter | Type | Default | 설명 |
|-----------|------|---------|------|
| `page` | int | 1 | 페이지 번호 |
| `page_size` | int | 10 | 페이지 크기 |
| `status` | string | - | 상태 필터 |
| `from` | datetime | - | 시작 시간 |
| `to` | datetime | - | 종료 시간 |

**Response (200 OK):**

```json
{
  "success": true,
  "data": {
    "battles": [
      {
        "battle_id": "bat_abc123def456",
        "status": "completed",
        "rounds": 100,
        "detection_rate": 0.75,
        "created_at": "2025-01-17T12:00:00Z"
      }
    ],
    "pagination": { ... }
  }
}
```

### 6. 배틀 통계 조회

```
GET /api/v1/battles/{battle_id}/stats
```

**Response (200 OK):**

```json
{
  "success": true,
  "data": {
    "battle_id": "bat_abc123def456",
    "summary": {
      "total_rounds": 100,
      "detection_count": 75,
      "evasion_count": 25,
      "detection_rate": 0.75,
      "evasion_rate": 0.25
    },
    "by_strategy": {
      "unicode_evasion": {
        "total": 30,
        "detected": 25,
        "detection_rate": 0.833
      },
      "llm_evasion": {
        "total": 50,
        "detected": 35,
        "detection_rate": 0.700
      },
      "homoglyph": {
        "total": 20,
        "detected": 15,
        "detection_rate": 0.750
      }
    },
    "score_distribution": {
      "0.0-0.2": 5,
      "0.2-0.4": 10,
      "0.4-0.6": 20,
      "0.6-0.8": 30,
      "0.8-1.0": 35
    },
    "round_progression": [
      {"round": 10, "cumulative_detection_rate": 0.60},
      {"round": 20, "cumulative_detection_rate": 0.65},
      {"round": 30, "cumulative_detection_rate": 0.70},
      // ...
    ]
  }
}
```

---

## 🤖 Model API

### 1. 모델 목록 조회

```
GET /api/v1/models
```

**Response (200 OK):**

```json
{
  "success": true,
  "data": {
    "models": [
      {
        "name": "content-filter",
        "alias": "champion",
        "version": 3,
        "run_id": "run_xyz789",
        "metrics": {
          "f1": 0.87,
          "precision": 0.85,
          "recall": 0.89,
          "accuracy": 0.92
        },
        "created_at": "2025-01-15T10:00:00Z",
        "is_active": true
      },
      {
        "name": "content-filter",
        "alias": "challenger",
        "version": 4,
        "run_id": "run_abc123",
        "metrics": {
          "f1": 0.89,
          "precision": 0.87,
          "recall": 0.91,
          "accuracy": 0.93
        },
        "created_at": "2025-01-17T08:00:00Z",
        "is_active": false
      }
    ]
  }
}
```

### 2. 특정 모델 조회

```
GET /api/v1/models/{alias}
```

**Path Parameters:**

| Parameter | 설명 |
|-----------|------|
| `alias` | `champion` 또는 `challenger` |

**Response (200 OK):**

```json
{
  "success": true,
  "data": {
    "name": "content-filter",
    "alias": "champion",
    "version": 3,
    "run_id": "run_xyz789",
    "base_model": "bert-base-multilingual-cased",
    "lora_config": {
      "r": 16,
      "lora_alpha": 32,
      "target_modules": ["query", "value"]
    },
    "training_config": {
      "epochs": 3,
      "batch_size": 2,
      "learning_rate": 0.0002,
      "train_samples": 50000,
      "eval_samples": 5000
    },
    "metrics": {
      "f1": 0.87,
      "precision": 0.85,
      "recall": 0.89,
      "accuracy": 0.92,
      "auc_roc": 0.94
    },
    "mlflow_uri": "http://localhost:5000/#/models/content-filter/versions/3",
    "created_at": "2025-01-15T10:00:00Z"
  }
}
```

### 3. 모델 비교

```
GET /api/v1/models/compare
```

**Response (200 OK):**

```json
{
  "success": true,
  "data": {
    "champion": {
      "version": 3,
      "metrics": {
        "f1": 0.87,
        "precision": 0.85,
        "recall": 0.89
      }
    },
    "challenger": {
      "version": 4,
      "metrics": {
        "f1": 0.89,
        "precision": 0.87,
        "recall": 0.91
      }
    },
    "improvement": {
      "f1": 0.02,
      "precision": 0.02,
      "recall": 0.02
    },
    "recommendation": {
      "should_promote": true,
      "reason": "Challenger shows 2.3% improvement in F1 score"
    }
  }
}
```

### 4. Challenger 승격

```
POST /api/v1/models/promote
```

**Request Body:**

```json
{
  "confirm": true,
  "reason": "Improved F1 score by 2.3%"
}
```

**Response (200 OK):**

```json
{
  "success": true,
  "data": {
    "promoted_version": 4,
    "previous_champion_version": 3,
    "new_champion": {
      "alias": "champion",
      "version": 4,
      "metrics": { ... }
    },
    "promoted_at": "2025-01-17T14:00:00Z",
    "inference_reloaded": true
  }
}
```

### 5. 재학습 트리거

```
POST /api/v1/models/retrain
```

**Request Body:**

```json
{
  "reason": "High evasion rate detected",
  "config_override": {          // 선택적 설정 오버라이드
    "epochs": 5,
    "learning_rate": 0.0001
  }
}
```

**Response (202 Accepted):**

```json
{
  "success": true,
  "data": {
    "training_job_id": "train_xyz123",
    "status": "queued",
    "estimated_duration_minutes": 30,
    "created_at": "2025-01-17T14:00:00Z"
  }
}
```

### 6. 학습 상태 조회

```
GET /api/v1/models/training/{job_id}
```

**Response (200 OK):**

```json
{
  "success": true,
  "data": {
    "job_id": "train_xyz123",
    "status": "running",          // queued, running, completed, failed
    "progress": {
      "current_epoch": 2,
      "total_epochs": 3,
      "current_step": 1500,
      "total_steps": 2250,
      "percentage": 66.7
    },
    "current_metrics": {
      "train_loss": 0.35,
      "eval_loss": 0.42,
      "eval_f1": 0.85
    },
    "mlflow_run_id": "run_abc123",
    "started_at": "2025-01-17T14:05:00Z",
    "estimated_completion": "2025-01-17T14:35:00Z"
  }
}
```

---

## 🔍 Inference API (FastAPI)

> ML 추론 전용 서비스 (Python/FastAPI)  
> Base URL: `http://localhost:8001`

### 1. 단일 텍스트 분류

```
POST /classify
```

**Request Body:**

```json
{
  "text": "분류할 텍스트입니다",
  "model_alias": "champion"       // 선택 (기본: champion)
}
```

**Response (200 OK):**

```json
{
  "toxic_score": 0.15,
  "is_toxic": false,
  "confidence": 0.85,
  "model_version": "v3",
  "inference_time_ms": 12.5
}
```

### 2. 배치 분류

```
POST /classify/batch
```

**Request Body:**

```json
{
  "texts": [
    "첫 번째 텍스트",
    "두 번째 텍스트",
    "세 번째 텍스트"
  ],
  "model_alias": "champion"
}
```

**Response (200 OK):**

```json
{
  "results": [
    {
      "text_index": 0,
      "toxic_score": 0.15,
      "is_toxic": false,
      "confidence": 0.85
    },
    {
      "text_index": 1,
      "toxic_score": 0.92,
      "is_toxic": true,
      "confidence": 0.92
    },
    {
      "text_index": 2,
      "toxic_score": 0.05,
      "is_toxic": false,
      "confidence": 0.95
    }
  ],
  "model_version": "v3",
  "total_inference_time_ms": 35.2,
  "avg_inference_time_ms": 11.7
}
```

### 3. Shadow 모드 분류 (Champion + Challenger)

```
POST /classify/shadow
```

**Request Body:**

```json
{
  "text": "비교 분류할 텍스트"
}
```

**Response (200 OK):**

```json
{
  "champion": {
    "toxic_score": 0.45,
    "is_toxic": false,
    "confidence": 0.55,
    "model_version": "v3"
  },
  "challenger": {
    "toxic_score": 0.72,
    "is_toxic": true,
    "confidence": 0.72,
    "model_version": "v4"
  },
  "agreement": false,
  "score_diff": 0.27
}
```

### 4. 모델 핫 리로드

```
POST /reload
```

**Response (200 OK):**

```json
{
  "status": "reloaded",
  "previous_version": "v3",
  "current_version": "v4",
  "reload_time_ms": 1520.3
}
```

### 5. 헬스 체크

```
GET /health
```

**Response (200 OK):**

```json
{
  "status": "healthy",
  "model_loaded": true,
  "model_version": "v3",
  "gpu_available": true,
  "gpu_memory_used_mb": 3200,
  "gpu_memory_total_mb": 8192
}
```

---

## 📊 Metrics API

### 1. 품질 추이 조회

```
GET /api/v1/metrics/quality
```

**Query Parameters:**

| Parameter | Type | Default | 설명 |
|-----------|------|---------|------|
| `from` | datetime | 7일 전 | 시작 시간 |
| `to` | datetime | 현재 | 종료 시간 |
| `interval` | string | `1h` | 집계 간격 (1h, 6h, 1d) |

**Response (200 OK):**

```json
{
  "success": true,
  "data": {
    "time_series": [
      {
        "timestamp": "2025-01-10T00:00:00Z",
        "detection_rate": 0.60,
        "model_version": "v1",
        "battles_count": 5
      },
      {
        "timestamp": "2025-01-11T00:00:00Z",
        "detection_rate": 0.65,
        "model_version": "v2",
        "battles_count": 8
      },
      {
        "timestamp": "2025-01-12T00:00:00Z",
        "detection_rate": 0.72,
        "model_version": "v2",
        "battles_count": 10
      }
    ],
    "summary": {
      "avg_detection_rate": 0.72,
      "max_detection_rate": 0.85,
      "min_detection_rate": 0.60,
      "total_battles": 45,
      "total_rounds": 4500,
      "improvement_percentage": 25.0
    }
  }
}
```

### 2. 모델 성능 히스토리

```
GET /api/v1/metrics/models
```

**Response (200 OK):**

```json
{
  "success": true,
  "data": {
    "versions": [
      {
        "version": 1,
        "created_at": "2025-01-05T00:00:00Z",
        "promoted_at": "2025-01-05T12:00:00Z",
        "demoted_at": "2025-01-10T12:00:00Z",
        "metrics": {
          "f1": 0.75,
          "precision": 0.73,
          "recall": 0.77
        },
        "battle_stats": {
          "battles_served": 20,
          "avg_detection_rate": 0.62
        }
      },
      {
        "version": 2,
        "created_at": "2025-01-10T08:00:00Z",
        "promoted_at": "2025-01-10T12:00:00Z",
        "demoted_at": "2025-01-15T12:00:00Z",
        "metrics": {
          "f1": 0.82,
          "precision": 0.80,
          "recall": 0.84
        },
        "battle_stats": {
          "battles_served": 35,
          "avg_detection_rate": 0.70
        }
      },
      {
        "version": 3,
        "created_at": "2025-01-15T08:00:00Z",
        "promoted_at": "2025-01-15T12:00:00Z",
        "demoted_at": null,
        "metrics": {
          "f1": 0.87,
          "precision": 0.85,
          "recall": 0.89
        },
        "battle_stats": {
          "battles_served": 15,
          "avg_detection_rate": 0.78
        }
      }
    ]
  }
}
```

### 3. Prometheus 메트릭 엔드포인트

```
GET /api/v1/metrics/prometheus
```

**Response (200 OK, text/plain):**

```prometheus
# HELP content_arena_battles_total Total number of battles
# TYPE content_arena_battles_total counter
content_arena_battles_total{status="completed"} 45
content_arena_battles_total{status="failed"} 2

# HELP content_arena_detection_rate Current detection rate
# TYPE content_arena_detection_rate gauge
content_arena_detection_rate 0.78

# HELP content_arena_model_f1_score Model F1 score
# TYPE content_arena_model_f1_score gauge
content_arena_model_f1_score{alias="champion"} 0.87
content_arena_model_f1_score{alias="challenger"} 0.89

# HELP content_arena_inference_latency_seconds Inference latency
# TYPE content_arena_inference_latency_seconds histogram
content_arena_inference_latency_seconds_bucket{le="0.01"} 100
content_arena_inference_latency_seconds_bucket{le="0.025"} 450
content_arena_inference_latency_seconds_bucket{le="0.05"} 890
content_arena_inference_latency_seconds_bucket{le="0.1"} 990
content_arena_inference_latency_seconds_bucket{le="+Inf"} 1000
content_arena_inference_latency_seconds_sum 35.5
content_arena_inference_latency_seconds_count 1000
```

---

## 🔧 시스템 API

### 1. 헬스 체크

```
GET /health
```

**Response (200 OK):**

```json
{
  "status": "healthy",
  "version": "1.0.0",
  "uptime_seconds": 86400,
  "dependencies": {
    "database": "healthy",
    "redis": "healthy",
    "mlflow": "healthy",
    "inference_service": "healthy",
    "ollama": "healthy"
  }
}
```

### 2. 레디니스 체크

```
GET /ready
```

**Response (200 OK):**

```json
{
  "ready": true,
  "checks": {
    "database_connection": true,
    "redis_connection": true,
    "model_loaded": true,
    "ollama_available": true
  }
}
```

---

## 📝 Go API 구현 예시

```go
// internal/handler/battle_handler.go
package handler

import (
    "net/http"
    "github.com/gin-gonic/gin"
    "content-arena/internal/service"
)

type BattleHandler struct {
    battleService *service.BattleService
}

func NewBattleHandler(bs *service.BattleService) *BattleHandler {
    return &BattleHandler{battleService: bs}
}

// CreateBattle godoc
// @Summary Create a new battle
// @Description Start a new battle session between attacker and defender
// @Tags battles
// @Accept json
// @Produce json
// @Param request body CreateBattleRequest true "Battle configuration"
// @Success 201 {object} Response{data=Battle}
// @Failure 400 {object} ErrorResponse
// @Router /api/v1/battles [post]
func (h *BattleHandler) CreateBattle(c *gin.Context) {
    var req CreateBattleRequest
    if err := c.ShouldBindJSON(&req); err != nil {
        c.JSON(http.StatusBadRequest, ErrorResponse{
            Success: false,
            Error: ErrorDetail{
                Code:    "VALIDATION_ERROR",
                Message: err.Error(),
            },
        })
        return
    }
    
    battle, err := h.battleService.CreateBattle(c.Request.Context(), req)
    if err != nil {
        c.JSON(http.StatusInternalServerError, ErrorResponse{
            Success: false,
            Error: ErrorDetail{
                Code:    "INTERNAL_ERROR",
                Message: err.Error(),
            },
        })
        return
    }
    
    c.JSON(http.StatusCreated, Response{
        Success: true,
        Data:    battle,
    })
}

// GetBattle godoc
// @Summary Get battle by ID
// @Description Get battle status and statistics
// @Tags battles
// @Produce json
// @Param battle_id path string true "Battle ID"
// @Success 200 {object} Response{data=Battle}
// @Failure 404 {object} ErrorResponse
// @Router /api/v1/battles/{battle_id} [get]
func (h *BattleHandler) GetBattle(c *gin.Context) {
    battleID := c.Param("battle_id")
    
    battle, err := h.battleService.GetBattle(c.Request.Context(), battleID)
    if err != nil {
        c.JSON(http.StatusNotFound, ErrorResponse{
            Success: false,
            Error: ErrorDetail{
                Code:    "BATTLE_NOT_FOUND",
                Message: "Battle not found",
            },
        })
        return
    }
    
    c.JSON(http.StatusOK, Response{
        Success: true,
        Data:    battle,
    })
}
```

---

## 📝 Python Inference API 구현 예시

```python
# ml-service/app/main.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Optional
import time

app = FastAPI(
    title="Content Filter Inference API",
    description="콘텐츠 필터링 ML 추론 서비스",
    version="1.0.0"
)

class ClassifyRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=5000)
    model_alias: Optional[str] = Field(default="champion")

class ClassifyResponse(BaseModel):
    toxic_score: float
    is_toxic: bool
    confidence: float
    model_version: str
    inference_time_ms: float

@app.post("/classify", response_model=ClassifyResponse)
async def classify(request: ClassifyRequest):
    start_time = time.time()
    
    result = model_service.classify(
        text=request.text,
        alias=request.model_alias
    )
    
    inference_time = (time.time() - start_time) * 1000
    
    return ClassifyResponse(
        toxic_score=result["toxic_score"],
        is_toxic=result["is_toxic"],
        confidence=result["confidence"],
        model_version=result["model_version"],
        inference_time_ms=round(inference_time, 2)
    )
```

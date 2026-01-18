# API 명세

> Battle API, Model API, Inference API 스펙

## 서비스 구조

| 서비스 | 포트 | 설명 |
|--------|------|------|
| API Gateway (Go/Gin) | 8080 | Battle, Model, Metrics API |
| Inference API (FastAPI) | 8001 | ML 추론 |
| Ollama | 11434 | 로컬 LLM |
| MLflow | 5000 | 모델 레지스트리 |

## 공통 응답 형식

```json
// 성공
{"success": true, "data": {...}, "meta": {"timestamp": "...", "request_id": "..."}}

// 에러
{"success": false, "error": {"code": "...", "message": "..."}, "meta": {...}}
```

### 에러 코드

| HTTP | Code | 설명 |
|------|------|------|
| 400 | `INVALID_REQUEST` | 잘못된 요청 |
| 401 | `UNAUTHORIZED` | 인증 필요 |
| 404 | `NOT_FOUND` | 리소스 없음 |
| 429 | `RATE_LIMITED` | 요청 한도 초과 |
| 500 | `INTERNAL_ERROR` | 서버 오류 |

## Battle API

### 엔드포인트 요약

| Method | Path | 설명 |
|--------|------|------|
| `POST` | `/api/v1/battles` | 배틀 생성 |
| `GET` | `/api/v1/battles` | 배틀 목록 |
| `GET` | `/api/v1/battles/:id` | 배틀 상태 |
| `GET` | `/api/v1/battles/:id/rounds` | 라운드 목록 |
| `GET` | `/api/v1/battles/:id/stats` | 배틀 통계 |
| `POST` | `/api/v1/battles/:id/stop` | 배틀 중지 |

### 배틀 생성

```http
POST /api/v1/battles
Content-Type: application/json

{
  "rounds": 100,
  "attack_strategy": "mixed",
  "defender_alias": "champion",
  "async": true
}
```

**공격 전략**: `unicode_evasion`, `homoglyph`, `leetspeak`, `llm_evasion`, `adversarial_llm`, `mixed`

### 배틀 상태 응답

```json
{
  "battle_id": "bat_abc123",
  "status": "running",
  "progress": {"completed_rounds": 45, "total_rounds": 100},
  "current_stats": {"detection_rate": 0.711, "evasion_rate": 0.289}
}
```

**상태**: `pending`, `running`, `completed`, `failed`

## Model API

### 엔드포인트 요약

| Method | Path | 설명 |
|--------|------|------|
| `GET` | `/api/v1/models` | 모델 목록 |
| `GET` | `/api/v1/models/:alias` | 모델 상세 |
| `GET` | `/api/v1/models/compare` | Champion vs Challenger |
| `POST` | `/api/v1/models/promote` | Challenger 승격 |

### 모델 목록 응답

```json
{
  "models": [
    {"alias": "champion", "version": 3, "metrics": {"f1": 0.87}},
    {"alias": "challenger", "version": 4, "metrics": {"f1": 0.89}}
  ]
}
```

### 모델 승격

```http
POST /api/v1/models/promote
Content-Type: application/json

{"confirm": true, "reason": "Improved F1 by 2.3%"}
```

## Inference API (ML Service)

### 엔드포인트 요약

| Method | Path | 설명 |
|--------|------|------|
| `POST` | `/classify` | 단일 텍스트 분류 |
| `POST` | `/classify/batch` | 배치 분류 |
| `POST` | `/reload` | 모델 핫 리로드 |
| `GET` | `/health` | 헬스 체크 |
| `GET` | `/metrics` | Prometheus 메트릭 |

### 텍스트 분류

```http
POST /classify
Content-Type: application/json

{"text": "분류할 텍스트", "model_alias": "champion"}
```

**응답**:
```json
{"toxic_score": 0.85, "is_toxic": true, "confidence": 0.85, "model_version": "v3"}
```

### 배치 분류

```http
POST /classify/batch
Content-Type: application/json

{"texts": ["텍스트1", "텍스트2"], "model_alias": "champion"}
```

### 모델 리로드

```http
POST /reload
```

Champion 모델이 변경되었을 때 서비스 재시작 없이 새 모델 로드.

## Metrics API

### 엔드포인트 요약

| Method | Path | 설명 |
|--------|------|------|
| `GET` | `/api/v1/metrics` | Prometheus 메트릭 |
| `GET` | `/api/v1/metrics/quality` | 품질 추이 |
| `GET` | `/api/v1/metrics/battles` | 배틀 통계 |

### 품질 추이 응답

```json
{
  "period": "7d",
  "data_points": [
    {"date": "2025-01-11", "detection_rate": 0.65},
    {"date": "2025-01-17", "detection_rate": 0.75}
  ]
}
```

## 시스템 API

| Method | Path | 설명 |
|--------|------|------|
| `GET` | `/health` | 헬스 체크 |
| `GET` | `/ready` | 레디니스 체크 |

### 헬스 체크 응답

```json
{
  "status": "healthy",
  "components": {
    "database": "ok",
    "redis": "ok",
    "ollama": "ok",
    "ml_service": "ok"
  }
}
```

## 인증 (선택)

### API Key 인증

```http
GET /api/v1/battles
X-API-Key: your-api-key
```

### JWT 인증

```http
GET /api/v1/battles
Authorization: Bearer eyJhbGciOiJIUzI1NiIs...
```

상세: `dev-06-security.md`

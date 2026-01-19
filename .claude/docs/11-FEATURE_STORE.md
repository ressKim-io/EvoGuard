# Feature Store 설계

> 경량 Feature Store 자체 구현으로 ML 파이프라인의 feature 관리 체계화

## 개요

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Data Sources  │    │ Feature Compute │    │  Feature Stores │
│  (Battle/Round) │───►│  (Transformers) │───►│ Offline/Online  │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                                       │
┌─────────────────┐                                    │
│ Feature Registry│◄───────────────────────────────────┘
│  (PostgreSQL)   │◄───► MLflow Integration
└─────────────────┘
```

**관련 문서**:
- `05-MLOPS.md` - MLOps 파이프라인
- `py-05-mlflow.md` - MLflow 상세
- `04-ML_PIPELINE.md` - ML 파이프라인

## 핵심 컴포넌트

| 컴포넌트 | 저장소 | 역할 |
|----------|--------|------|
| **Feature Registry** | PostgreSQL | 메타데이터, 버전, 리니지 관리 |
| **Offline Store** | Parquet + DuckDB | 학습용 대용량 데이터 |
| **Online Store** | Redis | 실시간 서빙용 캐시 |
| **Feature Compute** | Python | Feature 계산 로직 |

---

## 1. Feature Registry (PostgreSQL)

### 역할

- Feature 정의 메타데이터 저장
- Feature Group 관리
- 버전 히스토리 추적
- 모델-Feature 리니지

### DB 스키마

#### feature_definitions

```sql
CREATE TABLE feature_definitions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name VARCHAR(100) UNIQUE NOT NULL,
    data_type VARCHAR(20) NOT NULL,  -- int, float, string, embedding
    entity_type VARCHAR(50) NOT NULL, -- text, battle, user
    source_type VARCHAR(20) NOT NULL, -- computed, raw, aggregated
    computation_config JSONB,          -- transformer config
    version INT DEFAULT 1,
    is_active BOOLEAN DEFAULT true,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_feature_definitions_entity ON feature_definitions(entity_type);
CREATE INDEX idx_feature_definitions_active ON feature_definitions(is_active);
```

#### feature_groups

```sql
CREATE TABLE feature_groups (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name VARCHAR(100) UNIQUE NOT NULL,
    description TEXT,
    entity_type VARCHAR(50) NOT NULL,
    version INT DEFAULT 1,
    schema_hash VARCHAR(64),          -- feature 구성 해시
    is_active BOOLEAN DEFAULT true,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Feature Group과 Feature Definition 매핑
CREATE TABLE feature_group_features (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    feature_group_id UUID REFERENCES feature_groups(id) ON DELETE CASCADE,
    feature_definition_id UUID REFERENCES feature_definitions(id) ON DELETE CASCADE,
    UNIQUE(feature_group_id, feature_definition_id)
);
```

#### model_feature_lineage

```sql
CREATE TABLE model_feature_lineage (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    mlflow_run_id VARCHAR(100) NOT NULL,
    mlflow_model_name VARCHAR(100),
    mlflow_model_version INT,
    feature_group_id UUID REFERENCES feature_groups(id),
    feature_group_version INT NOT NULL,
    feature_schema_snapshot JSONB NOT NULL, -- 학습 시점 스키마 스냅샷
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_lineage_run ON model_feature_lineage(mlflow_run_id);
CREATE INDEX idx_lineage_model ON model_feature_lineage(mlflow_model_name, mlflow_model_version);
```

---

## 2. Offline Store (Parquet + DuckDB)

### 역할

- 학습용 대용량 데이터 저장
- Point-in-time correct joins
- DVC로 버전 관리

### 저장 구조

```
data/features/
├── text_features/
│   ├── v1/
│   │   ├── 2024-01-01.parquet
│   │   ├── 2024-01-02.parquet
│   │   └── ...
│   └── v2/
│       └── ...
├── battle_features/
│   └── v1/
│       └── ...
└── _metadata/
    └── feature_groups.json
```

### Parquet 스키마

```python
# Text Features Parquet 스키마
text_features_schema = pa.schema([
    pa.field("entity_id", pa.string()),          # UUID
    pa.field("event_timestamp", pa.timestamp("us", tz="UTC")),
    pa.field("text_length", pa.int32()),
    pa.field("word_count", pa.int32()),
    pa.field("unicode_ratio", pa.float32()),
    pa.field("special_char_ratio", pa.float32()),
    pa.field("repeated_char_ratio", pa.float32()),
    pa.field("created_at", pa.timestamp("us", tz="UTC")),
])
```

### Point-in-Time Correct Join

```python
def point_in_time_join(
    entity_df: pd.DataFrame,  # entity_id, event_timestamp
    feature_df: pd.DataFrame, # entity_id, event_timestamp, features...
) -> pd.DataFrame:
    """
    각 entity의 event_timestamp 이전의 최신 feature 조회.
    미래 데이터 누수 방지.
    """
    # DuckDB로 효율적인 ASOF JOIN
    return duckdb.query("""
        SELECT e.*, f.* EXCLUDE (entity_id, event_timestamp)
        FROM entity_df e
        ASOF JOIN feature_df f
        ON e.entity_id = f.entity_id
        AND e.event_timestamp >= f.event_timestamp
    """).df()
```

### DVC 통합

```yaml
# data/features/.dvc
outs:
  - md5: abc123...
    path: text_features/v1/
    size: 1048576
```

```bash
# Feature 데이터 버전 관리
dvc add data/features/text_features/v1/
dvc push
```

---

## 3. Online Store (Redis)

### 역할

- 실시간 서빙용 캐시
- TTL 기반 만료
- 배치 조회 지원

### 키 구조

```
feature:{entity_type}:{entity_id}:{feature_group}:{version}
```

**예시**:
```
feature:text:550e8400-e29b-41d4-a716-446655440000:text_features:v1
```

### 데이터 형식

```python
# Hash 타입 저장
{
    "text_length": "150",
    "word_count": "25",
    "unicode_ratio": "0.15",
    "special_char_ratio": "0.02",
    "repeated_char_ratio": "0.01",
    "_updated_at": "2024-01-15T10:30:00Z"
}
```

### TTL 정책

| Feature Group | TTL | 이유 |
|---------------|-----|------|
| text_features | 24h | 텍스트 분석 결과 캐시 |
| battle_features | 1h | 빈번한 업데이트 |
| user_features | 6h | 사용자 통계 캐시 |

### 배치 조회

```python
async def get_features_batch(
    entity_ids: list[str],
    feature_group: str,
) -> dict[str, dict]:
    """Pipeline으로 배치 조회"""
    pipe = redis.pipeline()
    for entity_id in entity_ids:
        key = f"feature:text:{entity_id}:{feature_group}:v1"
        pipe.hgetall(key)
    results = await pipe.execute()
    return {eid: r for eid, r in zip(entity_ids, results)}
```

---

## 4. Feature Compute

### 추상 베이스 클래스

```python
from abc import ABC, abstractmethod
from typing import Any

class FeatureTransformer(ABC):
    """Feature 계산 추상 클래스"""

    @property
    @abstractmethod
    def feature_names(self) -> list[str]:
        """계산되는 feature 이름 목록"""
        pass

    @property
    @abstractmethod
    def entity_type(self) -> str:
        """엔티티 타입 (text, battle, user)"""
        pass

    @abstractmethod
    def transform(self, data: Any) -> dict[str, Any]:
        """단일 데이터에서 feature 추출"""
        pass

    def transform_batch(self, data_list: list[Any]) -> list[dict[str, Any]]:
        """배치 처리 (기본 구현)"""
        return [self.transform(d) for d in data_list]
```

### TextFeatureTransformer

```python
import re
from collections import Counter

class TextFeatureTransformer(FeatureTransformer):
    """텍스트에서 특성 추출"""

    @property
    def feature_names(self) -> list[str]:
        return [
            "text_length",
            "word_count",
            "unicode_ratio",
            "special_char_ratio",
            "repeated_char_ratio",
        ]

    @property
    def entity_type(self) -> str:
        return "text"

    def transform(self, text: str) -> dict[str, Any]:
        if not text:
            return {name: 0 for name in self.feature_names}

        length = len(text)
        words = text.split()

        # 비-ASCII 문자 비율
        non_ascii = sum(1 for c in text if ord(c) > 127)
        unicode_ratio = non_ascii / length if length > 0 else 0

        # 특수문자 비율
        special = sum(1 for c in text if not c.isalnum() and not c.isspace())
        special_ratio = special / length if length > 0 else 0

        # 연속 반복 문자 비율
        repeated = self._count_repeated_chars(text)
        repeated_ratio = repeated / length if length > 0 else 0

        return {
            "text_length": length,
            "word_count": len(words),
            "unicode_ratio": round(unicode_ratio, 4),
            "special_char_ratio": round(special_ratio, 4),
            "repeated_char_ratio": round(repeated_ratio, 4),
        }

    def _count_repeated_chars(self, text: str) -> int:
        """연속 3회 이상 반복되는 문자 수"""
        count = 0
        i = 0
        while i < len(text):
            j = i
            while j < len(text) and text[j] == text[i]:
                j += 1
            if j - i >= 3:
                count += j - i
            i = j
        return count
```

### BattleFeatureTransformer

```python
class BattleFeatureTransformer(FeatureTransformer):
    """배틀 통계에서 특성 추출"""

    @property
    def feature_names(self) -> list[str]:
        return [
            "detection_rate",
            "evasion_rate",
            "avg_confidence",
            "round_count",
            "success_streak",
        ]

    @property
    def entity_type(self) -> str:
        return "battle"

    def transform(self, battle: dict) -> dict[str, Any]:
        rounds = battle.get("rounds", [])
        if not rounds:
            return {name: 0.0 for name in self.feature_names}

        total = len(rounds)
        detected = sum(1 for r in rounds if r.get("detected"))
        evaded = sum(1 for r in rounds if r.get("evaded"))
        confidences = [r.get("confidence", 0) for r in rounds]

        return {
            "detection_rate": round(detected / total, 4),
            "evasion_rate": round(evaded / total, 4),
            "avg_confidence": round(sum(confidences) / total, 4),
            "round_count": total,
            "success_streak": self._max_streak(rounds),
        }

    def _max_streak(self, rounds: list[dict]) -> int:
        """최대 연속 성공 횟수"""
        max_streak = current = 0
        for r in rounds:
            if r.get("detected"):
                current += 1
                max_streak = max(max_streak, current)
            else:
                current = 0
        return max_streak
```

---

## 5. MLflow 통합

### Feature 로깅

```python
def log_feature_metadata(
    run_id: str,
    feature_group: str,
    feature_group_version: int,
    feature_schema: dict,
):
    """MLflow run에 feature 메타데이터 로깅"""
    with mlflow.start_run(run_id=run_id):
        mlflow.log_param("feature_group", feature_group)
        mlflow.log_param("feature_group_version", feature_group_version)
        mlflow.log_dict(feature_schema, "feature_schema.json")
```

### 리니지 추적

```python
def record_model_lineage(
    mlflow_run_id: str,
    model_name: str,
    model_version: int,
    feature_group_id: UUID,
    feature_group_version: int,
) -> None:
    """모델과 Feature Group 간 리니지 기록"""
    # Feature Group의 현재 스키마 스냅샷 저장
    feature_group = get_feature_group(feature_group_id)
    schema_snapshot = {
        "features": [f.dict() for f in feature_group.features],
        "version": feature_group_version,
        "captured_at": datetime.utcnow().isoformat(),
    }

    # DB에 리니지 기록
    lineage = ModelFeatureLineage(
        mlflow_run_id=mlflow_run_id,
        mlflow_model_name=model_name,
        mlflow_model_version=model_version,
        feature_group_id=feature_group_id,
        feature_group_version=feature_group_version,
        feature_schema_snapshot=schema_snapshot,
    )
    db.add(lineage)
    db.commit()
```

### 스키마 검증

```python
def validate_serving_features(
    model_name: str,
    model_version: int,
    current_features: dict,
) -> bool:
    """서빙 시 feature 스키마 일치 검증"""
    lineage = get_model_lineage(model_name, model_version)
    expected_schema = lineage.feature_schema_snapshot

    for feature in expected_schema["features"]:
        name = feature["name"]
        if name not in current_features:
            raise FeatureMissingError(f"Missing feature: {name}")

        expected_type = feature["data_type"]
        actual_type = type(current_features[name]).__name__
        if not _types_compatible(expected_type, actual_type):
            raise FeatureTypeMismatchError(
                f"Feature {name}: expected {expected_type}, got {actual_type}"
            )

    return True
```

---

## 6. API 인터페이스

### REST API 엔드포인트

| Method | Path | 설명 |
|--------|------|------|
| GET | `/features/groups` | Feature Group 목록 |
| GET | `/features/groups/{name}` | Feature Group 상세 |
| POST | `/features/groups` | Feature Group 생성 |
| GET | `/features/definitions` | Feature 정의 목록 |
| POST | `/features/definitions` | Feature 정의 생성 |
| POST | `/features/compute` | Feature 계산 요청 |
| GET | `/features/online/{entity_id}` | Online Store 조회 |
| POST | `/features/online/batch` | 배치 조회 |
| GET | `/features/lineage/{run_id}` | 모델 리니지 조회 |

### Pydantic 스키마

```python
from pydantic import BaseModel
from datetime import datetime
from uuid import UUID

class FeatureDefinitionCreate(BaseModel):
    name: str
    data_type: str  # int, float, string, embedding
    entity_type: str
    source_type: str
    computation_config: dict | None = None

class FeatureDefinitionResponse(BaseModel):
    id: UUID
    name: str
    data_type: str
    entity_type: str
    source_type: str
    computation_config: dict | None
    version: int
    is_active: bool
    created_at: datetime

class FeatureGroupCreate(BaseModel):
    name: str
    description: str | None
    entity_type: str
    feature_names: list[str]

class FeatureGroupResponse(BaseModel):
    id: UUID
    name: str
    description: str | None
    entity_type: str
    version: int
    schema_hash: str
    features: list[FeatureDefinitionResponse]
    is_active: bool

class ComputeFeaturesRequest(BaseModel):
    feature_group: str
    entity_ids: list[str]
    data: list[dict]  # 원본 데이터

class ComputeFeaturesResponse(BaseModel):
    feature_group: str
    features: list[dict]  # 계산된 feature 값들
    computed_at: datetime

class OnlineFeaturesRequest(BaseModel):
    feature_group: str
    entity_ids: list[str]

class OnlineFeaturesResponse(BaseModel):
    features: dict[str, dict]  # entity_id -> features
    cache_hits: int
    cache_misses: int
```

---

## 7. 모듈 구조

```
ml-service/src/ml_service/feature_store/
├── __init__.py
├── registry/                    # PostgreSQL 메타데이터
│   ├── __init__.py
│   ├── models.py               # SQLAlchemy ORM 모델
│   ├── schemas.py              # Pydantic 스키마
│   └── repository.py           # CRUD 연산
├── offline/                     # Parquet + DuckDB
│   ├── __init__.py
│   ├── writer.py               # Parquet 저장
│   ├── reader.py               # DuckDB 조회
│   └── pit_join.py             # Point-in-time Join
├── online/                      # Redis
│   ├── __init__.py
│   ├── redis_store.py          # Redis 클라이언트
│   └── sync.py                 # Offline → Online 동기화
├── compute/                     # Feature 계산
│   ├── __init__.py
│   ├── base.py                 # FeatureTransformer ABC
│   ├── text_features.py        # TextFeatureTransformer
│   └── battle_features.py      # BattleFeatureTransformer
├── mlflow_integration/          # MLflow 연동
│   ├── __init__.py
│   ├── logger.py               # Feature 메타데이터 로깅
│   └── lineage.py              # 모델-Feature 리니지
└── api/                         # FastAPI 라우터
    ├── __init__.py
    ├── routes.py               # API 엔드포인트
    └── dependencies.py         # 의존성 주입
```

---

## 8. EvoGuard 특화 Feature

### Text Features

| Name | Type | Description | 계산 |
|------|------|-------------|------|
| text_length | int | 전체 문자 수 | `len(text)` |
| word_count | int | 단어 수 | `len(text.split())` |
| unicode_ratio | float | 비-ASCII 문자 비율 | 특수 문자/전체 |
| special_char_ratio | float | 특수문자 비율 | 특수문자/전체 |
| repeated_char_ratio | float | 연속 반복 문자 비율 | 3회 이상 연속/전체 |

### Battle Features

| Name | Type | Description | 계산 |
|------|------|-------------|------|
| detection_rate | float | 탐지 성공률 | 탐지 성공/전체 라운드 |
| evasion_rate | float | 우회 성공률 | 우회 성공/전체 라운드 |
| avg_confidence | float | 평균 신뢰도 | 신뢰도 평균 |
| round_count | int | 총 라운드 수 | 라운드 개수 |
| success_streak | int | 최대 연속 성공 | 연속 탐지 성공 최대 |

### 향후 확장 Feature

| Name | Type | Description |
|------|------|-------------|
| embedding_vector | float[768] | BERT 임베딩 |
| toxicity_score | float | 독성 점수 |
| sentiment_score | float | 감성 점수 |
| language_code | string | 언어 코드 |

---

## 9. 구현 로드맵

### Phase 1: 기반 구축 (MVP) ✅
- [x] Feature Registry 스키마 및 마이그레이션
- [x] 기본 Pydantic 스키마 정의
- [x] FeatureTransformer 베이스 클래스
- [x] TextFeatureTransformer 구현

### Phase 2: Offline Store ✅
- [x] Parquet Writer 구현
- [x] DuckDB Reader 구현
- [x] Point-in-time Join 구현
- [ ] DVC 통합

### Phase 3: Online Store ✅
- [x] Redis Store 구현
- [x] Offline → Online 동기화 (FeatureSync)
- [x] TTL 정책 적용
- [x] 배치 조회 최적화

### Phase 4: MLflow 통합 ✅
- [x] Feature 메타데이터 로깅 (FeatureLogger)
- [x] 모델-Feature 리니지 추적 (FeatureLineageTracker)
- [x] 스키마 검증 로직 (FeatureSchemaValidator)

### Phase 5: API 및 배포
- [x] FastAPI 라우터 구현
- [x] API 문서화 (OpenAPI)
- [ ] 통합 테스트
- [ ] 배포

### 추가 구현 ✅
- [x] BattleFeatureTransformer 구현

---

## 10. 참고 자료

### 외부 리소스
- [Feast Feature Store](https://feast.dev/) - 참고 아키텍처
- [Tecton Feature Store Concepts](https://www.tecton.ai/feature-store/)
- [DuckDB Documentation](https://duckdb.org/docs/)

### 내부 문서
- `05-MLOPS.md` - MLOps 파이프라인
- `py-05-mlflow.md` - MLflow 상세
- `04-ML_PIPELINE.md` - ML 파이프라인

---

*설계 완료: 2026-01-19*
*MVP 구현 완료: 2026-01-19*
*전체 구현 완료: 2026-01-19*

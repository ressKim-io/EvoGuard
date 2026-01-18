# 테스트 전략 가이드

> EvoGuard 프로젝트의 테스트 전략 및 가이드라인

## 테스트 피라미드

```
        /\
       /  \
      / E2E \     10% - 주요 시나리오
     /--------\
    /Integration\   20% - 서비스 간 통신
   /--------------\
  /     Unit       \  70% - 비즈니스 로직
 /------------------\
```

| 레벨 | 비율 | 목적 | 실행 속도 |
|------|------|------|-----------|
| Unit | 70% | 개별 함수/클래스 로직 검증 | 빠름 (ms) |
| Integration | 20% | 컴포넌트 간 상호작용 | 보통 (sec) |
| E2E | 10% | 전체 시스템 플로우 | 느림 (min) |

---

## 언어별 테스트 도구

### Python (pytest)

```bash
# 테스트 실행
uv run pytest

# 특정 파일/폴더 실행
uv run pytest tests/test_unicode_evasion.py

# 마커로 필터링
uv run pytest -m "not slow"

# 커버리지 포함
uv run pytest --cov=. --cov-report=xml --cov-report=term
```

#### 설정 파일: `pyproject.toml`

```toml
[tool.pytest.ini_options]
testpaths = ["tests"]
python_files = ["test_*.py"]
python_functions = ["test_*"]
asyncio_mode = "auto"
addopts = ["-v", "--tb=short"]
markers = [
    "slow: marks tests as slow",
    "integration: marks tests requiring external services",
]
```

### Go (go test)

```bash
# 테스트 실행
go test ./...

# 상세 출력
go test -v ./...

# 커버리지 포함
go test -coverprofile=coverage.out ./...

# Race condition 검사
go test -race ./...
```

---

## 커버리지 목표

| 서비스 | 목표 | 비고 |
|--------|------|------|
| attacker | 60% | 핵심 evasion 로직 |
| defender | 60% | 탐지 로직 |
| ml-service | 60% | 모델 서빙 |
| training | 60% | 학습 파이프라인 |
| api-service (Go) | 60% | API 핸들러 |

### CI 강제 설정

- **Python**: `--cov-fail-under=60`
- **Go**: 커버리지 60% 미만 시 스크립트로 실패 처리

---

## 테스트 디렉토리 구조

### Python 서비스

```
attacker/
├── pyproject.toml
├── strategies/
│   ├── __init__.py
│   ├── base.py
│   ├── leetspeak.py
│   └── unicode_evasion.py
└── tests/
    ├── __init__.py
    ├── conftest.py          # 공유 fixtures
    ├── test_leetspeak.py
    ├── test_unicode_evasion.py
    └── integration/
        └── test_api.py
```

### Go 서비스

```
api-service/
├── go.mod
├── main.go
├── handlers/
│   ├── battle.go
│   └── battle_test.go      # 같은 폴더에 _test.go
├── services/
│   ├── moderation.go
│   └── moderation_test.go
└── integration/
    └── api_test.go
```

---

## Pytest Markers 사용법

### 마커 정의

```python
# pyproject.toml 또는 conftest.py
markers = [
    "slow: marks tests as slow (> 1s)",
    "integration: requires external services (DB, API)",
    "e2e: end-to-end tests",
    "gpu: requires GPU",
]
```

### 마커 적용

```python
import pytest

# 단일 마커
@pytest.mark.slow
def test_heavy_computation():
    ...

# 다중 마커
@pytest.mark.integration
@pytest.mark.slow
def test_external_api():
    ...

# 클래스 레벨
@pytest.mark.integration
class TestDatabaseOperations:
    def test_insert(self):
        ...
```

### 마커로 실행 제어

```bash
# slow 마커 제외
uv run pytest -m "not slow"

# integration만 실행
uv run pytest -m integration

# 조합
uv run pytest -m "integration and not slow"
```

---

## Fixtures 가이드 (conftest.py)

### 기본 구조

```python
# tests/conftest.py
import pytest
from pathlib import Path

@pytest.fixture
def sample_text():
    """테스트용 샘플 텍스트"""
    return "This is a sample text for testing"

@pytest.fixture
def temp_dir(tmp_path):
    """임시 디렉토리 생성"""
    test_dir = tmp_path / "test_data"
    test_dir.mkdir()
    return test_dir

@pytest.fixture(scope="module")
def expensive_resource():
    """모듈 레벨 공유 리소스"""
    resource = create_expensive_resource()
    yield resource
    resource.cleanup()
```

### Scope 옵션

| Scope | 설명 | 사용 예시 |
|-------|------|-----------|
| function | 각 테스트마다 (기본값) | 독립적인 상태 필요 시 |
| class | 클래스당 한 번 | 클래스 내 테스트 공유 |
| module | 파일당 한 번 | DB 연결 등 |
| session | 전체 세션에 한 번 | 비용이 큰 초기화 |

---

## E2E 테스트 시나리오

### 1. Battle Flow (대결 시나리오)

```
┌─────────┐     ┌─────────┐     ┌─────────┐
│ Attacker│────>│   API   │────>│ Defender│
│  Agent  │<────│ Service │<────│   LLM   │
└─────────┘     └─────────┘     └─────────┘
```

**테스트 단계:**
1. Attacker가 evasion 전략 적용
2. API로 콘텐츠 전송
3. Defender가 탐지 시도
4. 결과 기록 및 점수 계산

```python
@pytest.mark.e2e
async def test_battle_flow():
    # Given
    attacker = AttackerAgent(strategy="leetspeak")
    original = "harmful content"

    # When
    evaded = await attacker.evade(original)
    result = await api_client.moderate(evaded)

    # Then
    assert result.detected in [True, False]
    assert result.confidence >= 0.0
```

### 2. Auto Retrain (자동 재학습)

**테스트 단계:**
1. 새로운 evasion 패턴 수집
2. 데이터셋 업데이트
3. 모델 재학습 트리거
4. 성능 검증 후 배포

```python
@pytest.mark.e2e
@pytest.mark.slow
async def test_auto_retrain_pipeline():
    # Given
    new_samples = collect_failed_detections(n=100)

    # When
    await training_pipeline.update_dataset(new_samples)
    new_model = await training_pipeline.retrain()

    # Then
    metrics = await evaluate(new_model, test_set)
    assert metrics.f1_score > 0.85
```

### 3. Champion/Challenger 배포

**테스트 단계:**
1. 새 모델(Challenger)을 일부 트래픽에 배포
2. Champion vs Challenger 성능 비교
3. 기준 충족 시 Challenger 승격

```python
@pytest.mark.e2e
async def test_champion_challenger():
    # Given
    champion = load_model("production")
    challenger = load_model("candidate")

    # When - 트래픽 분할 테스트
    results = await ab_test(
        champion=champion,
        challenger=challenger,
        traffic_split=0.1,
        duration_minutes=60
    )

    # Then
    assert results.challenger.latency_p99 < results.champion.latency_p99 * 1.1
    assert results.challenger.accuracy >= results.champion.accuracy
```

---

## Mocking 가이드

### Python (pytest-mock)

```python
def test_api_call(mocker):
    # Mock 설정
    mock_response = {"status": "ok"}
    mocker.patch(
        "httpx.AsyncClient.post",
        return_value=AsyncMock(json=lambda: mock_response)
    )

    # 테스트 실행
    result = await client.send_request()

    assert result == mock_response
```

### Go (testify/mock)

```go
type MockService struct {
    mock.Mock
}

func (m *MockService) Process(input string) (string, error) {
    args := m.Called(input)
    return args.String(0), args.Error(1)
}

func TestHandler(t *testing.T) {
    mockSvc := new(MockService)
    mockSvc.On("Process", "test").Return("result", nil)

    handler := NewHandler(mockSvc)
    result := handler.Handle("test")

    assert.Equal(t, "result", result)
    mockSvc.AssertExpectations(t)
}
```

---

## CI/CD 연동

### PR 테스트 워크플로우

```yaml
# .github/workflows/pr-test.yml
- name: Run pytest
  run: uv run pytest --cov=. --cov-report=xml --cov-fail-under=60

- name: Upload coverage
  uses: codecov/codecov-action@v4
```

### 커버리지 리포트

- **Codecov**: PR 코멘트로 커버리지 변화 표시
- **프로젝트 목표**: 60%
- **Patch 목표**: 70% (새로 추가된 코드)

---

## 테스트 작성 Best Practices

### 1. AAA 패턴

```python
def test_example():
    # Arrange - 준비
    input_data = prepare_data()

    # Act - 실행
    result = function_under_test(input_data)

    # Assert - 검증
    assert result == expected
```

### 2. 테스트 이름 규칙

```python
# Good - 무엇을 테스트하는지 명확
def test_leetspeak_converts_a_to_4():
    ...

def test_unicode_evasion_handles_empty_string():
    ...

# Bad - 모호함
def test_function():
    ...
```

### 3. 하나의 테스트, 하나의 검증

```python
# Good
def test_add_returns_sum():
    assert add(2, 3) == 5

def test_add_handles_negative():
    assert add(-1, 1) == 0

# Bad - 여러 검증 혼합
def test_add():
    assert add(2, 3) == 5
    assert add(-1, 1) == 0
    assert add(0, 0) == 0
```

### 4. 독립적인 테스트

- 각 테스트는 독립적으로 실행 가능해야 함
- 테스트 순서에 의존하지 않음
- 공유 상태 최소화

---

## 참고 자료

- [pytest 공식 문서](https://docs.pytest.org/)
- [Go testing 패키지](https://pkg.go.dev/testing)
- [Codecov 설정 가이드](https://docs.codecov.io/docs)
- [테스트 피라미드](https://martinfowler.com/articles/practical-test-pyramid.html)

---

*마지막 업데이트: 2026-01-18*

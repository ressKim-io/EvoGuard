# Go 1.24 주요 변경사항

> 릴리스: 2025년 2월

## 1. 성능 개선

```
┌─────────────────────────────────────────────┐
│  Swiss Tables 기반 Map                       │
│  ├── 대용량 맵(1024+) 접근: ~30% 향상        │
│  ├── Pre-sized 맵 할당: ~35% 향상            │
│  └── 순회: ~10% 향상 (저부하 ~60%)           │
├─────────────────────────────────────────────┤
│  런타임                                      │
│  ├── CPU 오버헤드: 2-3% 감소                │
│  ├── 소규모 객체 메모리 할당 효율화          │
│  └── GC 일시정지: 15-25% 개선               │
└─────────────────────────────────────────────┘
```

## 2. 언어 기능

### Generic Type Aliases (완전 지원)
```go
type Set[T comparable] = map[T]struct{}
type Result[T any] = Either[error, T]

var userSet Set[string]
```

### JSON `omitzero` 태그
```go
type Config struct {
    Count   int       `json:"count,omitempty"`   // 0이면 생략
    DueDate time.Time `json:"due_date,omitzero"` // zero time이면 생략
}
```

## 3. 도구 개선

### go.mod 도구 의존성 (tools.go 불필요)
```bash
go get -tool golang.org/x/tools/cmd/goimports
go tool goimports -w .
```

### 빌드 시 버전 자동 삽입
```bash
go build  # VCS 태그/커밋 자동 삽입
# 결과: v1.2.3 또는 v1.2.3+dirty
```

## 4. 벤치마크 개선

```go
// 기존
func BenchmarkOld(b *testing.B) {
    for i := 0; i < b.N; i++ {
        doWork()
    }
}

// Go 1.24 (권장)
func BenchmarkNew(b *testing.B) {
    for b.Loop() {
        doWork()
    }
}
```

## 5. DevOps 영향

| 변경 | 영향 |
|-----|-----|
| Swiss Tables | 캐시/맵 처리 성능 향상, 메모리 재조정 고려 |
| GC 개선 | P99 레이턴시 개선 |
| `go get -tool` | CI/CD 파이프라인 단순화 |
| 버전 자동 삽입 | `-ldflags` 제거 가능 |

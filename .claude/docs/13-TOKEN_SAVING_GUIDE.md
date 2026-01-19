# Claude Code 토큰 절약 가이드

> Claude Code 사용 시 토큰 소비를 최적화하여 비용을 절감하고 효율적인 개발을 진행하는 방법

## 목차

1. [비용 개요](#비용-개요)
2. [핵심 절약 전략](#핵심-절약-전략)
3. [CLAUDE.md 최적화](#claudemd-최적화)
4. [세션 관리](#세션-관리)
5. [MCP 서버 관리](#mcp-서버-관리)
6. [프롬프트 엔지니어링](#프롬프트-엔지니어링)
7. [팀 비용 관리](#팀-비용-관리)
8. [명령어 레퍼런스](#명령어-레퍼런스)

---

## 비용 개요

### 평균 비용
| 항목 | 비용 |
|------|------|
| 일일 평균 | $6/개발자 |
| 90th 백분위 | $12 이하 |
| 월간 추정 | $100-200/개발자 |

### 비용 확인 방법
```bash
# 현재 세션 비용 확인
/cost

# 출력 예시:
# Total cost:            $0.55
# Total duration (API):  6m 19.7s
# Total duration (wall): 6h 33m 10.2s
```

---

## 핵심 절약 전략

### 1. `/compact` 명령어 활용 (가장 중요)

Claude Code는 컨텍스트 용량의 95%에 도달하면 자동으로 compact를 수행합니다. 하지만 **수동으로 더 일찍 실행**하는 것이 권장됩니다.

```bash
# 기본 compact
/compact

# 특정 내용에 집중한 compact
/compact 코드 변경사항과 API 사용법에 집중해서 요약해줘

# 약 40개 메시지마다 실행 권장
```

**자동 compact 임계값 조정:**
```bash
# 환경 변수로 50%에서 자동 compact 실행
export CLAUDE_AUTOCOMPACT_PCT_OVERRIDE=50
```

### 2. 새 채팅 시작하기

관련 없는 작업은 **새 채팅**에서 시작하세요.
```bash
# 컨텍스트 완전 초기화
/clear

# 또는 새 터미널에서 claude 시작
claude
```

### 3. 구체적인 쿼리 작성

**나쁜 예:**
```
이 코드 개선해줘
```

**좋은 예:**
```
src/ml_service/feature_store/compute/text_features.py의
transform 메서드(46-82행)에서 unicode_ratio 계산 로직을
리스트 컴프리헨션 대신 generator expression으로 최적화해줘
```

### 4. 복잡한 작업 분할

대규모 작업을 작은 단위로 분할하여 컨텍스트 윈도우 bloat를 방지합니다.

```bash
# 대신에: "전체 Feature Store 구현해줘"
# 이렇게:
# 1. "Feature Registry 모델만 구현해줘"
# 2. (완료 후) /clear
# 3. "Repository 패턴으로 CRUD 구현해줘"
```

---

## CLAUDE.md 최적화

### 권장 구조 (5k 토큰 이하 유지)

```markdown
# Project: EvoGuard

## 프로젝트 요약
- 적대적 공격 시뮬레이션 ML 파이프라인
- Python 3.12, FastAPI, SQLAlchemy 2.0

## 현재 활성 기능
- Feature Store MVP (진행중)
- ML 추론 서비스 (완료)

## 코드 스타일
- ruff 린터 사용
- Type hints 필수
- Docstring: Google 스타일

## 알려진 이슈
- #123: Redis 연결 타임아웃

## TODO
- [ ] Offline Store 구현
- [ ] Model Monitoring 추가
```

### 대용량 문서 분리

CLAUDE.md가 5k 토큰을 초과하면:

```
.claude/
├── CLAUDE.md          # 핵심 정보만 (5k 이하)
└── docs/
    ├── progress.md    # 진행 상황 로그
    ├── api_docs.md    # API 상세 문서
    └── conventions.md # 코딩 컨벤션 상세
```

**필요시에만 로드:**
```bash
# 특정 문서 참조
@.claude/docs/api_docs.md 를 참고해서 엔드포인트 추가해줘
```

---

## 세션 관리

### 세션 종료 프로토콜

1. `/compact`로 요약 (특정 포커스 지정)
2. 요약을 `docs/progress.md`에 추가
3. `session_summary.md`로 독립 복사본 저장

### 세션 시작 프로토콜

```bash
# 새 대화 시작
claude

# 필요한 컨텍스트만 로드
@CLAUDE.md @docs/progress.md 를 참고해서 작업 계속해줘
```

### 세션 재개

```bash
# 이전 세션 재개 (자동 요약 로드)
claude --resume

# 특정 세션 ID로 재개
claude --resume abc123
```

---

## MCP 서버 관리

### 문제점
활성화된 각 MCP 서버는 시스템 프롬프트에 도구 정의를 추가하여 컨텍스트를 소비합니다.

### Tool Search 기능 (2026년 1월)
- 46.9% 토큰 감소 달성 (51K → 8.5K)
- 모든 도구를 미리 로드하지 않고 필요시 검색

### MCP 컨텍스트 확인
```bash
# MCP 서버별 토큰 사용량 확인
/context
```

### 불필요한 서버 비활성화
```json
// ~/.claude/settings.json
{
  "mcpServers": {
    "unused-server": {
      "enabled": false
    }
  }
}
```

---

## 프롬프트 엔지니어링

### 효율적인 프롬프트 패턴

| 비효율적 | 효율적 |
|---------|--------|
| "이 파일 봐줘" | "src/x.py의 클래스 A만 봐줘" |
| "에러 고쳐줘" | "TypeError at line 45 고쳐줘" |
| "테스트 작성해줘" | "transform() 메서드에 대한 단위 테스트 3개만 작성해줘" |
| 순차적 질문 | 관련 질문 그룹화 |

### XML 태그 활용

```xml
<task>
Feature Store에 Redis Online Store 추가
</task>

<constraints>
- 기존 FeatureTransformer 인터페이스 유지
- TTL 기본값: 1시간
- 배치 조회 지원 필수
</constraints>

<files>
- src/ml_service/feature_store/online/redis_store.py (새 파일)
- src/ml_service/feature_store/__init__.py (export 추가)
</files>
```

### 파일 위치 명시

```bash
# 좋은 예
src/ml_service/feature_store/compute/text_features.py:46-82 의
transform 메서드 리팩토링해줘

# 나쁜 예
text_features 파일의 transform 함수 수정해줘
```

---

## 팀 비용 관리

### 권장 Rate Limit (TPM/RPM)

| 팀 규모 | 사용자당 TPM | 사용자당 RPM |
|--------|-------------|-------------|
| 1-5명 | 200k-300k | 5-7 |
| 5-20명 | 100k-150k | 2.5-3.5 |
| 20-50명 | 50k-75k | 1.25-1.75 |
| 50-100명 | 25k-35k | 0.62-0.87 |

### Workspace 지출 한도 설정

Claude Console에서 Admin 권한으로:
1. Settings → Workspace
2. Spend limits 설정
3. "Claude Code" workspace 자동 생성됨

### 멀티 클라우드 비용 추적

Bedrock/Vertex 사용 시 [LiteLLM](https://docs.litellm.ai/docs/proxy/virtual_keys#tracking-spend) 활용:
```bash
# LiteLLM으로 키별 지출 추적
pip install litellm
```

---

## 명령어 레퍼런스

| 명령어 | 설명 | 토큰 영향 |
|--------|------|----------|
| `/compact` | 대화 요약 및 압축 | ⬇️ 대폭 감소 |
| `/compact [지시]` | 특정 내용 집중 요약 | ⬇️ 대폭 감소 |
| `/clear` | 컨텍스트 완전 초기화 | ⬇️ 초기화 |
| `/cost` | 현재 비용 확인 | - |
| `/context` | 컨텍스트 사용량 확인 | - |
| `/config` | 설정 (auto-compact 등) | - |
| `@파일명` | 특정 파일 참조 로드 | ⬆️ 파일 크기만큼 |

---

## CLAUDE.md에 추가할 Compact 지시

```markdown
# Summary Instructions

compact 실행 시 다음에 집중해주세요:
- 완료된 작업 목록
- 발생한 에러와 해결 방법
- 다음 단계 TODO
- 중요 코드 변경사항
```

---

## 체크리스트

### 일일 습관
- [ ] 작업 시작 전 `/cost` 확인
- [ ] 40개 메시지마다 `/compact` 실행
- [ ] 작업 전환 시 `/clear` 또는 새 세션
- [ ] 구체적인 파일 경로와 라인 번호 명시

### 주간 점검
- [ ] CLAUDE.md 크기 확인 (5k 이하)
- [ ] 불필요한 MCP 서버 비활성화
- [ ] Claude Console에서 팀 사용량 리뷰

---

## 참고 자료

- [Claude Code 공식 비용 관리 문서](https://code.claude.com/docs/en/costs)
- [토큰 최적화 60% 달성 가이드](https://medium.com/@jpranav97/stop-wasting-tokens-how-to-optimize-claude-code-context-by-60-bfad6fd477e5)
- [세션 관리 워크플로우](https://gist.github.com/artemgetmann/74f28d2958b53baf50597b669d4bce43)
- [컨텍스트 54% 감소 전략](https://gist.github.com/johnlindquist/849b813e76039a908d962b2f0923dc9a)
- [Tool Search로 46.9% 토큰 절감](https://medium.com/@joe.njenga/claude-code-just-cut-mcp-context-bloat-by-46-9-51k-tokens-down-to-8-5k-with-new-tool-search-ddf9e905f734)

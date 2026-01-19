# EvoGuard Project

## 프로젝트 요약
적대적 공격 시뮬레이션을 통한 ML 모델 강건성 검증 파이프라인

## 기술 스택
- **Backend**: Go 1.21+ (api-service), Python 3.12 (ml-service)
- **ML**: PyTorch, scikit-learn, MLflow
- **DB**: PostgreSQL, Redis
- **Infra**: Docker, Kubernetes, GitHub Actions

## 주요 서비스
| 서비스 | 경로 | 설명 |
|--------|------|------|
| api-service | `/api-service` | Go REST API (배틀 관리) |
| ml-service | `/ml-service` | Python ML 추론 서비스 |
| attacker | `/attacker` | 적대적 공격 생성 |
| defender | `/defender` | 방어 모델 |

## 코드 스타일
- **Python**: ruff 린터, type hints 필수, Google docstring
- **Go**: golangci-lint, 표준 포맷팅
- **Commit**: Conventional Commits (`feat:`, `fix:`, `docs:`)

## 현재 진행 상황
- Feature Store MVP 완료
- Model Monitoring 설계 완료
- 다음: Offline/Online Store 구현

## 참고 문서
> 아래 문서들은 필요시 `@파일경로`로 로드하세요

| 문서 | 경로 | 설명 |
|------|------|------|
| 토큰 절약 가이드 | `.claude/docs/13-TOKEN_SAVING_GUIDE.md` | Claude Code 비용 최적화 |
| 프로젝트 체크리스트 | `.claude/docs/00-PROJECT_CHECKLIST.md` | 전체 진행 상황 |
| 개발 로드맵 | `.claude/docs/07-DEVELOPMENT_ROADMAP.md` | 단계별 계획 |
| Feature Store | `.claude/docs/11-FEATURE_STORE.md` | Feature Store 설계 |
| Model Monitoring | `.claude/docs/12-MODEL_MONITORING.md` | 모니터링 설계 |

## Summary Instructions

compact 실행 시 다음에 집중:
- 완료된 작업 목록
- 발생한 에러와 해결 방법
- 다음 단계 TODO
- 중요 코드 변경사항

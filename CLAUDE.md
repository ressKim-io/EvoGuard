# EvoGuard Project

## 프로젝트 요약
**한국어 혐오표현 탐지를 위한 공격-방어 공진화(Co-Evolution) 시스템**

적대적 공격 시뮬레이션을 통해 ML 모델의 강건성을 지속적으로 향상시키는 파이프라인

---

## 핵심: 한국어 공격-방어 시스템

### 작동 원리
```
┌─────────────┐                    ┌─────────────┐
│   Attacker  │ ── evasion>30% ──▶ │  Defender   │ 재학습
│  (공격자)    │                    │  (방어자)    │
│             │ ◀── evasion<10% ── │             │ 진화
└─────────────┘                    └─────────────┘
        └────── 10~30% = 균형 상태 ──────┘
```

### 현재 최고 성능
| 모델 | F1 Score | FP | FN | 용도 |
|------|----------|----|----|------|
| **앙상블 (P2+Coevo AND)** | **0.9696** | **60** | 168 | 프로덕션 권장 |
| Phase 2 Combined | 0.9675 | 80 | 164 | 단일 모델 최고 |
| Coevolution Latest | 0.9245 | 611 | 8 | 공격 방어 특화 (FN↓) |
| Phase 5 CNN | 0.8946 | 419 | 362 | 실험 |

### 베이스 모델
- `beomi/KcELECTRA-base-v2022` (한국어 ELECTRA)

### 학습 데이터셋 (한국어)
- **KOTOX**: Korean Toxic 데이터셋
- **BEEP**: 한국어 혐오표현
- **UnSmile**: 여성가족부 혐오표현 데이터
- **욕설 데이터셋**: 한국어 비속어

---

## 품질 향상 스크립트

### 단계별 학습 (Phase 1-5)
| Phase | 스크립트 | 설명 |
|-------|----------|------|
| 1 | `ml-service/scripts/phase1_deobfuscation.py` | 난독화 해제 학습 |
| 2 | `ml-service/scripts/phase2_combined_data.py` | 통합 데이터 학습 (최고 성능) |
| 3 | `ml-service/scripts/phase3_large_model.py` | 대형 모델 실험 |
| 4 | `ml-service/scripts/phase4_augmented.py` | 에러 기반 증강 학습 |
| 5 | `ml-service/scripts/phase5_cnn_enhanced.py` | CNN 레이어 추가 |

### MLOps 자동화 (반복 실행)

> **"학습 시작해"** 라고 하면 `.claude/docs/14-TRAINING_GUIDE.md` 참조

```bash
cd ml-service && source .venv/bin/activate

# 연속 공진화 (권장) - 트리거 기반, GPU 최적화
python scripts/run_continuous_coevolution.py --max-cycles 100

# 목표 달성까지 실행
python scripts/run_continuous_coevolution.py --target-evasion 0.03

# 시간 제한 실행
python scripts/run_optimized_coevolution.py --hours 4
```

**연속 공진화 특징:**
- 작업 완료 즉시 다음 사이클 (시간 기반 X)
- AMP(Mixed Precision) 적용
- 트리거 기반 재학습 (evasion>8%, 샘플 축적, 주기적)
- 수렴 감지 시 공격 자동 강화

### 분석 및 유틸리티
| 스크립트 | 설명 |
|----------|------|
| `ml-service/scripts/error_analysis.py` | FP/FN 에러 분석 |
| `ml-service/scripts/augment_data.py` | 데이터 증강 |
| `ml-service/scripts/retrain_from_samples.py` | 실패 샘플로 재학습 |

---

## 주요 모델 경로
| 모델 | 경로 |
|------|------|
| 앙상블 추론 | `ml-service/src/ml_service/inference/ensemble_classifier.py` |
| Phase 2 | `ml-service/models/phase2-combined/` |
| Coevolution Latest | `ml-service/models/coevolution-latest/` |
| Phase 4 (증강) | `ml-service/models/phase4-augmented/` |
| 학습 결과 | `ml-service/models/TRAINING_RESULTS.md` |

---

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
- Phase 1-5 학습 완료
- 슬랭 강화 공진화 500 사이클 완료 (evasion 20.5% → 0.0%)
- **AND 앙상블 적용 완료** (F1: 0.9696, FP: 60)
- 프로덕션 배포 준비 완료

## 참고 문서
> 아래 문서들은 필요시 `@파일경로`로 로드하세요

| 문서 | 경로 | 설명 |
|------|------|------|
| **학습 가이드** | `.claude/docs/14-TRAINING_GUIDE.md` | **학습 시작 시 필수 참조** |
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

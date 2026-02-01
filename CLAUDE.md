# EvoGuard Project

## 프로젝트 요약
**한국어 혐오표현 탐지를 위한 공격-방어 공진화(Co-Evolution) 시스템**

적대적 공격 시뮬레이션을 통해 ML 모델의 강건성을 지속적으로 향상시키는 파이프라인

---

## 핵심: 한국어 공격-방어 시스템

### 작동 원리 (균형 공진화)
```
┌─────────────┐                    ┌─────────────┐
│   Attacker  │ ── evasion>8% ───▶ │  Defender   │ 재학습
│  (공격자)    │                    │  (방어자)    │
│             │ ◀── evasion<5% ─── │             │ 진화
└─────────────┘                    └─────────────┘
        └────── 5~8% = 균형 구간 ──────┘

AttackerEvolver (evasion < 5%):
  - aggressive (<3%): 새 전략 5개, 슬랭 확장, 탐색 +1.0
  - normal (3-5%): 새 전략 2개, 탐색 +0.3

HardNegativeMiner (어려운 샘플 집중 학습):
  - FN 가중치 2.0 (독성→정상, 가장 위험)
  - FP 가중치 1.5 (정상→독성)
  - 경계 케이스 가중치 1.0 (confidence 0.4~0.6)
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

# 균형 공진화 (권장) - 공격자/방어자 양쪽 자동 진화
python scripts/run_balanced_coevolution.py --max-cycles 100

# 목표 달성까지 실행
python scripts/run_balanced_coevolution.py --target-evasion 0.03

# 연속 공진화 (기존 방식)
python scripts/run_continuous_coevolution.py --max-cycles 100

# 백그라운드 실행 (SSH 끊김 대비)
nohup python scripts/run_balanced_coevolution.py --max-cycles 500 > logs/balanced_$(date +%Y%m%d_%H%M%S).log 2>&1 &
```

**균형 공진화 특징:**
- 공격자 자동 진화 (evasion < 5%): AttackerEvolver
- 어려운 샘플 집중 학습: HardNegativeMiner
- 균형 구간 추적 (5-8%)
- AMP(Mixed Precision) 적용

### 분석 및 유틸리티
| 스크립트 | 설명 |
|----------|------|
| `ml-service/scripts/error_analysis.py` | FP/FN 에러 분석 |
| `ml-service/scripts/augment_data.py` | 데이터 증강 |
| `ml-service/scripts/retrain_from_samples.py` | 실패 샘플로 재학습 |
| `ml-service/scripts/cleanup_models.py` | 모델 정리 (--dry-run/--execute) |
| `ml-service/scripts/model_version_manager.py` | 버전 관리 (save/list/prune/restore) |

---

## 주요 모델 경로

```
models/
├── phase2-combined/          # 프로덕션 (F1: 0.9675)
├── phase2-slang-enhanced/    # 슬랭 강화 베이스
├── phase4-augmented/         # 프로덕션 백업
├── coevolution-latest/       # 공진화 최신
├── coevolution/versions/     # 공진화 버전 (최근 3개)
├── archive/                  # 압축된 실험 모델
└── MODEL_REGISTRY.json       # 모델 레지스트리
```

| 모델 | 경로 | 설명 |
|------|------|------|
| 앙상블 추론 | `ml-service/src/ml_service/inference/ensemble_classifier.py` | AND 앙상블 |
| Phase 2 | `ml-service/models/phase2-combined/` | 단일 최고 성능 |
| Coevolution Latest | `ml-service/models/coevolution-latest/` | 공진화 최신 |
| 레지스트리 | `ml-service/models/MODEL_REGISTRY.json` | 모델 목록 |
| 학습 결과 | `ml-service/models/TRAINING_RESULTS.md` | 성능 기록 |

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
- **2026-01-27**: 연속 공진화 100 사이클 완료 (evasion 19.9% → 0.3%, 재학습 36회, 50분)
- **2026-02-01**: 균형 공진화 시스템 구현 (AttackerEvolver, HardNegativeMiner, BalancedCoevolution)
- **2026-02-01**: 균형 공진화 10 사이클 테스트 완료 (evasion 29.5% → 3.2%, 5.2분)

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

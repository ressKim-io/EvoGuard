# Adversarial MLOps Pipeline 구현 계획

## 목표
공격 → 품질 측정 → 임계값 미달 시 재학습 → 배포가 **자동으로 계속 돌아가는** 시스템 구축

## 아키텍처 개요

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    ADVERSARIAL MLOPS PIPELINE                           │
│                                                                         │
│   ┌─────────┐    ┌─────────┐    ┌─────────┐    ┌─────────┐             │
│   │ ATTACK  │───▶│ EVALUATE│───▶│ DECIDE  │───▶│ RETRAIN │             │
│   │         │    │ QUALITY │    │         │    │         │             │
│   └─────────┘    └─────────┘    └─────────┘    └─────────┘             │
│        │              │              │              │                   │
│        ▼              ▼              ▼              ▼                   │
│   공격 변형 생성   evasion_rate   threshold 비교   QLoRA 학습           │
│   (5 strategies)  F1/Precision    (<15%? >85%?)   + 데이터 증강         │
│                                        │                                │
│                                        ▼                                │
│                              ┌─────────────────┐                        │
│                              │ CHAMPION/       │                        │
│                              │ CHALLENGER      │◀────── A/B Test        │
│                              │ PROMOTION       │                        │
│                              └─────────────────┘                        │
└─────────────────────────────────────────────────────────────────────────┘
```

## 구현할 모듈 (8개)

| # | 파일 | 역할 |
|---|------|------|
| 1 | `pipeline/config.py` | 설정 클래스 (임계값, 배치 크기) |
| 2 | `pipeline/attack_runner.py` | 공격 실행 및 결과 수집 |
| 3 | `pipeline/quality_gate.py` | 품질 판단 (PASS/FAIL) |
| 4 | `pipeline/sample_collector.py` | 실패 샘플 수집/저장 |
| 5 | `pipeline/data_augmentor.py` | 데이터 증강 |
| 6 | `pipeline/model_promoter.py` | Champion/Challenger 관리 |
| 7 | `pipeline/orchestrator.py` | 전체 파이프라인 조율 |
| 8 | `pipeline/api.py` | REST API 엔드포인트 |

## 품질 임계값 (설정 가능)

```yaml
quality_gate:
  max_evasion_rate: 0.15      # 15% 초과 시 재학습
  min_f1_score: 0.85          # 85% 미만 시 재학습
  min_f1_drop: 0.05           # 5%p 하락 시 재학습
```

## 실행 방식
- **수동 트리거만** (자동 스케줄 없음)
- CLI: `python scripts/run_pipeline.py`
- API: `POST /pipeline/trigger`

## 구현 순서

- [x] Step 1: 설정 및 기본 구조 (`config.py`, `__init__.py`)
- [ ] Step 2: 공격 실행 (`attack_runner.py`)
- [ ] Step 3: 품질 판단 (`quality_gate.py`)
- [ ] Step 4: 실패 샘플 수집 (`sample_collector.py`)
- [ ] Step 5: 데이터 증강 (`data_augmentor.py`)
- [ ] Step 6: 모델 승격 (`model_promoter.py`)
- [ ] Step 7: 메인 오케스트레이터 (`orchestrator.py`)
- [ ] Step 8: API 엔드포인트 (`api.py`)
- [ ] Step 9: CLI 스크립트 (`scripts/run_pipeline.py`)
- [ ] Step 10: 테스트 및 통합 확인

---
*Created: 2026-01-20*

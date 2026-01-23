# 실험: 3-Phase 학습 파이프라인

> 날짜: 2026-01-20
> 목표: 단계별 학습을 통한 한국어 독성 분류 모델 최적화

## 1. 배경

### 1.1 기존 상태
- 기본 toxic-classifier 모델: F1 0.9277
- 단일 데이터셋 학습의 한계

### 1.2 가설
- 단계별 학습이 단일 학습보다 효과적일 것
- 다양한 데이터셋 통합이 일반화 성능을 높일 것

## 2. 실험 설계

### 2.1 3-Phase 구성

| Phase | 목표 | 데이터 | 모델 |
|-------|------|--------|------|
| Phase 1 | 난독화 해제 | KOTOX 난독화 패턴 | KcELECTRA |
| Phase 2 | 통합 학습 | 전체 데이터셋 | KcELECTRA |
| Phase 3 | 대형 모델 | 전체 데이터셋 | KoELECTRA-base-v3 |

### 2.2 데이터셋

| 데이터셋 | 크기 | 특징 |
|----------|------|------|
| KOTOX | 11,010 | 난독화 포함 |
| BEEP | 7,896 | 혐오 표현 |
| UnSmile | 15,005 | 다중 레이블 |
| 욕설 데이터 | 5,825 | 직접적 욕설 |
| **합계** | **48,199** | |

## 3. 실험 결과

### 3.1 Phase별 결과

| Phase | Epochs | F1 | 특징 |
|-------|--------|-----|------|
| Phase 1 | 15 | 0.9036 | 난독화 패턴 학습 |
| **Phase 2** | 10 | **0.9597** | 최고 성능 |
| Phase 3 | 10 | 0.9060 | 모델 크기 증가 |

### 3.2 분석

**Phase 2가 최고 성능인 이유:**
1. 통합 데이터셋 (48K)으로 다양한 패턴 학습
2. KcELECTRA가 한국어에 최적화
3. 적절한 모델 크기 (과적합 방지)

**Phase 3이 낮은 이유:**
1. 데이터 대비 모델이 커서 과적합 가능성
2. 더 긴 학습 필요

## 4. 결론

### 4.1 성과
- F1 0.9277 → 0.9597 **(+3.2%p 개선)**
- 통합 데이터셋의 효과 입증
- Phase 2가 production 모델로 적합

### 4.2 향후 과제
- 에러 분석을 통한 취약점 파악
- 데이터 증강으로 추가 개선

## 5. 재현 방법

```bash
# 전체 Phase 실행
./scripts/run_all_phases.sh

# 개별 실행
python scripts/phase1_deobfuscation.py --epochs 15
python scripts/phase2_combined_data.py --epochs 10
python scripts/phase3_large_model.py --epochs 10
```

## 6. 관련 파일

- `scripts/run_all_phases.sh` - 전체 실행 스크립트
- `scripts/phase1_deobfuscation.py`
- `scripts/phase2_combined_data.py`
- `scripts/phase3_large_model.py`
- `models/phase1-deobfuscated/`
- `models/phase2-combined/`
- `models/phase3-large/`

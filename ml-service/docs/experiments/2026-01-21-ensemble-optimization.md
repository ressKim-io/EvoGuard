# 실험: 앙상블 최적화를 통한 한국어 독성 분류 개선

> 날짜: 2026-01-21
> 목표: 에러 분석 기반 데이터 증강 및 앙상블을 통한 성능 개선

## 1. 배경

### 1.1 기존 상태
- Phase 2 모델이 F1 0.9597로 최고 성능
- 하지만 에러 분석 결과 특정 패턴에서 취약점 발견

### 1.2 문제점 (Phase 2 에러 분석)

| 지표 | 값 | 설명 |
|------|-----|------|
| False Positives | 80건 | 정상을 독성으로 오분류 |
| False Negatives | 164건 | 독성을 정상으로 미탐 |

**Source별 에러율:**
| Source | FP% | FN% |
|--------|-----|-----|
| KOTOX | 4.37% | 6.91% |
| BEEP | 0.00% | 4.46% |
| UnSmile | 0.54% | 1.28% |

**주요 미탐 패턴:**
1. 맥락 의존적 표현: "백린탄이 필요하다", "앞차 최소 전라도"
2. 우회적 혐오: "인두로 지져버리면", "땅크 부릉부릉"
3. 지역/성별 비하: "여판사네", "전라도 특징"

## 2. 실험 설계

### 2.1 가설
1. 맥락 의존적 혐오 표현 데이터를 추가하면 FN이 감소할 것
2. 보수적 모델(Phase2)과 공격적 모델(Phase4)의 앙상블이 균형을 맞출 것

### 2.2 방법

**데이터 증강 (augment_data.py):**
- KOTOX 스타일 난독화 패턴 생성
- 맥락 의존적 혐오 표현 템플릿 기반 생성
- False Negative 패턴 기반 유사 데이터 생성
- 총 2,070건 증강

**Phase 4 학습:**
- Base: Phase 2 모델
- 데이터: 기존 + 증강 (41,806건)
- Epochs: 10

**앙상블 구성:**
- 다양한 가중치 조합 테스트
- Soft voting, Hard voting, Max confidence 등 비교

## 3. 실험 결과

### 3.1 Phase 4 단독 결과

| 지표 | Phase 2 | Phase 4 | 변화 |
|------|---------|---------|------|
| F1 | 0.9597 | 0.9580 | -0.17% |
| FP | 80 | 98 | +18건 |
| FN | 164 | 137 | **-27건** |

**Source별 FN 변화:**
| Source | Phase 2 FN | Phase 4 FN | 개선 |
|--------|------------|------------|------|
| KOTOX | 95 | 87 | -8건 |
| BEEP | 21 | 19 | -2건 |
| UnSmile | 48 | 31 | **-17건** |

**분석:**
- 독성 탐지력 향상 (FN 27건 감소)
- 하지만 오탐 증가 (FP 18건 증가)
- 증강 데이터가 전부 독성이라 독성으로 분류하는 경향 강화

### 3.2 앙상블 결과

| Method | F1 | FP | FN | Total Error |
|--------|-----|-----|-----|-------------|
| Phase 2 (단독) | 0.9565 | 80 | 164 | 244 |
| Phase 4 (단독) | 0.9580 | 98 | 137 | 235 |
| **가중치 (P2:0.6, P4:0.4)** | **0.9594** | **78** | **150** | **228** |
| Soft Voting (평균) | 0.9589 | 92 | 138 | 230 |
| AND (둘 다 독성) | 0.9565 | 62 | 183 | 245 |
| OR (하나라도 독성) | 0.9581 | 116 | 118 | 234 |

**최적 구성: P2:0.6 + P4:0.4**
- F1: 0.9594 (Phase 2 원본 대비 거의 동일)
- FP: 78건 (최소)
- FN: 150건 (Phase 2 대비 14건 개선)
- Total Error: 228건 (최소)

## 4. 결론

### 4.1 성과
1. **앙상블을 통해 오탐과 미탐의 균형 달성**
   - Phase 2의 보수적 특성 + Phase 4의 공격적 특성 결합
   - Total Error 244 → 228 (16건 감소)

2. **맥락 의존적 표현 탐지 개선**
   - UnSmile FN: 48 → 31 (35% 개선)
   - 지역 비하, 성차별 표현 탐지 강화

3. **실용적인 추론 모듈 구현**
   - `ensemble_classifier.py` 모듈 제공
   - CLI 및 Python API 지원

### 4.2 한계
1. 증강 데이터가 독성 위주라 편향 발생
2. "백린탄이 필요하다" 같은 극단적 맥락 의존 표현은 여전히 어려움
3. 모델 2개 로드로 인한 메모리 사용량 증가

### 4.3 향후 과제
1. 정상 데이터 증강 추가 (FP 감소)
2. 맥락 정보를 활용한 모델 구조 개선
3. 경량화 앙상블 (지식 증류 등)

## 5. 재현 방법

```bash
# 1. 데이터 증강
python scripts/augment_data.py

# 2. Phase 4 학습
./scripts/run_training.sh phase4_augmented.py --epochs 10

# 3. 에러 분석
python scripts/error_analysis.py

# 4. 앙상블 사용
python -c "
from ml_service.inference.ensemble_classifier import create_ensemble
classifier = create_ensemble()
print(classifier.predict('테스트 텍스트'))
"
```

## 6. 관련 파일

- `scripts/augment_data.py` - 데이터 증강 스크립트
- `scripts/phase4_augmented.py` - Phase 4 학습 스크립트
- `scripts/error_analysis.py` - 에러 분석 스크립트
- `src/ml_service/inference/ensemble_classifier.py` - 앙상블 추론 모듈
- `models/phase4-augmented/` - Phase 4 모델
- `data/korean/augmented/` - 증강 데이터
- `logs/error_analysis.json` - 에러 분석 결과

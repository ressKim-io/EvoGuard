# 🏟️ Content Arena: Self-Improving 콘텐츠 모더레이션 시스템

> **Adversarial Learning 기반 자가 개선 ML 시스템**  
> 공격자(Evasion Bot)와 방어자(Filter Model)가 경쟁하며 진화하는 MLOps 프로젝트

---

## 📌 프로젝트 개요

### 핵심 컨셉
```
┌─────────────────────────────────────────────────────────────────┐
│                                                                 │
│   ┌─────────────┐         경쟁 (Battle)       ┌─────────────┐  │
│   │   ATTACKER  │ ◄─────────────────────────► │  DEFENDER   │  │
│   │ (로컬 LLM)  │                             │(Fine-tuned) │  │
│   │             │                             │             │  │
│   │ Mistral 7B  │                             │ BERT/QLoRA  │  │
│   └──────┬──────┘                             └──────┬──────┘  │
│          │                                          │          │
│          │ 우회 패턴 생성                           │ 탐지 시도│
│          │                                          │          │
│          ▼                                          ▼          │
│   ┌─────────────────────────────────────────────────────────┐  │
│   │                   BATTLE ARENA                          │  │
│   │  - 라운드 단위 대결                                      │  │
│   │  - 우회 성공/실패 기록                                   │  │
│   │  - 품질 메트릭 측정 (정확도, F1 Score)                  │  │
│   └─────────────────────────────────────────────────────────┘  │
│                             │                                  │
│                             ▼                                  │
│   ┌─────────────────────────────────────────────────────────┐  │
│   │              AUTO RETRAIN (로컬 GPU)                    │  │
│   │  - 우회 성공 패턴 → 방어 모델 재학습 데이터             │  │
│   │  - Champion/Challenger 비교                             │  │
│   │  - 더 나은 모델 자동 승격                               │  │
│   └─────────────────────────────────────────────────────────┘  │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 왜 이 프로젝트인가?

| 일반 AI 프로젝트 | Content Arena |
|-----------------|---------------|
| LLM API 붙이고 끝 | **Self-improving 시스템** |
| 품질 = "그냥 작동함" | **품질 = 측정 가능한 개선** |
| 단일 모델 | **Adversarial 구조 + Champion/Challenger** |
| 수동 배포 | **Automated MLOps 파이프라인** |

### 품질 향상 예시

```
Round 1:  탐지율 60% → 미탐지 패턴으로 재학습
Round 2:  탐지율 68% → 새로운 우회 패턴 학습
Round 3:  탐지율 75% → ...
...
Round N:  탐지율 95% ✅

→ 그래프로 품질 향상 "증명" 가능!
→ 면접에서 "라운드 1~10까지 탐지율 60%→95% 개선" 시연
```

---

## 🎯 프로젝트 목표

### 기술적 목표
1. **Self-Improving System 구현**: 피드백 루프를 통한 자동 품질 향상
2. **MLOps 파이프라인 구축**: 학습 → 평가 → 배포 자동화
3. **로컬 LLM 활용**: Ollama + Mistral 7B로 비용 0원 운영
4. **QLoRA Fine-tuning**: 8GB VRAM으로 모델 커스터마이징

### 포트폴리오 목표
- **백엔드**: Go (Gin) - 고성능 API 서버
- **ML Engineering**: Python - QLoRA Fine-tuning, 모델 서빙
- **DevOps**: K8s, Docker, CI/CD, 모니터링
- **AI/ML**: Adversarial Learning, Champion/Challenger, MLflow

---

## 🛠️ 기술 스택

### 서비스 레이어 (Go)
```
├── Gin Framework      - 고성능 REST API
├── GORM               - ORM
├── go-redis           - 캐싱
└── Prometheus client  - 메트릭 노출
```

### ML 레이어 (Python)
```
├── Transformers + PEFT  - QLoRA Fine-tuning
├── Ollama               - 로컬 LLM 실행
├── FastAPI              - 추론 API 서버
├── MLflow               - 실험 추적 & 모델 레지스트리
└── LangChain            - LLM 오케스트레이션
```

### 인프라 레이어
```
├── Docker + Docker Compose  - 컨테이너화
├── Kubernetes (k3d)         - 오케스트레이션 (Docker 기반)
├── PostgreSQL               - 메인 데이터베이스
├── Redis                    - 캐싱 & 세션
├── Prometheus + Grafana     - 모니터링
└── GitHub Actions           - CI/CD
```

---

## 📂 문서 구조

```
docs/
├── 01-ARCHITECTURE.md       # 상세 아키텍처
├── 02-TECH_STACK.md         # 기술 스택 상세
├── 03-ENVIRONMENT_SETUP.md  # 환경 설정 가이드
├── 04-ML_PIPELINE.md        # ML 파이프라인 상세
├── 05-MLOPS.md              # MLOps 파이프라인
├── 06-API_SPEC.md           # API 명세
├── 07-DEVELOPMENT_ROADMAP.md # 개발 로드맵
└── 08-PROJECT_STRUCTURE.md  # 프로젝트 구조
```

---

## 🚀 빠른 시작

### 1. 환경 요구사항
```
- GPU: NVIDIA 4060Ti (VRAM 8GB) 이상
- RAM: 32GB 권장
- OS: Windows (WSL2) / Linux / macOS
- Docker & Docker Compose
- Python 3.10+
- Go 1.21+
```

### 2. 설치 및 실행
```bash
# 저장소 클론
git clone https://github.com/your-username/content-arena.git
cd content-arena

# 환경 설정
cp .env.example .env
# .env 파일 수정 (DB, Redis, MLflow 설정)

# Docker Compose로 인프라 실행
docker-compose up -d

# Ollama 설치 & 모델 다운로드
curl -fsSL https://ollama.com/install.sh | sh
ollama pull mistral:7b-instruct-v0.2-q4_K_S

# Python 환경 설정
cd ml-service
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Go 서비스 실행
cd ../api-service
go mod download
go run cmd/server/main.go

# ML 서비스 실행
cd ../ml-service
uvicorn app.main:app --reload --port 8001
```

### 3. 첫 배틀 실행
```bash
# Battle API 호출
curl -X POST http://localhost:8080/api/v1/battles \
  -H "Content-Type: application/json" \
  -d '{"rounds": 10, "attack_strategy": "unicode_evasion"}'
```

---

## 📊 주요 기능

### 1. Battle Arena
- 라운드 기반 공격/방어 대결
- 다양한 공격 전략 (유니코드 변형, LLM 생성 등)
- 실시간 결과 기록 및 시각화

### 2. Auto Retrain
- 우회 성공 패턴 자동 수집
- 트리거 기반 재학습 (탐지율 임계값 도달 시)
- Champion/Challenger 비교 평가

### 3. MLOps Dashboard
- 라운드별 품질 추이 그래프
- 모델 버전 관리 및 비교
- 실험 추적 (MLflow UI)

---

## 📅 개발 일정 (12주)

| Phase | 기간 | 목표 |
|-------|------|------|
| **Phase 1** | 1-2주 | 환경 셋업, 기본 API 구조 |
| **Phase 2** | 3-5주 | 공격자 구축 (Ollama + 전략) |
| **Phase 3** | 6-9주 | 방어자 + Fine-tuning 파이프라인 |
| **Phase 4** | 10-11주 | MLOps 자동화, 모니터링 |
| **Phase 5** | 12주 | 문서화, 데모, 포트폴리오 정리 |

---

## 🏆 예상 성과

### 기술적 성과
- [ ] Self-improving 시스템 동작 검증
- [ ] 탐지율 60% → 90%+ 개선 달성
- [ ] 완전 자동화된 MLOps 파이프라인
- [ ] 로컬 GPU로 Fine-tuning 성공

### 포트폴리오 성과
- [ ] 품질 향상 그래프 (핵심!)
- [ ] 아키텍처 다이어그램
- [ ] 데모 영상 (5분)
- [ ] 기술 블로그 포스트

---

## 📚 참고 자료

- [Ollama Documentation](https://ollama.com/)
- [QLoRA Paper](https://arxiv.org/abs/2305.14314)
- [MLflow Documentation](https://mlflow.org/docs/latest/index.html)
- [Gin Web Framework](https://gin-gonic.com/)
- [Jigsaw Toxic Comment Dataset](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge)

---

## 📝 라이선스

MIT License

---

## 🤝 기여

이 프로젝트는 개인 학습 및 포트폴리오 목적으로 개발되었습니다.

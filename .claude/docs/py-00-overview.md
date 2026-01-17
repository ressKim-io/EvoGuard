# Python ML Stack Best Practices Guide

> 📅 Last Updated: January 2026

## 📦 Stack Overview

| Package | Version | Purpose |
|---------|---------|---------|
| Python | 3.12.8 | Runtime |
| PyTorch | 2.5.1+cu124 | Deep Learning Framework |
| transformers | 4.48.3 | Pre-trained Models |
| PEFT | 0.14.0 | Parameter-Efficient Fine-Tuning |
| bitsandbytes | 0.49.1 | Quantization |
| accelerate | 1.5.2 | Distributed Training |
| datasets | 3.2.0 | Data Loading |
| MLflow | 2.22.4 | Experiment Tracking |

## 📁 Guide Structure

```
python-ml-guide/
├── README.md                    # 이 파일
├── 01_project_setup.md          # uv + pyproject.toml 설정
├── 02_pytorch_best_practices.md # PyTorch 최적화
├── 03_transformers_peft.md      # Transformers & PEFT
├── 04_quantization.md           # bitsandbytes 양자화
├── 05_mlflow_tracking.md        # 실험 추적
└── 06_common_patterns.md        # 공통 패턴 & 팁
```

## 🚀 Quick Start

```bash
# uv 설치
curl -LsSf https://astral.sh/uv/install.sh | sh

# 프로젝트 초기화
uv init ml-project
cd ml-project

# 의존성 추가
uv add torch transformers peft bitsandbytes accelerate datasets mlflow

# GPU 지원 PyTorch (CUDA 12.4)
uv add torch --index-url https://download.pytorch.org/whl/cu124
```

## 🎯 Key Principles

1. **환경 격리**: uv로 가상환경 자동 관리
2. **재현성**: uv.lock으로 정확한 버전 고정
3. **메모리 효율**: 4-bit/8-bit 양자화로 VRAM 절약
4. **실험 추적**: MLflow로 모든 실험 기록
5. **성능**: torch.compile()로 2-3x 속도 향상

## 📌 Version Compatibility Matrix

```
Python 3.12 ─┬─ PyTorch 2.4+ (torch.compile 지원)
             ├─ CUDA 12.1+ 권장
             └─ bitsandbytes 0.43+

transformers 4.40+ ─┬─ BitsAndBytesConfig 지원
                    └─ PEFT 0.10+ 호환
```

## ⚠️ Common Pitfalls

- `pip install` 대신 `uv add` 사용
- PyTorch: `device_map="auto"`는 **추론 전용**
- 4-bit 양자화: `bnb_4bit_compute_dtype=torch.bfloat16` 필수
- MLflow: `mlflow.pytorch.autolog()`는 Lightning에서만 동작

## 🔗 References

- [PyTorch Performance Tuning Guide](https://pytorch.org/tutorials/recipes/recipes/tuning_guide.html)
- [HuggingFace Quantization](https://huggingface.co/docs/transformers/quantization/bitsandbytes)
- [MLflow PyTorch Guide](https://mlflow.org/docs/latest/ml/deep-learning/pytorch/)
- [uv Documentation](https://docs.astral.sh/uv/)

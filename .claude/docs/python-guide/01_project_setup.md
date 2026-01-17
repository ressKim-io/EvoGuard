# 01. Project Setup with uv

## uv 소개

uv는 Rust로 작성된 Python 패키지 매니저로, pip보다 10-100배 빠릅니다.
2026년 현재 ML 프로젝트의 표준 도구로 자리잡았습니다.

## 설치

```bash
# macOS/Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# Windows
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"

# 버전 확인
uv --version
```

## 프로젝트 초기화

```bash
# 새 프로젝트 생성
uv init ml-project --python 3.12
cd ml-project

# 구조
# ml-project/
# ├── .python-version    # Python 버전 고정
# ├── .venv/             # 가상환경 (자동 생성)
# ├── pyproject.toml     # 프로젝트 설정
# ├── uv.lock            # 정확한 버전 잠금
# └── main.py
```

## pyproject.toml 구성

```toml
[project]
name = "ml-project"
version = "0.1.0"
description = "ML Training Pipeline"
requires-python = ">=3.12"

dependencies = [
    "torch>=2.5.0",
    "transformers>=4.48.0",
    "peft>=0.14.0",
    "bitsandbytes>=0.49.0",
    "accelerate>=1.5.0",
    "datasets>=3.2.0",
    "mlflow>=2.22.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=8.0",
    "ruff>=0.4.0",
    "mypy>=1.10",
]

[dependency-groups]
notebook = ["jupyter>=1.0", "ipywidgets>=8.0"]

[tool.uv]
# CUDA 12.4 PyTorch wheel 사용
index-url = "https://download.pytorch.org/whl/cu124"

[tool.ruff]
line-length = 100
target-version = "py312"
```

## 의존성 관리

```bash
# 런타임 의존성 추가
uv add torch transformers peft

# 개발 의존성 추가
uv add --dev pytest ruff

# 특정 그룹에 추가
uv add --group notebook jupyter

# 의존성 제거
uv remove package-name

# 잠금 파일 동기화
uv sync

# 잠금 파일만 업데이트
uv lock
```

## 실행

```bash
# 스크립트 실행 (자동 환경 활성화)
uv run python train.py

# 또는 환경 활성화 후 실행
source .venv/bin/activate  # Linux/Mac
python train.py
```

## CUDA PyTorch 설정

```toml
# pyproject.toml - CUDA 12.4 버전
[tool.uv.sources]
torch = { index = "pytorch-cu124" }
torchvision = { index = "pytorch-cu124" }

[[tool.uv.index]]
name = "pytorch-cu124"
url = "https://download.pytorch.org/whl/cu124"
explicit = true
```

## 환경 재현

```bash
# 다른 머신에서 동일 환경 구성
git clone <repo>
cd ml-project
uv sync  # uv.lock 기반으로 정확히 동일한 버전 설치
```

## Tips

1. **uv.lock은 반드시 git에 커밋** - 재현성 보장
2. **버전 범위 사용** - `>=2.5.0,<3.0` 형태 권장
3. **Python 버전 고정** - `.python-version` 파일 활용
4. **CI/CD에서도 uv 사용** - `pip install uv && uv sync`

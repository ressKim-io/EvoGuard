#!/usr/bin/env python3
"""
EvoGuard 모델 정리 스크립트

사용법:
    python scripts/cleanup_models.py --dry-run    # 삭제 대상 확인
    python scripts/cleanup_models.py --execute    # 실제 삭제 실행
"""

import argparse
import shutil
from pathlib import Path

# 기준 경로
BASE_DIR = Path(__file__).parent.parent
MODELS_DIR = BASE_DIR / "models"
CHECKPOINTS_DIR = BASE_DIR / "checkpoints"
LOGS_DIR = BASE_DIR / "logs"

# 프로덕션 모델 (절대 삭제 금지)
PROTECTED_MODELS = {
    "phase2-combined",
    "phase2-slang-enhanced",
    "phase4-augmented",
    "coevolution-latest",
}

# 삭제 대상 모델
OBSOLETE_MODELS = [
    "robust-kotox",
    "robust-kotox-v2",
    "robust-kotox-v3",
    "robust-kotox-full",
    "robust-kotox-test",
    "toxic-classifier",
    "coevolution-model",
    "adversarial-retrained",
    "checkpoints",
    "pipeline",
]

# 압축 보관 대상 모델
ARCHIVE_MODELS = [
    "phase1-deobfuscated",
    "phase3-large",
    "phase5-cnn-enhanced",
    "korean-coevolution-model",
]


def get_dir_size(path: Path) -> int:
    """디렉토리 크기 계산 (bytes)"""
    if not path.exists():
        return 0
    total = 0
    for f in path.rglob("*"):
        if f.is_file():
            total += f.stat().st_size
    return total


def format_size(size_bytes: int) -> str:
    """바이트를 사람이 읽기 쉬운 형식으로 변환"""
    for unit in ["B", "KB", "MB", "GB"]:
        if size_bytes < 1024:
            return f"{size_bytes:.1f}{unit}"
        size_bytes /= 1024
    return f"{size_bytes:.1f}TB"


def cleanup_checkpoints(dry_run: bool) -> int:
    """checkpoints/robust 디렉토리 삭제"""
    robust_dir = CHECKPOINTS_DIR / "robust"
    if not robust_dir.exists():
        print("  [SKIP] checkpoints/robust 없음")
        return 0

    size = get_dir_size(robust_dir)
    print(f"  [DELETE] checkpoints/robust/ ({format_size(size)})")

    if not dry_run:
        shutil.rmtree(robust_dir)
        print("    -> 삭제 완료")
    return size


def cleanup_obsolete_models(dry_run: bool) -> int:
    """Obsolete 모델 삭제"""
    total_freed = 0

    for model_name in OBSOLETE_MODELS:
        model_path = MODELS_DIR / model_name
        if not model_path.exists():
            continue

        if model_name in PROTECTED_MODELS:
            print(f"  [PROTECTED] {model_name} - 삭제 금지")
            continue

        size = get_dir_size(model_path)
        print(f"  [DELETE] models/{model_name}/ ({format_size(size)})")

        if not dry_run:
            shutil.rmtree(model_path)
            print("    -> 삭제 완료")
        total_freed += size

    return total_freed


def archive_models(dry_run: bool) -> int:
    """실험 모델 압축 보관"""
    import tarfile

    archive_dir = MODELS_DIR / "archive"
    if not dry_run:
        archive_dir.mkdir(exist_ok=True)

    total_freed = 0

    for model_name in ARCHIVE_MODELS:
        model_path = MODELS_DIR / model_name
        if not model_path.exists():
            print(f"  [SKIP] {model_name} 없음")
            continue

        if model_name in PROTECTED_MODELS:
            print(f"  [PROTECTED] {model_name} - 압축 건너뜀")
            continue

        size = get_dir_size(model_path)
        archive_path = archive_dir / f"{model_name}.tar.gz"

        print(f"  [ARCHIVE] models/{model_name}/ ({format_size(size)}) -> archive/{model_name}.tar.gz")

        if not dry_run:
            # 압축
            with tarfile.open(archive_path, "w:gz") as tar:
                tar.add(model_path, arcname=model_name)
            # 원본 삭제
            shutil.rmtree(model_path)
            compressed_size = archive_path.stat().st_size
            print(f"    -> 압축 완료 ({format_size(compressed_size)})")
            total_freed += size - compressed_size
        else:
            # dry-run: 예상 절약량 (약 70% 압축률 가정)
            total_freed += int(size * 0.7)

    return total_freed


def cleanup_logs(dry_run: bool) -> int:
    """오래된 로그 정리"""
    total_freed = 0

    # 빈 training 디렉토리
    training_logs = LOGS_DIR / "training"
    if training_logs.exists() and not any(training_logs.iterdir()):
        print(f"  [DELETE] logs/training/ (빈 디렉토리)")
        if not dry_run:
            training_logs.rmdir()

    # 루트의 training 로그 파일들
    for log_file in BASE_DIR.glob("training_*.log"):
        size = log_file.stat().st_size
        print(f"  [DELETE] {log_file.name} ({format_size(size)})")
        if not dry_run:
            log_file.unlink()
        total_freed += size

    return total_freed


def create_directory_structure(dry_run: bool):
    """새 디렉토리 구조 생성"""
    new_dirs = [
        MODELS_DIR / "production",
        MODELS_DIR / "coevolution" / "versions",
        MODELS_DIR / "archive",
    ]

    for dir_path in new_dirs:
        if not dir_path.exists():
            print(f"  [CREATE] {dir_path.relative_to(BASE_DIR)}/")
            if not dry_run:
                dir_path.mkdir(parents=True, exist_ok=True)


def main():
    parser = argparse.ArgumentParser(description="EvoGuard 모델 정리 스크립트")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--dry-run", action="store_true", help="삭제 대상만 확인 (실제 삭제 안함)")
    group.add_argument("--execute", action="store_true", help="실제 삭제 실행")
    args = parser.parse_args()

    dry_run = args.dry_run

    print("=" * 60)
    print("EvoGuard 모델 정리")
    print("=" * 60)
    if dry_run:
        print("모드: DRY-RUN (실제 삭제 없음)\n")
    else:
        print("모드: EXECUTE (실제 삭제 실행)\n")

    total_freed = 0

    # Step 1: Checkpoints 삭제
    print("\n[Step 1] Checkpoints 삭제")
    total_freed += cleanup_checkpoints(dry_run)

    # Step 2: Obsolete 모델 삭제
    print("\n[Step 2] Obsolete 모델 삭제")
    total_freed += cleanup_obsolete_models(dry_run)

    # Step 3: 실험 모델 압축
    print("\n[Step 3] 실험 모델 압축 보관")
    total_freed += archive_models(dry_run)

    # Step 4: 로그 정리
    print("\n[Step 4] 로그 정리")
    total_freed += cleanup_logs(dry_run)

    # Step 5: 디렉토리 구조 생성
    print("\n[Step 5] 새 디렉토리 구조 생성")
    create_directory_structure(dry_run)

    # 결과 출력
    print("\n" + "=" * 60)
    print(f"예상 절약 용량: {format_size(total_freed)}")
    print("=" * 60)

    if dry_run:
        print("\n--execute 옵션으로 실제 삭제를 실행하세요.")


if __name__ == "__main__":
    main()

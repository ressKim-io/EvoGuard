#!/usr/bin/env python3
"""
EvoGuard 모델 버전 관리 스크립트

사용법:
    python scripts/model_version_manager.py save              # 현재 latest를 버전으로 저장
    python scripts/model_version_manager.py list              # 저장된 버전 목록
    python scripts/model_version_manager.py prune --keep 3    # 최근 3개만 유지
    python scripts/model_version_manager.py restore <version> # 특정 버전 복원
"""

import argparse
import json
import shutil
from datetime import datetime
from pathlib import Path

# 기준 경로
BASE_DIR = Path(__file__).parent.parent
MODELS_DIR = BASE_DIR / "models"
COEVOLUTION_DIR = MODELS_DIR / "coevolution"
VERSIONS_DIR = COEVOLUTION_DIR / "versions"
LATEST_DIR = MODELS_DIR / "coevolution-latest"
REGISTRY_PATH = MODELS_DIR / "MODEL_REGISTRY.json"


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


def load_registry() -> dict:
    """레지스트리 로드"""
    if REGISTRY_PATH.exists():
        with open(REGISTRY_PATH) as f:
            return json.load(f)
    return {
        "production_models": {
            "phase2-combined": {
                "path": "models/phase2-combined",
                "description": "Phase 2 Combined - 단일 모델 최고 성능",
                "f1_score": 0.9675,
            },
            "phase4-augmented": {
                "path": "models/phase4-augmented",
                "description": "Phase 4 에러 기반 증강 학습",
                "f1_score": 0.9650,
            },
        },
        "coevolution_versions": [],
        "ensemble_config": {
            "models": ["phase2-combined", "coevolution-latest"],
            "method": "AND",
            "f1_score": 0.9696,
        },
    }


def save_registry(registry: dict):
    """레지스트리 저장"""
    REGISTRY_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(REGISTRY_PATH, "w") as f:
        json.dump(registry, f, indent=2, ensure_ascii=False)


def cmd_save(args):
    """현재 coevolution-latest를 버전으로 저장"""
    if not LATEST_DIR.exists():
        print("Error: coevolution-latest 디렉토리가 없습니다.")
        return 1

    # 버전명 생성
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    version_name = f"v_{timestamp}"

    if args.tag:
        version_name = f"v_{timestamp}_{args.tag}"

    version_dir = VERSIONS_DIR / version_name

    # 디렉토리 생성
    VERSIONS_DIR.mkdir(parents=True, exist_ok=True)

    # 복사
    print(f"Saving coevolution-latest -> versions/{version_name}")
    shutil.copytree(LATEST_DIR, version_dir)

    # 메타데이터 저장
    metadata = {
        "version": version_name,
        "created_at": datetime.now().isoformat(),
        "tag": args.tag,
        "size_bytes": get_dir_size(version_dir),
    }

    # config.json에서 성능 정보 읽기
    config_path = version_dir / "config.json"
    if config_path.exists():
        with open(config_path) as f:
            config = json.load(f)
            if "training_metrics" in config:
                metadata["metrics"] = config["training_metrics"]

    metadata_path = version_dir / "version_metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    # 레지스트리 업데이트
    registry = load_registry()
    registry["coevolution_versions"].append({
        "version": version_name,
        "path": f"models/coevolution/versions/{version_name}",
        "created_at": metadata["created_at"],
        "tag": args.tag,
    })
    save_registry(registry)

    print(f"Saved: {version_name} ({format_size(metadata['size_bytes'])})")
    return 0


def cmd_list(args):
    """저장된 버전 목록 출력"""
    if not VERSIONS_DIR.exists():
        print("저장된 버전이 없습니다.")
        return 0

    versions = sorted(VERSIONS_DIR.iterdir(), reverse=True)
    if not versions:
        print("저장된 버전이 없습니다.")
        return 0

    print(f"\n{'Version':<30} {'Size':<10} {'Created':<20} {'Tag':<15}")
    print("-" * 75)

    for version_dir in versions:
        if not version_dir.is_dir():
            continue

        metadata_path = version_dir / "version_metadata.json"
        if metadata_path.exists():
            with open(metadata_path) as f:
                meta = json.load(f)
            created = meta.get("created_at", "")[:19].replace("T", " ")
            tag = meta.get("tag", "") or ""
        else:
            created = ""
            tag = ""

        size = format_size(get_dir_size(version_dir))
        print(f"{version_dir.name:<30} {size:<10} {created:<20} {tag:<15}")

    # 최신 모델 정보
    if LATEST_DIR.exists():
        print(f"\nCurrent latest: {format_size(get_dir_size(LATEST_DIR))}")

    return 0


def cmd_prune(args):
    """오래된 버전 삭제, 최근 N개만 유지"""
    if not VERSIONS_DIR.exists():
        print("저장된 버전이 없습니다.")
        return 0

    versions = sorted(VERSIONS_DIR.iterdir(), reverse=True)
    versions = [v for v in versions if v.is_dir()]

    keep = args.keep
    to_delete = versions[keep:]

    if not to_delete:
        print(f"삭제할 버전이 없습니다. (현재 {len(versions)}개, 유지: {keep}개)")
        return 0

    total_freed = 0
    for version_dir in to_delete:
        size = get_dir_size(version_dir)
        print(f"[DELETE] {version_dir.name} ({format_size(size)})")

        if not args.dry_run:
            shutil.rmtree(version_dir)
            total_freed += size

    if args.dry_run:
        print(f"\n--dry-run 모드: 실제 삭제되지 않음")
    else:
        print(f"\n삭제 완료: {format_size(total_freed)} 절약")

        # 레지스트리 업데이트
        registry = load_registry()
        deleted_names = {v.name for v in to_delete}
        registry["coevolution_versions"] = [
            v for v in registry["coevolution_versions"]
            if v["version"] not in deleted_names
        ]
        save_registry(registry)

    return 0


def cmd_restore(args):
    """특정 버전을 coevolution-latest로 복원"""
    version_dir = VERSIONS_DIR / args.version

    if not version_dir.exists():
        print(f"Error: 버전 '{args.version}'을 찾을 수 없습니다.")
        return 1

    # 현재 latest 백업
    if LATEST_DIR.exists():
        backup_name = f"latest_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        backup_dir = VERSIONS_DIR / backup_name
        print(f"Backing up current latest -> {backup_name}")
        shutil.move(str(LATEST_DIR), str(backup_dir))

    # 복원
    print(f"Restoring {args.version} -> coevolution-latest")
    shutil.copytree(version_dir, LATEST_DIR)
    print("복원 완료")

    return 0


def main():
    parser = argparse.ArgumentParser(description="EvoGuard 모델 버전 관리")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # save 명령
    save_parser = subparsers.add_parser("save", help="현재 latest를 버전으로 저장")
    save_parser.add_argument("--tag", "-t", help="버전 태그 (예: best, stable)")

    # list 명령
    subparsers.add_parser("list", help="저장된 버전 목록")

    # prune 명령
    prune_parser = subparsers.add_parser("prune", help="오래된 버전 삭제")
    prune_parser.add_argument("--keep", "-k", type=int, default=3, help="유지할 버전 수 (기본: 3)")
    prune_parser.add_argument("--dry-run", action="store_true", help="실제 삭제 없이 확인만")

    # restore 명령
    restore_parser = subparsers.add_parser("restore", help="특정 버전 복원")
    restore_parser.add_argument("version", help="복원할 버전명")

    args = parser.parse_args()

    if args.command == "save":
        return cmd_save(args)
    elif args.command == "list":
        return cmd_list(args)
    elif args.command == "prune":
        return cmd_prune(args)
    elif args.command == "restore":
        return cmd_restore(args)


if __name__ == "__main__":
    exit(main())

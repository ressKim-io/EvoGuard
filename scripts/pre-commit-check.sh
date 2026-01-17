#!/bin/bash
# Pre-commit check script for Claude Code hooks
# Runs lint and test before git commit

TOOL_INPUT="$1"

# Only run checks if this is a git commit command
if [[ ! "$TOOL_INPUT" =~ "git commit" ]]; then
    exit 0
fi

echo "=== Pre-commit checks ==="

PROJECT_ROOT="/Users/ress/my-file/project/EvoGuard"
FAILED=0

# Go checks (if api-service exists)
if [ -d "$PROJECT_ROOT/api-service" ]; then
    echo "[Go] Running golangci-lint..."
    cd "$PROJECT_ROOT/api-service"
    if command -v golangci-lint &> /dev/null; then
        golangci-lint run ./... || FAILED=1
    else
        echo "[Go] golangci-lint not installed, skipping..."
    fi

    echo "[Go] Running tests..."
    go test ./... -v || FAILED=1
fi

# Python checks (if ml-service exists)
if [ -d "$PROJECT_ROOT/ml-service" ]; then
    echo "[Python] Running ruff..."
    cd "$PROJECT_ROOT/ml-service"
    if command -v ruff &> /dev/null; then
        ruff check . || FAILED=1
    else
        echo "[Python] ruff not installed, skipping..."
    fi

    echo "[Python] Running pytest..."
    if command -v pytest &> /dev/null; then
        pytest -v || FAILED=1
    else
        echo "[Python] pytest not installed, skipping..."
    fi
fi

# Additional Python modules
for module in attacker defender training mlops; do
    if [ -d "$PROJECT_ROOT/$module" ]; then
        echo "[Python:$module] Running ruff..."
        cd "$PROJECT_ROOT/$module"
        if command -v ruff &> /dev/null; then
            ruff check . || FAILED=1
        fi
    fi
done

if [ $FAILED -ne 0 ]; then
    echo "=== Pre-commit checks FAILED ==="
    exit 1
fi

echo "=== Pre-commit checks PASSED ==="
exit 0

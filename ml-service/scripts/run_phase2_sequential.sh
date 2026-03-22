#!/bin/bash
# Phase 2: Sequential Full Training (SSH-safe via nohup)
#
# Usage:
#   cd ml-service
#   nohup bash scripts/run_phase2_sequential.sh > logs/phase2_$(date +%Y%m%d_%H%M%S).log 2>&1 &
#   tail -f logs/phase2_*.log  # monitor progress

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

# Activate venv
source .venv/bin/activate

# Create logs directory
mkdir -p logs

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RESULTS_FILE="logs/phase2_results_${TIMESTAMP}.json"

echo "============================================"
echo "Phase 2: Sequential Full Training"
echo "Started: $(date)"
echo "Results: $RESULTS_FILE"
echo "============================================"

# Initialize results
echo '{"phase2_results": {}, "started": "'$(date -Iseconds)'"}' > "$RESULTS_FILE"

# =============================================
# Task 2.1: LAHN 10-epoch Full Training
# =============================================
echo ""
echo "============================================"
echo "[1/3] LAHN 10-epoch Full Training"
echo "Started: $(date)"
echo "Expected: ~45 min"
echo "============================================"

python scripts/train_with_lahn.py --epochs 10 2>&1 | tee logs/lahn_${TIMESTAMP}.log

LAHN_EXIT=$?
if [ $LAHN_EXIT -eq 0 ]; then
    echo "[1/3] LAHN training COMPLETED at $(date)"
else
    echo "[1/3] LAHN training FAILED (exit=$LAHN_EXIT) at $(date)"
fi

# =============================================
# Task 2.2: DeBERTa 10-epoch Full Training
# =============================================
echo ""
echo "============================================"
echo "[2/3] DeBERTa 10-epoch Full Training"
echo "Started: $(date)"
echo "Expected: ~60 min"
echo "============================================"

python scripts/train_deberta.py --epochs 10 2>&1 | tee logs/deberta_${TIMESTAMP}.log

DEBERTA_EXIT=$?
if [ $DEBERTA_EXIT -eq 0 ]; then
    echo "[2/3] DeBERTa training COMPLETED at $(date)"
else
    echo "[2/3] DeBERTa training FAILED (exit=$DEBERTA_EXIT) at $(date)"
fi

# =============================================
# Task 2.3: EXAONE 4.0 1.2B QLoRA Training
# =============================================
echo ""
echo "============================================"
echo "[3/3] EXAONE 4.0 1.2B QLoRA Training"
echo "Started: $(date)"
echo "Expected: ~2-3 hours"
echo "============================================"

python scripts/train_exaone.py --epochs 5 2>&1 | tee logs/exaone_${TIMESTAMP}.log

EXAONE_EXIT=$?
if [ $EXAONE_EXIT -eq 0 ]; then
    echo "[3/3] EXAONE training COMPLETED at $(date)"
else
    echo "[3/3] EXAONE training FAILED (exit=$EXAONE_EXIT) at $(date)"
fi

# =============================================
# Summary
# =============================================
echo ""
echo "============================================"
echo "Phase 2 Complete!"
echo "Finished: $(date)"
echo "============================================"
echo "Results:"
echo "  LAHN:    exit=$LAHN_EXIT"
echo "  DeBERTa: exit=$DEBERTA_EXIT"
echo "  EXAONE:  exit=$EXAONE_EXIT"
echo ""
echo "Check individual logs:"
echo "  logs/lahn_${TIMESTAMP}.log"
echo "  logs/deberta_${TIMESTAMP}.log"
echo "  logs/exaone_${TIMESTAMP}.log"
echo ""
echo "Models saved to:"
echo "  models/lahn/       (LAHN)"
echo "  models/pmf/deberta/ (DeBERTa)"
echo "  models/pmf/exaone/  (EXAONE)"

#!/bin/bash
# Run all training phases sequentially

cd /home/resshome/project/EvoGuard/ml-service
source .venv-robust/bin/activate

echo "============================================================"
echo "STARTING ALL PHASES - $(date)"
echo "============================================================"

# Phase 1: Deobfuscation
echo ""
echo "[Phase 1/3] Deobfuscation Training..."
python scripts/phase1_deobfuscation.py --epochs 15 --batch_size 16 2>&1 | tee logs/phase1.log

# Phase 2: Combined Data
echo ""
echo "[Phase 2/3] Combined Data Training..."
python scripts/phase2_combined_data.py --epochs 10 --batch_size 16 2>&1 | tee logs/phase2.log

# Phase 3: Large Model
echo ""
echo "[Phase 3/3] Large Model Training..."
python scripts/phase3_large_model.py --epochs 10 --batch_size 8 2>&1 | tee logs/phase3.log

echo ""
echo "============================================================"
echo "ALL PHASES COMPLETE - $(date)"
echo "============================================================"

# Summary
echo ""
echo "=== RESULTS SUMMARY ==="
echo "Phase 1 (Deobfuscation):"
cat models/phase1-deobfuscated/results.json 2>/dev/null || echo "Not found"
echo ""
echo "Phase 2 (Combined Data):"
cat models/phase2-combined/results.json 2>/dev/null || echo "Not found"
echo ""
echo "Phase 3 (Large Model):"
cat models/phase3-large/results.json 2>/dev/null || echo "Not found"

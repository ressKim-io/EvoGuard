#!/bin/bash
# 학습 완료 대기 스크립트

while pgrep -f "train_multi_model" > /dev/null; do
    sleep 60
done

echo "=== PMF 학습 완료 ==="
date
echo ""
grep -E "Training complete|TRAINING SUMMARY" /home/resshome/project/EvoGuard/ml-service/logs/pmf_training_*.log | tail -10

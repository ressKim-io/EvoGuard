#!/bin/bash
# PMF 학습 상태 확인 스크립트

LOG=$(ls -t /home/resshome/project/EvoGuard/ml-service/logs/pmf_training_*.log 2>/dev/null | head -1)

if [ -z "$LOG" ]; then
    echo "로그 파일 없음"
    exit 1
fi

echo "=== PMF Training Status ==="
echo "Log: $LOG"
echo ""

# 프로세스 확인
if pgrep -f "train_multi_model" > /dev/null; then
    echo "Status: 🔄 실행 중"
    ps aux | grep train_multi_model | grep -v grep | awk '{print "PID:", $2, "CPU:", $3"%", "MEM:", $4"%"}'
else
    echo "Status: ✅ 완료 (또는 중단됨)"
fi

echo ""
echo "=== 최근 진행 상황 ==="
grep -E "Training:|complete|Best|Epoch [0-9]+:" "$LOG" | grep -v "0%\|1%\|2%" | tail -15

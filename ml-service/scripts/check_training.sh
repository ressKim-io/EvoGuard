#!/bin/bash
# =============================================================================
# check_training.sh - 학습 상태 확인 스크립트
# =============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
LOG_DIR="$PROJECT_DIR/logs"

# 색상 정의
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${BLUE}==============================================================================${NC}"
echo -e "${GREEN}EvoGuard 학습 상태 확인${NC}"
echo -e "${BLUE}==============================================================================${NC}"
echo ""

# 1. tmux 세션 확인
echo -e "${YELLOW}[tmux 세션]${NC}"
if command -v tmux &> /dev/null; then
    SESSIONS=$(tmux ls 2>/dev/null)
    if [ -n "$SESSIONS" ]; then
        echo "$SESSIONS"
        echo ""
        echo -e "연결하려면: ${GREEN}tmux attach -t <세션이름>${NC}"
    else
        echo "실행 중인 tmux 세션 없음"
    fi
else
    echo "tmux 미설치"
fi
echo ""

# 2. 백그라운드 학습 프로세스 확인
echo -e "${YELLOW}[백그라운드 프로세스]${NC}"
TRAINING_PROCS=$(ps aux | grep -E "python.*(train|phase|coevolution)" | grep -v grep)
if [ -n "$TRAINING_PROCS" ]; then
    echo "$TRAINING_PROCS"
else
    echo "실행 중인 학습 프로세스 없음"
fi
echo ""

# 3. 마지막 학습 정보
echo -e "${YELLOW}[마지막 백그라운드 학습]${NC}"
if [ -f "$LOG_DIR/.last_training_pid" ]; then
    LAST_PID=$(cat "$LOG_DIR/.last_training_pid")
    LAST_LOG=$(cat "$LOG_DIR/.last_training_log" 2>/dev/null)

    if ps -p "$LAST_PID" > /dev/null 2>&1; then
        echo -e "상태: ${GREEN}실행 중${NC} (PID: $LAST_PID)"
    else
        echo -e "상태: ${RED}종료됨${NC} (PID: $LAST_PID)"
    fi
    echo "로그: $LAST_LOG"
else
    echo "기록 없음"
fi
echo ""

# 4. 최근 로그 파일
echo -e "${YELLOW}[최근 로그 파일 (최근 5개)]${NC}"
if [ -d "$LOG_DIR" ]; then
    ls -lt "$LOG_DIR"/*.log 2>/dev/null | head -5 | awk '{print $NF}'
    echo ""
    echo -e "로그 보기: ${GREEN}tail -f <로그파일>${NC}"
else
    echo "로그 디렉토리 없음"
fi
echo ""

# 5. GPU 상태 (있는 경우)
echo -e "${YELLOW}[GPU 상태]${NC}"
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi --query-gpu=name,utilization.gpu,memory.used,memory.total --format=csv,noheader 2>/dev/null || echo "GPU 정보 조회 실패"
else
    echo "nvidia-smi 미설치 또는 GPU 없음"
fi
echo ""

echo -e "${BLUE}==============================================================================${NC}"

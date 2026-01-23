#!/bin/bash
# =============================================================================
# run_training.sh - SSH 연결이 끊겨도 학습이 계속되도록 실행하는 wrapper 스크립트
# =============================================================================
#
# 사용법:
#   ./run_training.sh <script.py> [args...]     # tmux 세션에서 실행 (기본값)
#   ./run_training.sh -b <script.py> [args...]  # nohup 백그라운드 실행
#   ./run_training.sh -f <script.py> [args...]  # foreground 실행 (테스트용)
#
# 예시:
#   ./run_training.sh phase1_deobfuscation.py --epochs 15
#   ./run_training.sh -b run_robust_training.py --epochs 10
#   ./run_training.sh run_all_phases.sh
#
# tmux 세션 관리:
#   tmux ls                    # 실행 중인 세션 목록
#   tmux attach -t training    # 세션에 다시 연결
#   Ctrl+B, D                  # 세션에서 분리 (detach)
#
# =============================================================================

set -e

# 색상 정의
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 기본 설정
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
LOG_DIR="$PROJECT_DIR/logs"
VENV_PATH="$PROJECT_DIR/.venv-robust"
MODE="tmux"  # 기본값: tmux
SESSION_NAME="training"

# 도움말
show_help() {
    echo -e "${BLUE}==============================================================================${NC}"
    echo -e "${GREEN}run_training.sh${NC} - SSH 연결이 끊겨도 학습이 계속되는 실행 스크립트"
    echo -e "${BLUE}==============================================================================${NC}"
    echo ""
    echo -e "${YELLOW}사용법:${NC}"
    echo "  ./run_training.sh [옵션] <스크립트> [인자...]"
    echo ""
    echo -e "${YELLOW}옵션:${NC}"
    echo "  -t, --tmux       tmux 세션에서 실행 (기본값, 추천)"
    echo "  -b, --background nohup으로 백그라운드 실행"
    echo "  -f, --foreground foreground에서 실행 (테스트용)"
    echo "  -s, --session    tmux 세션 이름 지정 (기본값: training)"
    echo "  -h, --help       도움말 표시"
    echo ""
    echo -e "${YELLOW}예시:${NC}"
    echo "  ./run_training.sh phase1_deobfuscation.py --epochs 15"
    echo "  ./run_training.sh -b run_robust_training.py --epochs 10"
    echo "  ./run_training.sh -s phase1 phase1_deobfuscation.py"
    echo ""
    echo -e "${YELLOW}tmux 명령어:${NC}"
    echo "  tmux ls                    # 실행 중인 세션 목록"
    echo "  tmux attach -t training    # 세션에 다시 연결"
    echo "  Ctrl+B, D                  # 세션에서 분리"
    echo ""
}

# 로그 디렉토리 생성
mkdir -p "$LOG_DIR"

# 인자 파싱
while [[ $# -gt 0 ]]; do
    case $1 in
        -t|--tmux)
            MODE="tmux"
            shift
            ;;
        -b|--background)
            MODE="background"
            shift
            ;;
        -f|--foreground)
            MODE="foreground"
            shift
            ;;
        -s|--session)
            SESSION_NAME="$2"
            shift 2
            ;;
        -h|--help)
            show_help
            exit 0
            ;;
        *)
            break
            ;;
    esac
done

# 스크립트 인자 확인
if [ $# -eq 0 ]; then
    echo -e "${RED}오류: 실행할 스크립트를 지정해주세요.${NC}"
    echo ""
    show_help
    exit 1
fi

SCRIPT="$1"
shift
ARGS="$@"

# 스크립트 경로 처리
if [[ ! "$SCRIPT" == /* ]]; then
    # 상대 경로인 경우
    if [[ -f "$SCRIPT_DIR/$SCRIPT" ]]; then
        SCRIPT="$SCRIPT_DIR/$SCRIPT"
    elif [[ -f "$PROJECT_DIR/$SCRIPT" ]]; then
        SCRIPT="$PROJECT_DIR/$SCRIPT"
    elif [[ -f "$SCRIPT" ]]; then
        SCRIPT="$(pwd)/$SCRIPT"
    else
        echo -e "${RED}오류: 스크립트를 찾을 수 없습니다: $SCRIPT${NC}"
        exit 1
    fi
fi

# 스크립트 존재 확인
if [[ ! -f "$SCRIPT" ]]; then
    echo -e "${RED}오류: 스크립트를 찾을 수 없습니다: $SCRIPT${NC}"
    exit 1
fi

# 로그 파일명 생성
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
SCRIPT_NAME=$(basename "$SCRIPT" | sed 's/\.[^.]*$//')
LOG_FILE="$LOG_DIR/${SCRIPT_NAME}_${TIMESTAMP}.log"

# 실행 명령 구성
if [[ "$SCRIPT" == *.py ]]; then
    # Python 스크립트
    RUN_CMD="cd $PROJECT_DIR && source $VENV_PATH/bin/activate && python $SCRIPT $ARGS"
elif [[ "$SCRIPT" == *.sh ]]; then
    # Shell 스크립트
    RUN_CMD="bash $SCRIPT $ARGS"
else
    echo -e "${RED}오류: 지원하지 않는 스크립트 형식입니다: $SCRIPT${NC}"
    exit 1
fi

echo -e "${BLUE}==============================================================================${NC}"
echo -e "${GREEN}EvoGuard 학습 실행기${NC}"
echo -e "${BLUE}==============================================================================${NC}"
echo -e "스크립트: ${YELLOW}$SCRIPT${NC}"
echo -e "인자:     ${YELLOW}$ARGS${NC}"
echo -e "모드:     ${YELLOW}$MODE${NC}"
echo -e "로그:     ${YELLOW}$LOG_FILE${NC}"
echo -e "${BLUE}==============================================================================${NC}"
echo ""

case $MODE in
    tmux)
        # tmux 설치 확인
        if ! command -v tmux &> /dev/null; then
            echo -e "${YELLOW}경고: tmux가 설치되어 있지 않습니다. nohup 모드로 전환합니다.${NC}"
            MODE="background"
        else
            # 기존 세션 확인
            if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
                echo -e "${YELLOW}기존 '$SESSION_NAME' 세션이 있습니다. 연결하시겠습니까? [y/N]${NC}"
                read -r response
                if [[ "$response" =~ ^[Yy]$ ]]; then
                    tmux attach -t "$SESSION_NAME"
                    exit 0
                else
                    echo -e "${YELLOW}새 세션 이름을 입력하세요 (기본값: ${SESSION_NAME}_${TIMESTAMP}):${NC}"
                    read -r new_name
                    SESSION_NAME="${new_name:-${SESSION_NAME}_${TIMESTAMP}}"
                fi
            fi

            echo -e "${GREEN}tmux 세션 '$SESSION_NAME'에서 학습을 시작합니다...${NC}"
            echo ""
            echo -e "${YELLOW}세션에 연결하려면: tmux attach -t $SESSION_NAME${NC}"
            echo -e "${YELLOW}세션에서 분리하려면: Ctrl+B, D${NC}"
            echo ""

            # tmux 세션 생성 및 실행
            tmux new-session -d -s "$SESSION_NAME" "bash -c '$RUN_CMD 2>&1 | tee $LOG_FILE; echo; echo 학습 완료! 아무 키나 누르면 종료됩니다.; read'"

            sleep 1
            echo -e "${GREEN}학습이 시작되었습니다!${NC}"
            echo ""
            echo -e "지금 세션에 연결하시겠습니까? [Y/n]"
            read -r attach_response
            if [[ ! "$attach_response" =~ ^[Nn]$ ]]; then
                tmux attach -t "$SESSION_NAME"
            fi
            exit 0
        fi
        ;;
esac

case $MODE in
    background)
        echo -e "${GREEN}백그라운드에서 학습을 시작합니다...${NC}"
        echo ""

        # nohup으로 실행
        nohup bash -c "$RUN_CMD" > "$LOG_FILE" 2>&1 &
        PID=$!

        echo -e "PID: ${YELLOW}$PID${NC}"
        echo -e "로그 확인: ${YELLOW}tail -f $LOG_FILE${NC}"
        echo -e "프로세스 확인: ${YELLOW}ps aux | grep $PID${NC}"
        echo -e "중지: ${YELLOW}kill $PID${NC}"
        echo ""

        # PID 저장
        echo "$PID" > "$LOG_DIR/.last_training_pid"
        echo "$LOG_FILE" > "$LOG_DIR/.last_training_log"

        echo -e "${GREEN}학습이 백그라운드에서 실행 중입니다!${NC}"
        ;;

    foreground)
        echo -e "${GREEN}Foreground에서 학습을 시작합니다...${NC}"
        echo ""
        bash -c "$RUN_CMD" 2>&1 | tee "$LOG_FILE"
        ;;
esac

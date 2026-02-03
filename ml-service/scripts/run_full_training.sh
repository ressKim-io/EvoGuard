#!/bin/bash
# PMF 앙상블 + 균형 공진화 순차 학습 스크립트
# 예상 시간: 4~7시간

set -e  # 에러 발생 시 중단

cd /home/resshome/project/EvoGuard/ml-service
source .venv/bin/activate

LOG_DIR="logs"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
MAIN_LOG="${LOG_DIR}/full_training_${TIMESTAMP}.log"

mkdir -p $LOG_DIR

echo "============================================" | tee -a $MAIN_LOG
echo "FULL TRAINING STARTED: $(date)" | tee -a $MAIN_LOG
echo "============================================" | tee -a $MAIN_LOG

# GPU 상태 기록
nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv | tee -a $MAIN_LOG

# ============================================
# STEP 1: PMF 멀티 모델 학습 (3~6시간)
# ============================================
echo "" | tee -a $MAIN_LOG
echo "============================================" | tee -a $MAIN_LOG
echo "STEP 1/3: PMF Multi-Model Training" | tee -a $MAIN_LOG
echo "Started: $(date)" | tee -a $MAIN_LOG
echo "============================================" | tee -a $MAIN_LOG

python scripts/train_multi_model.py --epochs 10 --batch_size 16 2>&1 | tee -a $MAIN_LOG

echo "STEP 1 COMPLETED: $(date)" | tee -a $MAIN_LOG

# ============================================
# STEP 2: 메타러너 학습 (수 분)
# ============================================
echo "" | tee -a $MAIN_LOG
echo "============================================" | tee -a $MAIN_LOG
echo "STEP 2/3: Meta-Learner Training" | tee -a $MAIN_LOG
echo "Started: $(date)" | tee -a $MAIN_LOG
echo "============================================" | tee -a $MAIN_LOG

python scripts/train_meta_learner.py 2>&1 | tee -a $MAIN_LOG

echo "STEP 2 COMPLETED: $(date)" | tee -a $MAIN_LOG

# ============================================
# STEP 3: 균형 공진화 (50분)
# ============================================
echo "" | tee -a $MAIN_LOG
echo "============================================" | tee -a $MAIN_LOG
echo "STEP 3/3: Balanced Coevolution (100 cycles)" | tee -a $MAIN_LOG
echo "Started: $(date)" | tee -a $MAIN_LOG
echo "============================================" | tee -a $MAIN_LOG

python scripts/run_balanced_coevolution.py --max-cycles 100 2>&1 | tee -a $MAIN_LOG

echo "STEP 3 COMPLETED: $(date)" | tee -a $MAIN_LOG

# ============================================
# 완료
# ============================================
echo "" | tee -a $MAIN_LOG
echo "============================================" | tee -a $MAIN_LOG
echo "FULL TRAINING COMPLETED: $(date)" | tee -a $MAIN_LOG
echo "============================================" | tee -a $MAIN_LOG

# 최종 GPU 상태
nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv | tee -a $MAIN_LOG

echo "Log saved to: $MAIN_LOG"

---
description: 학습 전 환경 점검 (GPU, 디스크, 데이터, 프로세스)
---

# 학습 전 환경 점검

아래 항목을 모두 점검하고 결과를 표로 정리해주세요.

## 1. GPU 상태
```bash
nvidia-smi --query-gpu=name,memory.total,memory.used,memory.free,temperature.gpu,utilization.gpu --format=csv,noheader,nounits
```
- VRAM 여유 확인 (학습에 최소 6GB 필요)
- 다른 프로세스가 GPU 점유 중인지 확인

## 2. 디스크 공간
```bash
df -h /home/resshome/project/EvoGuard/ml-service/models/
du -sh /home/resshome/project/EvoGuard/ml-service/models/*/
```
- 최소 10GB 여유 공간 필요

## 3. 실행 중인 학습 프로세스
```bash
ps aux | grep -E "python.*train|python.*pipeline|python.*coevolution" | grep -v grep
```
- 충돌 방지: 이미 학습 중이면 경고

## 4. 데이터셋 검증
```bash
cd /home/resshome/project/EvoGuard/ml-service
wc -l data/korean/standard_v1/*.csv 2>/dev/null
ls -la data/korean/standard_v4_filtered/*.csv 2>/dev/null
```
- train/valid/test 파일 존재 확인
- 행 수 일관성 확인

## 5. 현재 최고 성능 모델 확인
```bash
cat /home/resshome/project/EvoGuard/ml-service/models/pipeline_12h/pipeline_results.json 2>/dev/null | python3 -m json.tool | grep -E "best_f1|final_f1"
```

## 6. Python 환경
```bash
cd /home/resshome/project/EvoGuard/ml-service && source .venv/bin/activate && python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}, Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"
```

## 결과 형식
| 항목 | 상태 | 상세 |
|------|------|------|
| GPU | OK/WARNING/ERROR | ... |
| 디스크 | OK/WARNING | ... |
| 프로세스 | OK/WARNING | ... |
| 데이터셋 | OK/ERROR | ... |
| 최고 성능 | F1=X.XXXX | ... |
| Python 환경 | OK/ERROR | ... |

WARNING이나 ERROR가 있으면 해결 방법도 제시해주세요.

$ARGUMENTS

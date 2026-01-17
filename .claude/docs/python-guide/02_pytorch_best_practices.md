# 02. PyTorch 2.5 Best Practices

## torch.compile() - 핵심 최적화

PyTorch 2.0+의 가장 중요한 기능입니다. Python 3.12 완벽 지원.

```python
import torch

model = MyModel()

# 기본 사용 - 2-3x 속도 향상
model = torch.compile(model)

# 모드 선택
model = torch.compile(model, mode="reduce-overhead")  # 추론 최적화
model = torch.compile(model, mode="max-autotune")     # 최대 성능 (컴파일 느림)
model = torch.compile(model, mode="default")          # 균형
```

## Device 관리

```python
# 권장: 명시적 device 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 텐서 생성 시 device 직접 지정 (복사 방지)
# ❌ Bad
x = torch.rand(1000, 1000).cuda()

# ✅ Good
x = torch.rand(1000, 1000, device=device)
```

## Mixed Precision (AMP)

```python
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

for batch in dataloader:
    optimizer.zero_grad()
    
    with autocast(dtype=torch.bfloat16):  # bfloat16 권장
        output = model(batch)
        loss = criterion(output, target)
    
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
```

## 메모리 최적화

```python
# 1. Gradient Checkpointing
from torch.utils.checkpoint import checkpoint

class Model(nn.Module):
    def forward(self, x):
        # 메모리 절약, 연산 증가
        x = checkpoint(self.layer1, x, use_reentrant=False)
        x = checkpoint(self.layer2, x, use_reentrant=False)
        return x

# 2. zero_grad 최적화
optimizer.zero_grad(set_to_none=True)  # 메모리 절약

# 3. 추론 시 gradient 비활성화
with torch.no_grad():
    output = model(input)

# 4. inference_mode (더 엄격한 no_grad)
with torch.inference_mode():
    output = model(input)
```

## DataLoader 최적화

```python
from torch.utils.data import DataLoader

loader = DataLoader(
    dataset,
    batch_size=32,
    shuffle=True,
    num_workers=4,           # CPU 코어 수에 맞게
    pin_memory=True,         # GPU 전송 가속
    persistent_workers=True, # worker 재사용
    prefetch_factor=2,       # 미리 로드할 배치 수
)
```

## 분산 학습 (DDP)

```python
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

# 초기화
dist.init_process_group(backend="nccl")
local_rank = int(os.environ["LOCAL_RANK"])
torch.cuda.set_device(local_rank)

# 모델 래핑
model = model.to(local_rank)
model = DDP(model, device_ids=[local_rank])

# 실행: torchrun --nproc_per_node=4 train.py
```

## Gradient Accumulation

```python
accumulation_steps = 4

for i, batch in enumerate(dataloader):
    with autocast(dtype=torch.bfloat16):
        loss = model(batch) / accumulation_steps
    
    loss.backward()
    
    if (i + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
```

## 성능 프로파일링

```python
from torch.profiler import profile, ProfilerActivity

with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    record_shapes=True,
    profile_memory=True,
) as prof:
    model(input)

print(prof.key_averages().table(sort_by="cuda_time_total"))
```

## 체크포인트 저장/로드

```python
# 저장
torch.save({
    'epoch': epoch,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'loss': loss,
}, 'checkpoint.pt')

# 로드
checkpoint = torch.load('checkpoint.pt', weights_only=True)  # 보안 권장
model.load_state_dict(checkpoint['model_state_dict'])
```

## 주의사항

- `torch.compile()`은 첫 실행 시 컴파일 시간 필요
- `device_map="auto"`는 transformers 추론 전용
- CUDA 12.4 사용 시 PyTorch 2.5+ 필수
- bfloat16은 Ampere(30XX) 이상 GPU 권장

#!/usr/bin/env python3
import time
import argparse
import os
import torch
import psutil
import torch.nn.functional as F
from torch.optim import Adam
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, CPUOffload
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from datasets import load_dataset
import torch.distributed as dist
from pynvml import nvmlInit, nvmlDeviceGetHandleByIndex, nvmlDeviceGetUtilizationRates

# NVML 초기화 (GPU 부하 측정)
nvmlInit()
def get_load():
    cpu = psutil.cpu_percent(interval=None)
    handles = [nvmlDeviceGetHandleByIndex(i) for i in range(torch.cuda.device_count())]
    gpu = [nvmlDeviceGetUtilizationRates(h).gpu for h in handles]
    return cpu, gpu

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=4)
    args = parser.parse_args()

    # 분산 초기화
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    torch.cuda.set_device(rank)

    # 모델 불러와 CPUOffload FSDP 래핑
    base = GPT2LMHeadModel.from_pretrained("gpt2")
    fsdp_model = FSDP(
        base,
        cpu_offload=CPUOffload(offload_params=True),
        sharding_strategy=torch.distributed.fsdp.ShardingStrategy.FULL_SHARD,
    ).to(rank)

    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    optimizer = Adam(fsdp_model.parameters(), lr=1e-5)

    # 데이터 준비 (max_length 제한)
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    texts = [t for t in ds["text"] if t and t.strip()]
    batches = []
    for i in range(0, len(texts), args.batch_size):
        chunk = texts[i : i + args.batch_size]
        enc = tokenizer(
            chunk,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=128
        )
        batches.append(enc.input_ids.to(rank))
        if len(batches) >= 10:  # 예시: 10 스텝만
            break

    if rank == 0:
        t_all = time.time()

    # 학습 루프
    for step, input_ids in enumerate(batches):
        optimizer.zero_grad()
        outputs = fsdp_model(input_ids, labels=input_ids)
        loss = outputs.loss
        loss.backward()
        optimizer.step()

        cpu, gpu = get_load()
        if rank == 0:
            print(f"Step {step}: loss={loss.item():.4f}, CPU={cpu:.1f}%, GPU={gpu}")

    if rank == 0:
        print(f"Total time: {time.time() - t_all:.3f}s")

if __name__ == "__main__":
    main()

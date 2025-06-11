#!/usr/bin/env python3
import time
import argparse
import os
import torch
import psutil
import torch.nn.functional as F
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from datasets import load_dataset
from pynvml import nvmlInit, nvmlDeviceGetHandleByIndex, nvmlDeviceGetUtilizationRates

# NVML 초기화
nvmlInit()

def get_load():
    cpu = psutil.cpu_percent(interval=None)
    handles = [nvmlDeviceGetHandleByIndex(i) for i in range(torch.cuda.device_count())]
    gpu = [nvmlDeviceGetUtilizationRates(h).gpu for h in handles]
    return cpu, gpu

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size",   type=int,   default=1)
    parser.add_argument("--total_steps",  type=int,   default=100)
    parser.add_argument("--ratio1",       type=float, default=0.1,
                        help="GPU1에 할당할 전체 스텝 비율 (예: 0.1)")
    args = parser.parse_args()

    # 1) 데이터셋 간단히 준비
    os.environ["HF_DATASETS_CACHE"] = "/mnt/d/ai-data/cache/datasets"
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    texts = [t for t in ds["text"] if t and t.strip()]
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    # 2) 전체 마이크로배치 리스트 생성
    batches = []
    idx = 0
    while len(batches) < args.total_steps:
        txt = texts[idx % len(texts)]
        enc = tokenizer([txt], return_tensors="pt",
                        padding="max_length", truncation=True,
                        max_length=50)
        batches.append(enc.input_ids)
        idx += 1

    # 3) 스케줄: GPU1에는 총_steps * ratio1 개, 나머지는 GPU0
    N = args.total_steps
    n1 = int(N * args.ratio1)
    # 균등하게 뽑기: linspace → 정수 인덱스
    schedule1 = set([int(i) for i in torch.linspace(0, N-1, n1).tolist()])

    # 4) 모델 복제: 각 GPU에 올리기
    model0 = GPT2LMHeadModel.from_pretrained("gpt2").eval().to("cuda:0")
    model1 = GPT2LMHeadModel.from_pretrained("gpt2").eval().to("cuda:1")

    # 5) 벤치마크 루프
    t_all = time.time()
    for step, input_ids in enumerate(batches):
        dev = 1 if step in schedule1 else 0
        device = f"cuda:{dev}"

        x = input_ids.to(device)
        cpu_before, gpu_before = get_load()
        t0 = time.time()

        # 순전파만
        with torch.no_grad():
            _ = (model1 if dev==1 else model0)(x)

        dt = time.time() - t0
        cpu_after, gpu_after = get_load()
        print(f"Step {step:3d} → GPU{dev}: time={dt:.3f}s, CPU={cpu_after:.1f}%, GPU={gpu_after}")

    print(f"TOTAL: {time.time() - t_all:.3f}s")

if __name__ == "__main__":
    main()

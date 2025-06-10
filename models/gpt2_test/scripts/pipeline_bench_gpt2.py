#!/usr/bin/env python3
import time
import argparse
import os
import torch
import psutil
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from datasets import load_dataset
from pynvml import nvmlInit, nvmlDeviceGetHandleByIndex, nvmlDeviceGetUtilizationRates

# NVML 초기화 (GPU 부하 측정)
nvmlInit()

def get_load():
    cpu = psutil.cpu_percent(interval=None)
    count = torch.cuda.device_count()
    handles = [nvmlDeviceGetHandleByIndex(i) for i in range(count)]
    gpu = [nvmlDeviceGetUtilizationRates(h).gpu for h in handles]
    return cpu, gpu

def get_batches(batch_size, max_steps=5):
    os.environ["HF_DATASETS_CACHE"] = "/mnt/d/ai-data/cache/datasets"
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    texts = [t for t in ds["text"] if t and t.strip()]
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    batches = []
    for i in range(0, len(texts), batch_size):
        chunk = texts[i : i + batch_size]
        enc = tokenizer(chunk, return_tensors="pt", padding=True, truncation=True)
        batches.append(enc.input_ids)  # CPU LongTensor
        if len(batches) >= max_steps:
            break
    return batches, tokenizer

class BlockWrapper(nn.Module):
    """Wrap GPT2Block to return only hidden_states."""
    def __init__(self, block):
        super().__init__()
        self.block = block
    def forward(self, x):
        hidden, *_ = self.block(x, use_cache=False)
        return hidden

class Stage1(nn.Module):
    def __init__(self, wte, wpe, blocks):
        super().__init__()
        self.wte = wte
        self.wpe = wpe
        self.blocks = nn.Sequential(*[BlockWrapper(b) for b in blocks])
    def forward(self, input_ids):
        input_ids = input_ids.long()
        token_emb = self.wte(input_ids)
        seq_len = input_ids.size(1)
        pos_ids = torch.arange(seq_len, device=input_ids.device)\
                       .unsqueeze(0).expand_as(input_ids)
        pos_emb = self.wpe(pos_ids)
        x = token_emb + pos_emb
        return self.blocks(x)

class Stage2(nn.Module):
    def __init__(self, blocks, ln_f, lm_head):
        super().__init__()
        self.blocks = nn.Sequential(*[BlockWrapper(b) for b in blocks])
        self.ln_f = ln_f
        self.lm_head = lm_head
    def forward(self, x):
        x = self.blocks(x)
        x = self.ln_f(x)
        return self.lm_head(x)

def build_parts():
    base = GPT2LMHeadModel.from_pretrained("gpt2")
    # weight tying 해제
    base.lm_head.weight = nn.Parameter(base.transformer.wte.weight.clone())

    layers = base.transformer.h
    # 12개 중 11개를 Stage1, 1개를 Stage2에 할당
    mid = len(layers) * 11 // 12  # = 11

    part1 = Stage1(base.transformer.wte,
                   base.transformer.wpe,
                   layers[:mid]).to("cuda:0")

    part2 = Stage2(layers[mid:],
                   base.transformer.ln_f,
                   base.lm_head).to("cuda:1")

    opt1 = Adam(part1.parameters(), lr=1e-5)
    opt2 = Adam(part2.parameters(), lr=1e-5)

    return part1, part2, opt1, opt2

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=4)
    args = parser.parse_args()

    batches, tokenizer = get_batches(args.batch_size)
    if not batches:
        raise RuntimeError("No data batches.")

    part1, part2, opt1, opt2 = build_parts()
    start_all = time.time()

    for step, input_ids in enumerate(batches):
        if input_ids.numel() == 0:
            continue

        # Stage1 (cuda:0)
        x = input_ids.to("cuda:0")
        t0 = time.time()
        h = part1(x)

        # Stage2 (cuda:1)
        h = h.to("cuda:1")
        logits = part2(h)

        # Loss 계산
        labels = x.view(-1).to("cuda:1")
        loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            labels,
            ignore_index=tokenizer.pad_token_id
        )

        # Backward & step
        opt2.zero_grad()
        loss.backward()
        opt2.step()

        # embedding 파트 업데이트
        opt1.step()
        opt1.zero_grad()

        dt = time.time() - t0
        cpu, gpu = get_load()
        print(f"Step {step}: time={dt:.3f}s, CPU={cpu}%, GPU={gpu}")

    print(f"All done in {time.time() - start_all:.3f}s")

if __name__ == "__main__":
    main()

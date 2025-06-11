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

# NVML 초기화
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
        enc = tokenizer(chunk, return_tensors="pt",
                        padding=True, truncation=True)
        batches.append(enc.input_ids)  # CPU LongTensor
        if len(batches) >= max_steps:
            break
    return batches, tokenizer

class BlockWrapper(nn.Module):
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
        x = input_ids.long()  # ensure LongTensor
        tok_emb = self.wte(x)
        seq_len = x.size(1)
        pos_ids = torch.arange(seq_len, device=x.device).unsqueeze(0).expand_as(x)
        pos_emb = self.wpe(pos_ids)
        h = tok_emb + pos_emb
        return self.blocks(h)

class Stage2(nn.Module):
    def __init__(self, blocks, ln_f):
        super().__init__()
        self.blocks = nn.Sequential(*[BlockWrapper(b) for b in blocks])
        self.ln_f = ln_f
    def forward(self, x):
        h = self.blocks(x)
        return self.ln_f(h)

class ShardedHead(nn.Module):
    def __init__(self, base_head: nn.Linear, split_ratio=0.8):
        super().__init__()
        weight = base_head.weight.data  # [vocab_size, hidden_size]
        vocab_size, hidden_size = weight.shape
        split = int(vocab_size * split_ratio)
        # 80% → GPU0
        self.w1 = nn.Parameter(weight[:split].clone().to("cuda:0"))
        # 20% → GPU1
        self.w2 = nn.Parameter(weight[split:].clone().to("cuda:1"))
        self.split = split
    def forward(self, h):
        # h on cuda:1
        B, S, H = h.size()
        flat = h.view(-1, H)
        # part2 on cuda:1
        logits2 = F.linear(flat, self.w2)  # [B*S, vocab2]
        logits2 = logits2.view(B, S, -1)
        # part1 on cuda:0
        h0 = h.to("cuda:0").view(-1, H)
        logits1 = F.linear(h0, self.w1).view(B, S, -1).to("cuda:1")
        return torch.cat([logits1, logits2], dim=-1)

def build_parts():
    base = GPT2LMHeadModel.from_pretrained("gpt2")
    # break weight tying
    base.lm_head.weight = nn.Parameter(base.transformer.wte.weight.clone())

    layers = base.transformer.h
    # 12개 중 11개 블록 Stage1, 1개 블록 Stage2
    mid = len(layers) * 11 // 12

    part1 = Stage1(
        wte=base.transformer.wte,
        wpe=base.transformer.wpe,
        blocks=layers[:mid]
    ).to("cuda:0")

    part2 = Stage2(
        blocks=layers[mid:],
        ln_f=base.transformer.ln_f
    ).to("cuda:1")

    head = ShardedHead(base.lm_head, split_ratio=0.8)

    opt1 = Adam(list(part1.parameters()), lr=1e-5)
    opt2 = Adam(list(part2.parameters()), lr=1e-5)
    optH = Adam(list(head.parameters()), lr=1e-5)

    return part1, part2, head, opt1, opt2, optH

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=4)
    args = parser.parse_args()

    batches, tokenizer = get_batches(args.batch_size)
    if not batches:
        raise RuntimeError("No data batches.")

    part1, part2, head, opt1, opt2, optH = build_parts()
    t0_all = time.time()

    for step, input_ids in enumerate(batches):
        if input_ids.numel() == 0:
            continue

        # Stage1 (cuda:0)
        x = input_ids.to("cuda:0")
        t0 = time.time()
        h1 = part1(x)

        # Stage2 (cuda:1)
        h2 = part2(h1.to("cuda:1"))

        # Sharded head
        logits = head(h2)

        # compute loss on cuda:1
        labels = x.view(-1).to("cuda:1")
        loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            labels,
            ignore_index=tokenizer.pad_token_id
        )

        # backward & step
        opt2.zero_grad(); optH.zero_grad()
        loss.backward()
        opt2.step(); optH.step()

        # embedding part update
        opt1.step(); opt1.zero_grad()

        dt = time.time() - t0
        cpu, gpu = get_load()
        print(f"Step {step}: time={dt:.3f}s, CPU={cpu}%, GPU={gpu}")

    print(f"All done in {time.time() - t0_all:.3f}s")

if __name__ == "__main__":
    main()

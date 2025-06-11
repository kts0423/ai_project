#!/usr/bin/env python3
import time, argparse
import torch
from torch.optim import Adam
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import psutil
from pynvml import nvmlInit, nvmlDeviceGetHandleByIndex, nvmlDeviceGetUtilizationRates

nvmlInit()
gpu_count = torch.cuda.device_count()
handles = [nvmlDeviceGetHandleByIndex(i) for i in range(gpu_count)]

def get_load():
    cpu = psutil.cpu_percent()
    gpu = [nvmlDeviceGetUtilizationRates(h).gpu for h in handles]
    return cpu, gpu

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=1)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GPT2LMHeadModel.from_pretrained("gpt2").to(device)
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    model.resize_token_embeddings(len(tokenizer))
    optimizer = Adam(model.parameters(), lr=1e-5)

    # 데이터 준비
    from datasets import load_dataset
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    texts = [t for t in ds["text"] if t and t.strip()]
    batches = texts[:5*args.batch_size]
    batches = [batches[i:i+args.batch_size] for i in range(0,len(batches),args.batch_size)]

    start_all = time.time()
    for step, batch in enumerate(batches):
        enc = tokenizer(batch, return_tensors="pt", padding=True, truncation=True)
        input_ids = enc.input_ids.to(device)
        t0 = time.time()
        outputs = model(input_ids, labels=input_ids)
        loss = outputs.loss
        optimizer.zero_grad(); loss.backward(); optimizer.step()
        dt = time.time()-t0
        cpu, gpu = get_load()
        print(f"Step {step}: time={dt:.3f}s, CPU={cpu}%, GPU={gpu}")
    print(f"All done in {time.time()-start_all:.3f}s")

if __name__=="__main__":
    main()

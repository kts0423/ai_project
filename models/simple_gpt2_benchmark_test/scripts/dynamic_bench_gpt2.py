#!/usr/bin/env python3
import time, argparse, os, torch
from torch.optim import Adam
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import psutil
from pynvml import nvmlInit, nvmlDeviceGetHandleByIndex, nvmlDeviceGetUtilizationRates
from datasets import load_dataset

nvmlInit()

def get_load():
    gpu_count = torch.cuda.device_count()
    handles = [nvmlDeviceGetHandleByIndex(i) for i in range(gpu_count)]
    cpu = psutil.cpu_percent()
    gpu = [nvmlDeviceGetUtilizationRates(h).gpu for h in handles]
    return cpu, gpu

def get_batches(batch_size, max_steps=5):
    os.environ["HF_DATASETS_CACHE"] = "/mnt/d/ai-data/cache/datasets"
    ds = load_dataset("wikitext","wikitext-2-raw-v1", split="train")
    texts = [t for t in ds["text"] if t and t.strip()]
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    batches = []
    for i in range(0, len(texts), batch_size):
        enc = tokenizer(texts[i:i+batch_size],
                        return_tensors="pt", padding=True, truncation=True)
        batches.append(enc.input_ids)
        if len(batches)>=max_steps: break
    return batches

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=1)
    args = parser.parse_args()

    gpus = list(range(torch.cuda.device_count()))
    models, opts = [], []
    for dev in gpus:
        device = torch.device(f"cuda:{dev}")
        m = GPT2LMHeadModel.from_pretrained("gpt2").to(device)
        m.resize_token_embeddings(len(GPT2Tokenizer.from_pretrained("gpt2")))
        models.append(m)
        opts.append(Adam(m.parameters(), lr=1e-5))

    batches = get_batches(args.batch_size)
    start_all = time.time()

    for step, input_ids in enumerate(batches):
        cpu, gpu = get_load()
        tgt = gpu.index(min(gpu))
        device = torch.device(f"cuda:{tgt}")
        m, opt = models[tgt], opts[tgt]

        input_ids = input_ids.to(device)
        t0 = time.time()
        out = m(input_ids, labels=input_ids)
        loss = out.loss
        opt.zero_grad(); loss.backward(); opt.step()
        dt = time.time()-t0
        cpu, gpu = get_load()
        print(f"Step {step} → GPU{tgt}: time={dt:.3f}s, CPU={cpu}%, GPU={gpu}")

    print(f"All done in {time.time()-start_all:.3f}s")

if __name__=="__main__":
    main()

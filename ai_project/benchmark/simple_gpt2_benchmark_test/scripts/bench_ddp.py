import time, argparse, os
import torch, torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import Adam
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from datasets import load_dataset

def main():
    dist.init_process_group('nccl')
    rank = dist.get_rank()
    torch.cuda.set_device(rank)

    # 모델/토크나이저
    model = GPT2LMHeadModel.from_pretrained("gpt2").to(rank)
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    model = DDP(model, device_ids=[rank])
    optimizer = Adam(model.parameters(), lr=1e-5)

    # 데이터
    ds = load_dataset("wikitext","wikitext-2-raw-v1",split="train")
    texts = [t for t in ds["text"] if t and t.strip()]
    batch_size=4
    batches = [texts[i:i+batch_size] for i in range(0,len(texts),batch_size)]
    batches = batches[:10]  # 예시

    start = time.time()
    for step, chunk in enumerate(batches):
        enc = tokenizer(chunk, return_tensors="pt", padding=True, truncation=True)
        input_ids = enc.input_ids.to(rank)
        optimizer.zero_grad()
        outputs = model(input_ids, labels=input_ids)
        outputs.loss.backward()
        optimizer.step()
        if rank == 0:
            print(f"Step {step}: loss={outputs.loss.item():.3f}")
    if rank == 0:
        print(f"Total: {time.time()-start:.3f}s")

if __name__=="__main__":
    main()

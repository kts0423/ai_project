# scripts/bench_gpt2.py

import time
import argparse
import torch
import deepspeed
from torch.optim import Adam
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from load_data import get_dataloader
from monitor import get_load

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="gpt2")
    parser.add_argument("--deepspeed_config", type=str,
                        default="../configs/deepspeed_config.json")
    parser.add_argument("--local_rank", type=int, default=0)
    args = parser.parse_args()

    # device 설정
    device = f"cuda:{args.local_rank}" if torch.cuda.is_available() else "cpu"

    # 토크나이저 준비
    tokenizer = GPT2Tokenizer.from_pretrained(args.model_name)
    tokenizer.pad_token = tokenizer.eos_token

    # 모델 로드 및 resize
    model = GPT2LMHeadModel.from_pretrained(args.model_name).to(device)
    model.resize_token_embeddings(len(tokenizer))

    # 원본 PyTorch 옵티마이저 생성
    torch_optimizer = Adam(model.parameters(), lr=1e-5)

    # DeepSpeed 초기화 (원본 옵티마이저 전달)
    model_engine, ds_optimizer, _, _ = deepspeed.initialize(
        args=args,
        model=model,
        model_parameters=model.parameters(),
        optimizer=torch_optimizer
    )

    # checkpoint_engine이 없으면 dummy 주입
    if model_engine.checkpoint_engine is None:
        class DummyCheckpointEngine:
            def is_decoupled(self): return False
        model_engine.checkpoint_engine = DummyCheckpointEngine()

    # step() 호출 시 에러 나면 원본 옵티마이저로 fallback
    orig_step = model_engine.step
    def step_override(*a, **k):
        try:
            return orig_step(*a, **k)
        except AttributeError:
            # ZeRO-Offload wrapper에 step()이 없을 때
            torch_optimizer.step()
            torch_optimizer.zero_grad()
    model_engine.step = step_override

    # 마이크로 배치 크기 정수화
    micro_batch = model_engine.train_micro_batch_size_per_gpu
    micro_batch = int(micro_batch() if callable(micro_batch) else micro_batch)

    # 데이터 로더
    batches = get_dataloader(split="train", batch_size=micro_batch)

    for step, batch in enumerate(batches[:5]):
        if not batch:
            continue

        # 입력 처리
        inputs = tokenizer(batch,
                           return_tensors="pt",
                           padding=True,
                           truncation=True)
        input_ids = inputs.input_ids.to(device)
        print(f"Step {step}: input_ids shape = {input_ids.shape}")

        # 순전파 및 손실 계산
        outputs = model_engine(input_ids, labels=input_ids)
        loss = outputs.loss

        # backward + step
        loss.backward()
        model_engine.step()

        # 시간 및 부하 로깅
        elapsed = time.time() - start
        cpu_load, gpu_loads = get_load()
        print(f"Step {step}: time={elapsed:.3f}s, CPU={cpu_load}%, GPU={gpu_loads}")

if __name__ == "__main__":
    start = time.time()
    main()

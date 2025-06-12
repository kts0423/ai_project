import torch.multiprocessing as mp
mp.set_start_method("fork", force=True)

import time
import json
import torch
import pynvml
from torch.optim import AdamW
from train_daily import trainer, model, tokenizer, training_args

def auto_tune(num_trials: int = 3):
    # GPU 초기화
    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(torch.cuda.current_device())

    # 튜닝 후보 리스트
    workers_list = [0, 4, 8]
    batch_list   = [16, 24]
    acc_list     = [2, 4]
    r_list       = [4, 8]
    dropout_list = [0.05, 0.1]

    best = {"score": 0}
    loader_cache = {}

    # 미니 학습용 옵티마이저
    optimizer = AdamW(model.parameters(), lr=training_args.learning_rate)

    for nw in workers_list:
        for bs in batch_list:
            # DataLoader 캐싱 + 워밍업
            key = (nw, bs)
            if key not in loader_cache:
                trainer.args.dataloader_num_workers      = nw
                trainer.args.per_device_train_batch_size = bs
                loader = trainer.get_train_dataloader()
                # 워밍업 배치 1회
                for _, _ in zip(range(1), loader):
                    pass
                loader_cache[key] = loader
            loader = loader_cache[key]

            for acc in acc_list:
                for r in r_list:
                    for d in dropout_list:
                        # GPU 상태 체크
                        mem  = pynvml.nvmlDeviceGetMemoryInfo(handle).free / 1024**2
                        temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
                        if mem < bs * 100 or temp > 85:
                            continue

                        # Trainer/Model 설정
                        trainer.args.gradient_accumulation_steps = acc
                        model.peft_config["r"]            = r
                        model.peft_config["lora_dropout"] = d

                        # 여러 번 반복해서 rate 측정
                        rates = []
                        for trial in range(num_trials):
                            model.train()
                            total, t0 = 0, time.time()
                            for _, batch in zip(range(5), loader):  # 5 mini-steps
                                batch = {k: v.cuda() for k, v in batch.items()}
                                outputs = model(**batch)
                                loss = outputs.loss
                                loss.backward()
                                optimizer.step()
                                optimizer.zero_grad()
                                total += batch["input_ids"].size(0)
                            rates.append(total / (time.time() - t0))
                            torch.cuda.empty_cache()

                        avg_rate = sum(rates) / len(rates)
                        # 워커 수 패널티
                        penalty = 1 + nw * 0.05
                        score = avg_rate * (mem / 1024) * (1 - temp / 100) / penalty

                        print(f"🧪 nw={nw}, bs={bs}, acc={acc}, r={r}, d={d} → "
                              f"rates={['{:.1f}'.format(x) for x in rates]}, avg_rate={avg_rate:.1f}, score={score:.1f}")

                        if score > best["score"]:
                            best = {
                                "nw": nw, "bs": bs, "acc": acc,
                                "r": r, "d": d, "score": score
                            }

    # 최적 설정 저장
    with open("best_config.json", "w") as f:
        json.dump(best, f, indent=4)
    print("✅ Best config saved:", best)

if __name__ == "__main__":
    auto_tune(num_trials=3)

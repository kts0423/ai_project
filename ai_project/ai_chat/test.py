#!/usr/bin/env python
# train_daily.py

# ─────────────────────────────────────────────────
# 0) 자동 의존성 검사 및 설치 (최초 실행 시 한 번만)
# ─────────────────────────────────────────────────
import importlib, subprocess, sys
required_packages = [
    "numpy", "torch", "pynvml", "bitsandbytes", "transformers",
    "datasets", "peft", "tensorboard", "pillow", "psutil"
]
for pkg in required_packages:
    try:
        importlib.import_module(pkg)
    except ImportError:
        print(f"Installing missing package: {pkg}")
        subprocess.check_call([sys.executable, "-m", "pip", "install", pkg])

# ─────────────────────────────────────────────────
# 1) 라이브러리 임포트 및 로깅 설정
# ─────────────────────────────────────────────────
import os
import warnings
import random
import time
import math
import numpy as np
import torch
import psutil
import logging as py_logging
from logging.handlers import RotatingFileHandler
from datasets import load_dataset, load_from_disk
from torch.utils.data import DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding,
    BitsAndBytesConfig,
    logging
)
from transformers.integrations import TensorBoardCallback
from peft import LoraConfig, get_peft_model, TaskType

py_logging.basicConfig(
    level=py_logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        py_logging.StreamHandler(),
        RotatingFileHandler("train_daily.log", maxBytes=10*1024*1024, backupCount=5)
    ]
)
logger = py_logging.getLogger(__name__)
# cuDNN 벤치마크 모드 활성화
torch.backends.cudnn.benchmark = True

# ─────────────────────────────────────────────────
# 2) 환경 경고 무시 & 시드 고정
# ─────────────────────────────────────────────────
warnings.filterwarnings(
    "ignore",
    message="NVIDIA GeForce RTX 5070 Ti with CUDA capability sm_120"
)
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
logger.info("Using device: %s", torch.cuda.get_device_name(torch.cuda.current_device()))

# ─────────────────────────────────────────────────
# 3) 설정
# ─────────────────────────────────────────────────
MODEL_NAME = "gpt2"
DATA_FILE = "/mnt/d/aihub/processed/aihub_daily.jsonl"
OUTPUT_DIR = "/mnt/d/aihub/processed/finetuned_daily"
TENSORBOARD_DIR = "/mnt/d/aihub/processed/runs"
CACHE_DIR = "/mnt/d/aihub/processed/cached"
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(CACHE_DIR, exist_ok=True)

# ─────────────────────────────────────────────────
# 4) 데이터 전처리 및 캐싱
# ─────────────────────────────────────────────────
try:
    tok_train = load_from_disk(f"{CACHE_DIR}/tok_train")
    tok_eval  = load_from_disk(f"{CACHE_DIR}/tok_eval")
    logger.info("Loaded tokenized datasets from cache.")
except Exception:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token
    raw = load_dataset("json", data_files=DATA_FILE)["train"]
    small = raw.shuffle(seed=SEED).select(range(int(len(raw)*0.3)))
    split = small.train_test_split(test_size=0.05, seed=SEED)
    def tokenize_fn(examples):
        enc = tokenizer(
            examples["prompt"], examples["response"],
            truncation=True, max_length=256, padding="max_length"
        )
        enc["labels"] = enc["input_ids"].copy()
        return enc
    tok_train = split["train"].map(tokenize_fn, batched=True, remove_columns=["prompt","response"])
    tok_eval  = split["test"].map(tokenize_fn, batched=True, remove_columns=["prompt","response"])
    tok_train.save_to_disk(f"{CACHE_DIR}/tok_train")
    tok_eval.save_to_disk(f"{CACHE_DIR}/tok_eval")
    logger.info("Tokenized datasets saved to cache.")

# ─────────────────────────────────────────────────
# 5) 모델 로드 및 준비 (8bit 양자화 + LoRA)
# ─────────────────────────────────────────────────
bnb_config = BitsAndBytesConfig(load_in_8bit=True, llm_int8_threshold=6.0)
base_model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=bnb_config,
    torch_dtype=torch.bfloat16,
    device_map={"": torch.cuda.current_device()}
)
lora_cfg = LoraConfig(task_type=TaskType.CAUSAL_LM, inference_mode=False,
                      r=8, lora_alpha=16, lora_dropout=0.05)
model = get_peft_model(base_model, lora_cfg)
# quantized model은 torch.compile 하지 않음

# ─────────────────────────────────────────────────
# 6) Trainer 정의 (TensorBoard 포함)
# ─────────────────────────────────────────────────
class MyTrainer(Trainer):
    def __init__(self, *args, label_names=None, **kwargs):
        callbacks = kwargs.pop('callbacks', []) + [TensorBoardCallback()]
        super().__init__(*args, callbacks=callbacks, **kwargs)
        if label_names:
            self.label_names = label_names

    def get_train_dataloader(self):
        nw = self.args.dataloader_num_workers
        kw = dict(
            dataset=self.train_dataset,
            batch_size=self.args.per_device_train_batch_size,
            shuffle=True,
            collate_fn=self.data_collator,
            num_workers=nw,
            pin_memory=True,
            persistent_workers=(nw>0)
        )
        if nw>0: kw['prefetch_factor']=2
        return DataLoader(**kw)

    def get_eval_dataloader(self, eval_dataset=None):
        ds = eval_dataset or self.eval_dataset
        nw = self.args.dataloader_num_workers
        kw = dict(
            dataset=ds,
            batch_size=self.args.per_device_eval_batch_size,
            shuffle=False,
            collate_fn=self.data_collator,
            num_workers=nw,
            pin_memory=True,
            persistent_workers=(nw>0)
        )
        if nw>0: kw['prefetch_factor']=2
        return DataLoader(**kw)

# ─────────────────────────────────────────────────
# 7) 하이퍼파라미터 및 Trainer 초기화
# ─────────────────────────────────────────────────
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    run_name="gpt2_small_30pct_8bit",
    per_device_train_batch_size=16,
    gradient_accumulation_steps=2,
    per_device_eval_batch_size=16,
    num_train_epochs=1,
    learning_rate=3e-4,
    bf16=True, fp16=False,
    logging_steps=10,
    save_strategy="steps",
    save_steps=500,
    eval_strategy="steps",
    eval_steps=500,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    save_total_limit=3,
    report_to="tensorboard",
    logging_dir=TENSORBOARD_DIR,
    dataloader_num_workers=4
)
logging.set_verbosity_error()
trainer = MyTrainer(
    model=model,
    args=training_args,
    train_dataset=tok_train,
    eval_dataset=tok_eval,
    data_collator=DataCollatorWithPadding(tokenizer),
    label_names=["labels"]
)

# ─────────────────────────────────────────────────
# 8) 자동 튜닝 및 본 학습
# ─────────────────────────────────────────────────
if __name__ == "__main__":
    try:
        import pynvml; pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(torch.cuda.current_device())
    except:
        handle = None

    workers_list, batch_list, acc_list = [0,4,8,16], [16,24,32], [2,4]
    r_list, dropout_list = [4,8], [0.05,0.1]
    best = {"score":0}
    for nw in workers_list:
        for bs in batch_list:
            for acc in acc_list:
                for r in r_list:
                    for d in dropout_list:
                        if handle:
                            mem = pynvml.nvmlDeviceGetMemoryInfo(handle).free/1024**2
                            tmp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
                        else:
                            mem,tmp = float('inf'),0
                        if mem<bs*50 or tmp>85: continue
                        cpu = psutil.cpu_percent()
                        gpu_util = torch.cuda.utilization()
                        torch.cuda.empty_cache()
                        trainer.args.dataloader_num_workers=nw
                        trainer.args.per_device_train_batch_size=bs
                        trainer.args.gradient_accumulation_steps=acc
                        trainer.model.peft_config.r = r
                        trainer.model.peft_config.lora_dropout = d
                        loader = trainer.get_train_dataloader()
                        t0 = time.time(); total=0
                        for _,batch in zip(range(10),loader):
                            total+=batch["input_ids"].size(0)
                        rate = total/(time.time()-t0)
                        score = (math.log(rate+1)
                                 * math.log(mem+1)
                                 * (1-tmp/100)
                                 * (gpu_util/100)
                                 * (1-cpu/100))
                        logger.info(
                            f"Trial nw={nw}, bs={bs}, acc={acc}, r={r}, d={d}, "
                            f"mem={mem:.1f}MB, temp={tmp}C, gpu={gpu_util}%, cpu={cpu}% "
                            f"-> rate={rate:.1f}, score={score:.1f}"
                        )
                        if score>best["score"]:
                            best={"nw":nw,"bs":bs,"acc":acc,"r":r,"d":d,"score":score}
                        torch.cuda.empty_cache()
    logger.info(f"Best config: {best}")
    trainer.args.dataloader_num_workers=best['nw']
    trainer.args.per_device_train_batch_size=best['bs']
    trainer.args.gradient_accumulation_steps=best['acc']
    trainer.model.peft_config.r = best['r']
    trainer.model.peft_config.lora_dropout = best['d']
    logger.info("Starting fine-tuning...")
    trainer.train()
    metrics = trainer.evaluate()
    for k,v in metrics.items():
        logger.info(f"{k}: {v}")
    model.save_pretrained(OUTPUT_DIR)
    logger.info(f"Training complete. Model saved at {OUTPUT_DIR}")

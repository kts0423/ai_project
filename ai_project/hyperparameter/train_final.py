import os
# 스크립트 최상단에서 사용할 GPU를 0번으로 강제 지정합니다.
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import torch
import json # [추가] json 라이브러리를 import 합니다.
from datasets import load_dataset, disable_caching

# 캐시 기능을 비활성화하여 데이터 로딩 병목 현상을 방지합니다.
disable_caching()
print("[진단] datasets 캐시 기능을 비활성화했습니다. 모든 데이터는 RAM에서 처리됩니다.")

# 필요한 클래스들을 import 합니다.
from transformers import (
    GPT2LMHeadModel,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)

# --- 기본 설정 및 경로 ---
MODEL_NAME = "skt/kogpt2-base-v2"
DATA_FILE = "data/aihub_daily.jsonl"
PARAMS_FILE = "best_params.json" # [추가] 불러올 파라미터 파일 이름
OUTPUT_DIR = "results"
FINAL_MODEL_DIR = os.path.join(OUTPUT_DIR, "best_model_for_run")
SEED = 42
FINAL_TRAIN_SUBSET_SIZE = 100000 # 최종 훈련에 사용할 데이터 샘플 크기

# GPU 사용 가능 여부 확인
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")


# --- [수정] 파일에서 HPO 결과 불러오기 ---
try:
    with open(PARAMS_FILE, "r") as f:
        best_params = json.load(f)
except FileNotFoundError:
    print(f"오류: '{PARAMS_FILE}' 파일을 찾을 수 없습니다.")
    print("먼저 'hpo.py' 스크립트를 실행하여 최적의 하이퍼파라미터를 찾아주세요.")
    exit()

print("--- 파일에서 최적 하이퍼파라미터를 불러왔습니다 ---")
for key, value in best_params.items():
    print(f"  {key}: {value}")


# --- 데이터 준비 ---
print("\n토크나이저를 로딩합니다...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token

def is_valid_entry(example: dict) -> bool:
    prompt = example.get('prompt')
    response = example.get('response')
    return isinstance(prompt, str) and isinstance(response, str) and prompt.strip() and response.strip()

def preprocess_function(examples: dict) -> dict:
    texts = []
    for prompt, response in zip(examples['prompt'], examples['response']):
        text = prompt + response + tokenizer.eos_token
        texts.append(text)
    return tokenizer(texts, truncation=True, max_length=128, padding="max_length")

print("전체 데이터셋을 로딩하고 처리합니다...")
raw_dataset = load_dataset('json', data_files=DATA_FILE, split='train')
valid_dataset = raw_dataset.filter(is_valid_entry, num_proc=4)
tokenized_dataset = valid_dataset.map(
    preprocess_function,
    batched=True,
    remove_columns=["prompt", "response"],
    num_proc=4
)


# --- 최종 훈련용 데이터셋 준비 ---
print(f"\n최종 훈련을 위해 '{FINAL_TRAIN_SUBSET_SIZE}개 데이터 서브셋'을 준비합니다...")
if len(tokenized_dataset) < FINAL_TRAIN_SUBSET_SIZE:
    final_train_subset = tokenized_dataset
else:
    final_train_subset = tokenized_dataset.shuffle(seed=SEED).select(range(FINAL_TRAIN_SUBSET_SIZE))

final_train_val_split = final_train_subset.train_test_split(test_size=0.1, seed=SEED)
final_train_dataset = final_train_val_split['train']
final_eval_dataset = final_train_val_split['test']

print(f"최종 훈련 데이터: {len(final_train_dataset)}개")
print(f"최종 검증 데이터: {len(final_eval_dataset)}개")


# --- 최종 모델 훈련 ---
print("\n최종 모델 훈련을 시작합니다...")

final_training_args = TrainingArguments(
    output_dir=FINAL_MODEL_DIR,
    seed=SEED,
    learning_rate=best_params['learning_rate'],
    per_device_train_batch_size=best_params['per_device_train_batch_size'],
    num_train_epochs=best_params['num_train_epochs'],
    weight_decay=best_params['weight_decay'],
    warmup_ratio=best_params['warmup_ratio'],
    max_grad_norm=1.0,
    bf16=True, 
    dataloader_num_workers=16,
    logging_strategy="epoch",
    eval_strategy="epoch",
    save_strategy="epoch",
    report_to="none",
)

model = GPT2LMHeadModel.from_pretrained(
    MODEL_NAME,
    attn_implementation="flash_attention_2",
    torch_dtype=torch.bfloat16
).to(device)
model.resize_token_embeddings(len(tokenizer))

data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

final_trainer = Trainer(
    model=model,
    args=final_training_args,
    train_dataset=final_train_dataset,
    eval_dataset=final_eval_dataset,
    data_collator=data_collator,
)

final_trainer.train()
final_trainer.save_model(FINAL_MODEL_DIR)
print(f"\n최종 모델 훈련 완료! 모델이 '{FINAL_MODEL_DIR}' 에 저장되었습니다.")
print("이제 'run.py' 스크립트를 실행하여 훈련된 모델과 대화해볼 수 있습니다.")

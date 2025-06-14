import os
# 스크립트 최상단에서 사용할 GPU를 0번으로 강제 지정합니다.
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import torch
from datasets import load_dataset, disable_caching

# 캐시 기능을 비활성화합니다.
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
OUTPUT_DIR = "results"
SEED = 42

# GPU 사용 가능 여부 확인
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")


# --- 데이터 준비 ---
print("\n토크나이저를 로딩합니다...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token

def preprocess_function(examples: dict) -> dict:
    """prompt와 response를 합쳐서 하나의 훈련용 텍스트로 만듭니다."""
    texts = []
    for prompt, response in zip(examples['prompt'], examples['response']):
        text = prompt + response + tokenizer.eos_token
        texts.append(text)
    return tokenizer(texts, truncation=True, max_length=128, padding="max_length")

print("데이터셋을 로딩하고 전처리합니다...")
raw_dataset = load_dataset('json', data_files=DATA_FILE, split='train')
tokenized_dataset = raw_dataset.map(
    preprocess_function,
    batched=True,
    remove_columns=["prompt", "response"]
)
tokenized_dataset = tokenized_dataset.filter(lambda x: sum(x['attention_mask']) > 1)

if len(tokenized_dataset) == 0:
    raise ValueError("모든 데이터가 필터링되었습니다.")

print("데이터셋을 훈련/검증 세트로 분리합니다...")
train_val_split = tokenized_dataset.train_test_split(test_size=0.1, seed=SEED)
train_dataset = train_val_split['train']
eval_dataset = train_val_split['test']

print(f"훈련 데이터: {len(train_dataset)}개")
print(f"검증 데이터: {len(eval_dataset)}개")

data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)


# --- [핵심] 단일 훈련 실행 ---
print("\n단순화된 단일 훈련을 시작합니다...")

# 모델 로드 및 토크나이저 크기 동기화
model = GPT2LMHeadModel.from_pretrained(MODEL_NAME).to(device)
model.resize_token_embeddings(len(tokenizer))

# 고정된 하이퍼파라미터 설정
training_args = TrainingArguments(
    output_dir=os.path.join(OUTPUT_DIR, "simple_test_run"),
    seed=SEED,
    num_train_epochs=1,  # 테스트를 위해 1 에포크만 실행
    per_device_train_batch_size=8,
    learning_rate=5e-5,
    fp16=True if device == "cuda" else False,
    logging_strategy="steps",
    logging_steps=10, # 10 스텝마다 로그 출력
    eval_strategy="epoch",
    save_strategy="epoch",
    dataloader_num_workers=0, # 데이터 로더 워커 비활성화
)

# Trainer 객체 생성
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    data_collator=data_collator,
)

try:
    # 훈련 실행
    print("trainer.train()을 호출합니다...")
    trainer.train()
    print("\n[성공] 단순 훈련이 정상적으로 종료되었습니다!")

    # 평가 실행
    print("\n평가를 시작합니다...")
    eval_results = trainer.evaluate()
    print(f"평가 결과: {eval_results}")

except Exception as e:
    print(f"\n[실패] 훈련 중 오류가 발생했습니다: {e}")


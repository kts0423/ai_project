import os
# 스크립트 최상단에서 사용할 GPU를 0번으로 강제 지정합니다.
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import torch
import optuna
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
    EarlyStoppingCallback
)

# --- 기본 설정 및 경로 ---
MODEL_NAME = "skt/kogpt2-base-v2"
DATA_FILE = "data/aihub_daily.jsonl"
OUTPUT_DIR = "results"
SEED = 42
HPO_SUBSET_SIZE = 15000       # HPO에 사용할 데이터 샘플 크기
FINAL_TRAIN_SUBSET_SIZE = 100000 # 최종 훈련에 사용할 데이터 샘플 크기

# GPU 사용 가능 여부 확인
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")


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
print(f"총 유효 데이터 개수: {len(tokenized_dataset)}개")


# --- [1단계] HPO를 위한 데이터셋 서브샘플링 ---
print(f"\n[1단계] HPO를 위해 전체 데이터 중 {HPO_SUBSET_SIZE}개만 샘플링합니다.")
if len(tokenized_dataset) < HPO_SUBSET_SIZE:
    raise ValueError(f"전체 데이터({len(tokenized_dataset)})가 HPO 샘플링 크기({HPO_SUBSET_SIZE})보다 작습니다.")
    
hpo_dataset = tokenized_dataset.shuffle(seed=SEED).select(range(HPO_SUBSET_SIZE))

hpo_train_val_split = hpo_dataset.train_test_split(test_size=0.2, seed=SEED)
hpo_train_dataset = hpo_train_val_split['train']
hpo_eval_dataset = hpo_train_val_split['test']

print(f"HPO용 훈련 데이터: {len(hpo_train_dataset)}개")
print(f"HPO용 검증 데이터: {len(hpo_eval_dataset)}개")


# 데이터 콜레이터
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)


# --- [2단계] 서브셋 데이터로 빠른 HPO 수행 ---
def objective(trial: optuna.trial.Trial) -> float:
    print(f"\n--- Optuna Trial #{trial.number} 시작 ---")
    model = GPT2LMHeadModel.from_pretrained(MODEL_NAME).to(device)
    model.resize_token_embeddings(len(tokenizer))

    training_args = TrainingArguments(
        output_dir=os.path.join(OUTPUT_DIR, f"trial_{trial.number}"),
        seed=SEED,
        learning_rate=trial.suggest_float('learning_rate', 1e-5, 5e-4, log=True),
        per_device_train_batch_size=trial.suggest_categorical('per_device_train_batch_size', [16, 32]),
        num_train_epochs=trial.suggest_int('num_train_epochs', 2, 4),
        weight_decay=trial.suggest_float('weight_decay', 1e-3, 0.1, log=True),
        warmup_ratio=trial.suggest_float('warmup_ratio', 0.0, 0.2),
        max_grad_norm=1.0,
        fp16=True,
        dataloader_num_workers=16,
        logging_strategy="steps",
        logging_steps=100,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=hpo_train_dataset,
        eval_dataset=hpo_eval_dataset,
        data_collator=data_collator,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
    )

    trainer.train()
    result = trainer.evaluate()
    
    del model, trainer
    torch.cuda.empty_cache()

    return float(result["eval_loss"])


print("\n[2단계] 서브셋 데이터로 빠른 하이퍼파라미터 최적화를 시작합니다...")
study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=20)

print("\n하이퍼파라미터 최적화 종료!")
best_trial = study.best_trial
print("--- 최적 하이퍼파라미터 결과 ---")
print(f"  Value (최소 eval_loss): {best_trial.value}")
print("  Best_Params: ")
for key, value in best_trial.params.items():
    print(f"    {key}: {value}")


# --- [3단계] 더 큰 서브셋으로 최종 모델 훈련 및 저장 ---
print(f"\n[3단계] 찾은 최적의 파라미터로 '{FINAL_TRAIN_SUBSET_SIZE}개 데이터 서브셋'을 훈련합니다...")

if len(tokenized_dataset) < FINAL_TRAIN_SUBSET_SIZE:
    print(f"경고: 전체 데이터({len(tokenized_dataset)})가 최종 훈련 샘플링 크기({FINAL_TRAIN_SUBSET_SIZE})보다 작아, 전체 데이터를 사용합니다.")
    final_train_subset = tokenized_dataset
else:
    final_train_subset = tokenized_dataset.shuffle(seed=SEED).select(range(FINAL_TRAIN_SUBSET_SIZE))

final_train_val_split = final_train_subset.train_test_split(test_size=0.1, seed=SEED)
final_train_dataset = final_train_val_split['train']
final_eval_dataset = final_train_val_split['test']

print(f"최종 훈련 데이터: {len(final_train_dataset)}개")
print(f"최종 검증 데이터: {len(final_eval_dataset)}개")

best_params = study.best_trial
final_training_args = TrainingArguments(
    output_dir=os.path.join(OUTPUT_DIR, "best_model_for_run"),
    seed=SEED,
    learning_rate=best_params['learning_rate'],
    per_device_train_batch_size=best_params['per_device_train_batch_size'],
    num_train_epochs=best_params['num_train_epochs'],
    weight_decay=best_params['weight_decay'],
    warmup_ratio=best_params['warmup_ratio'],
    max_grad_norm=1.0,
    fp16=True,
    dataloader_num_workers=16,
    logging_strategy="epoch",
    eval_strategy="epoch",
    save_strategy="epoch",
    report_to="none",
)

model = GPT2LMHeadModel.from_pretrained(MODEL_NAME).to(device)
model.resize_token_embeddings(len(tokenizer))

final_trainer = Trainer(
    model=model,
    args=final_training_args,
    train_dataset=final_train_dataset,
    eval_dataset=final_eval_dataset,
    data_collator=data_collator,
)

final_trainer.train()
final_trainer.save_model(os.path.join(OUTPUT_DIR, "best_model_for_run"))
print(f"\n최종 평가를 위한 모델이 {os.path.join(OUTPUT_DIR, 'best_model_for_run')} 에 저장되었습니다.")

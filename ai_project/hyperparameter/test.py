import optuna
from transformers import GPT2LMHeadModel, GPT2Tokenizer, Trainer, TrainingArguments
import torch
from torch.utils.data import Dataset
from transformers import DataCollatorForSeq2Seq

# 예시 데이터셋 준비 (여기서는 단순한 텍스트 데이터셋을 사용합니다)
class CustomDataset(Dataset):
    def __init__(self, tokenizer, text, max_length=512):
        self.tokenizer = tokenizer
        self.text = text
        self.max_length = max_length

    def __len__(self):
        return len(self.text)

    def __getitem__(self, idx):
        # pad_token을 eos_token으로 설정
        if not self.tokenizer.pad_token:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        encodings = self.tokenizer(self.text[idx], truncation=True, padding="max_length", max_length=self.max_length)
        
        # 반환값을 dict로 포장, labels를 input_ids로 설정
        return {
            'input_ids': torch.tensor(encodings['input_ids']),
            'attention_mask': torch.tensor(encodings['attention_mask']),
            'labels': torch.tensor(encodings['input_ids'])  # labels를 input_ids로 설정
        }

# 모델 학습 함수
def train_model(trial):
    # 하이퍼파라미터 설정
    learning_rate = trial.suggest_loguniform('learning_rate', 1e-5, 1e-3)
    batch_size = trial.suggest_int('batch_size', 4, 32)
    epochs = trial.suggest_int('epochs', 1, 5)
    
    # 모델과 토크나이저 로딩
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    model = GPT2LMHeadModel.from_pretrained("gpt2")

    # pad_token 설정 (eos_token을 pad_token으로 설정)
    if not tokenizer.pad_token:
        tokenizer.pad_token = tokenizer.eos_token

    # 데이터셋 준비
    texts = ["This is a sample text for training.", "Another example text.", "Fine-tuning GPT models with Optuna."]
    dataset = CustomDataset(tokenizer, texts)
    
    # 평가용 데이터셋 준비 (여기서는 동일한 텍스트를 사용)
    eval_dataset = CustomDataset(tokenizer, texts)  # 평가 데이터셋 추가

    # 옵티마이저 및 학습 설정
    training_args = TrainingArguments(
        output_dir='./results',
        learning_rate=learning_rate,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        eval_strategy="epoch",  # "epoch"로 설정하여 매 에폭마다 평가
        save_strategy="epoch",  # 매 에폭마다 모델을 저장
        load_best_model_at_end=True  # 최상의 모델을 끝에서 로드
    )

    # 사용자 정의 데이터 콜레이터 추가 (batch에서 torch.Tensor를 dict로 처리)
    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

    # Trainer 객체 생성
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        eval_dataset=eval_dataset,  # 평가 데이터셋
        data_collator=data_collator  # 사용자 정의 데이터 콜레이터 추가
    )

    # 모델 학습
    trainer.train()
    
    # 모델 평가 및 최적화된 성능 반환 (여기서는 dummy로 accuracy 반환)
    return trainer.evaluate()["eval_loss"]

# Optuna 최적화
study = optuna.create_study(direction='minimize')
study.optimize(train_model, n_trials=10)

# 최적의 하이퍼파라미터 출력
print(f"Best trial: {study.best_trial.params}")

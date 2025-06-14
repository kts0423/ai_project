import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import json
import os
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Dict, Any, List
from torch import Tensor

# 예시 데이터셋 로딩 (실제 데이터셋은 ai 프로젝트에 맞게 작성)
class CustomDataset(Dataset[str]):
    def __init__(self, data_file: str, tokenizer) -> None:
        data_file = os.path.join(os.path.dirname(__file__), '..', 'data', 'aihub_daily.jsonl')
        with open(data_file, 'r') as f:
            self.data: List[str] = [line.strip() for line in f.readlines()]
        
        self.tokenizer = tokenizer
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> torch.Tensor:
        # 텍스트를 토큰화하여 수치형 데이터로 변환
        text = self.data[idx]
        encoded = self.tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=512)
        return encoded['input_ids'].squeeze(0)  # 0차원 크기를 제거하여 배치로 반환


def collate_fn(batch):
    # 패딩을 처리하여 배치 크기를 맞추기 위해 `torch.nn.utils.rnn.pad_sequence` 사용
    return torch.nn.utils.rnn.pad_sequence(batch, batch_first=True, padding_value=0)


# 모델 훈련 함수
def train_model(config: Dict[str, Any]) -> float:
    # 하이퍼파라미터 설정
    learning_rate = config['learning_rate']
    batch_size = config['batch_size']
    epochs = config['epochs']

    # 토크나이저 로딩
    tokenizer = AutoTokenizer.from_pretrained('gpt2')

    # 패딩 토큰 설정 (eos_token을 pad_token으로 사용)
    tokenizer.pad_token = tokenizer.eos_token  # eos_token을 pad_token으로 사용
    # 또는 새로운 패딩 토큰을 추가하고 싶다면
    # tokenizer.add_special_tokens({'pad_token': '[PAD]'})

    # 데이터셋 로딩
    dataset = CustomDataset("aihub_daily.jsonl", tokenizer)
    dataloader: DataLoader[CustomDataset] = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

    # 모델 정의
    model: nn.Module = AutoModelForCausalLM.from_pretrained('gpt2')

    # 옵티마이저 설정
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    # 훈련 루프
    total_loss: float = 0.0
    for epoch in range(epochs):
        for batch in dataloader:
            optimizer.zero_grad()

            # 모델 훈련
            outputs: dict = model(batch, labels=batch)  # outputs의 타입 명시
            loss: Tensor = outputs['loss']  # loss의 타입 명시
            total_loss += loss.item()

            loss.backward()
            optimizer.step()

    # 성능 평가
    avg_loss: float = total_loss / len(dataloader)
    print(f"Finished training with loss: {avg_loss}")
    return avg_loss

if __name__ == "__main__":
    with open('best_config.json') as f:
        best_config = json.load(f)['config']
    
    train_model(best_config)

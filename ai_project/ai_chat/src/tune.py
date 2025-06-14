import json
import os
from typing import Dict, Any
from sklearn.model_selection import ParameterGrid
from train import train_model

# 하이퍼파라미터 범위 정의
param_grid: Dict[str, list] = {
    'learning_rate': [1e-5, 1e-4, 1e-3],
    'batch_size': [8, 16, 32],
    'epochs': [3, 5, 10],
    'hidden_units': [64, 128, 256]
}

# 튜닝 결과를 저장할 파일 경로
best_config_path = 'best_config.json'

def save_best_config(config: Dict[str, Any], performance: float) -> None:
    """최적의 하이퍼파라미터와 성능을 파일에 저장"""
    with open(best_config_path, 'w') as f:
        json.dump({'config': config, 'performance': performance}, f)

def tune() -> None:
    """하이퍼파라미터 튜닝을 실행하고, 성능이 가장 좋은 설정을 저장"""
    best_performance: float = -float('inf')  # 성능을 float으로 초기화
    best_config = None

    # 파라미터 그리드 탐색
    for config in ParameterGrid(param_grid):
        print(f"Trying configuration: {config}")
        
        # 학습 실행 (성능 추적)
        performance: float = train_model(config)  # 여기서 모델 학습이 진행되며 GPU 사용
        
        # 성능이 가장 좋으면 업데이트
        if performance > best_performance:
            best_performance = performance
            best_config = config

    # 최적 파라미터 저장
    save_best_config(best_config, best_performance)
    print(f"Best configuration saved: {best_config} with performance: {best_performance}")

if __name__ == "__main__":
    # `best_config.json`이 없으면 튜닝을 실행
    if not os.path.exists(best_config_path):
        print("best_config.json not found, running tuning...")
        tune()  # 튜닝 실행하여 best_config.json 생성
    else:
        with open(best_config_path) as f:
            best_config = json.load(f)['config']
        print(f"Loaded best configuration: {best_config}")
        train_model(best_config)

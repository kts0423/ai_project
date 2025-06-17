import os
# 스크립트 최상단에서 사용할 GPU를 0번으로 강제 지정합니다.
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# --- 설정 ---
MODEL_DIR = "results/best_model_for_run" # 훈련된 모델이 저장된 경로

# GPU 사용 가능 여부 확인
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")


# --- 모델 및 토크나이저 로드 ---
print(f"'{MODEL_DIR}' 에서 모델과 토크나이저를 로드합니다...")

try:
    # 토크나이저 로드
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)

    # 모델 로드 (추론 시에도 Flash Attention 2와 bf16을 적용하여 속도 향상)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_DIR,
        attn_implementation="flash_attention_2",
        torch_dtype=torch.bfloat16
    ).to(device)
    
    # 모델을 평가 모드로 설정
    model.eval()
    print("모델 로딩 완료. 대화를 시작합니다. (종료하려면 'exit' 또는 'quit' 입력)")
    print("-" * 50)

except OSError:
    print(f"오류: '{MODEL_DIR}' 경로에서 모델을 찾을 수 없습니다.")
    print("먼저 'train_final.py' 스크립트를 실행하여 모델을 훈련하고 저장해주세요.")
    exit()


# --- 대화 시작 ---
while True:
    try:
        # 사용자 입력 받기
        user_input = input("User: ")

        if user_input.lower() in ["exit", "quit"]:
            print("대화를 종료합니다.")
            break

        # 프롬프트 형식에 맞춰 입력 구성
        prompt = f"User: {user_input}\nBot:"
        
        # 입력 텍스트를 토큰화하고 GPU로 이동
        input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)

        # 모델을 사용하여 답변 생성
        with torch.no_grad():
            output_sequences = model.generate(
                input_ids=input_ids,
                max_length=128,
                temperature=0.7,
                top_k=50,
                top_p=0.95,
                repetition_penalty=1.2,
                do_sample=True,
                num_return_sequences=1,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id # pad_token_id 설정 추가
            )
        
        # 생성된 텍스트에서 입력 프롬프트를 제외하고 답변만 추출
        response_text = tokenizer.decode(output_sequences[0], skip_special_tokens=True)
        # 입력 프롬프트 부분을 제거하여 순수한 답변만 남김
        bot_response = response_text.replace(prompt, "").strip()

        print(f"Bot: {bot_response}")

    except KeyboardInterrupt:
        print("\n대화를 종료합니다.")
        break

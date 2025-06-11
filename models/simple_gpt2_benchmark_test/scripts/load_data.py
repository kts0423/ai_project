from datasets import load_dataset
import os

def get_dataloader(split="train", batch_size=1):
    # HF Datasets 캐시 경로 (D드라이브)
    os.environ["HF_DATASETS_CACHE"] = "/mnt/d/ai-data/cache/datasets"

    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split=split)
    texts = [t for t in ds["text"] if t and t.strip()]  # 빈 문자열 제거

    batch_size = int(batch_size) if not callable(batch_size) else int(batch_size())

    # 빈 데이터셋 대비
    if len(texts) == 0:
        raise ValueError(f"No texts loaded for split '{split}'")

    return [texts[i : i + batch_size] for i in range(0, len(texts), batch_size)]

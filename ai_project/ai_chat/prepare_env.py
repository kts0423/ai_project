import os
import shutil
import fileinput

# 기준 경로
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# 원본 (D드라이브) → 대상 (프로젝트 폴더)
ORIG_DIR = "/mnt/d/aihub/processed"
ORIG_DATA = os.path.join(ORIG_DIR, "aihub_daily.jsonl")
ORIG_CACHE_TRAIN = os.path.join(ORIG_DIR, "cached", "tok_train")
ORIG_CACHE_EVAL  = os.path.join(ORIG_DIR, "cached", "tok_eval")

TARGET_DATA = os.path.join(BASE_DIR, "aihub_daily.jsonl")
TARGET_CACHE_TRAIN = os.path.join(BASE_DIR, "cache", "tok_train")
TARGET_CACHE_EVAL  = os.path.join(BASE_DIR, "cache", "tok_eval")

def safe_copy(src, dst):
    if not os.path.exists(dst):
        print(f"📁 Copying {src} → {dst}")
        shutil.copytree(src, dst)
    else:
        print(f"✅ Exists: {dst}")

def run():
    os.makedirs(os.path.join(BASE_DIR, "cache"), exist_ok=True)
    os.makedirs(os.path.join(BASE_DIR, "output"), exist_ok=True)
    os.makedirs(os.path.join(BASE_DIR, "runs"), exist_ok=True)

    if not os.path.exists(TARGET_DATA):
        shutil.copy2(ORIG_DATA, TARGET_DATA)
        print(f"📄 Copied aihub_daily.jsonl")
    else:
        print(f"✅ aihub_daily.jsonl already exists")

    safe_copy(ORIG_CACHE_TRAIN, TARGET_CACHE_TRAIN)
    safe_copy(ORIG_CACHE_EVAL, TARGET_CACHE_EVAL)

    replacements = {
        "/mnt/d/aihub/processed/aihub_daily.jsonl": TARGET_DATA,
        "/mnt/d/aihub/processed/finetuned_daily": os.path.join(BASE_DIR, "output"),
        "/mnt/d/aihub/processed/runs": os.path.join(BASE_DIR, "runs"),
        "/mnt/d/aihub/processed/cached": os.path.join(BASE_DIR, "cache"),
    }

    with fileinput.FileInput("train_daily.py", inplace=True, backup=".bak") as file:
        for line in file:
            for old, new in replacements.items():
                line = line.replace(old, new)
            print(line, end='')

    print("✅ Environment is ready.")

if __name__ == "__main__":
    run()

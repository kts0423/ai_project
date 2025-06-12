#!/usr/bin/env python3
import os
import re
import csv
import subprocess

def run_dynamic(bs, multi):
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = "0,1" if multi else "0"
    cmd = ["python", "dynamic_bench_gpt2.py", "--batch_size", str(bs)]
    proc = subprocess.run(cmd, capture_output=True, text=True, env=env)
    out = proc.stdout
    try:
        # “All done in X.XXXs” 파싱
        m = re.search(r"All done in ([0-9.]+)s", out)
        return float(m.group(1))
    except:
        # 실패 시 None 리턴
        return None

def main():
    batch_sizes = [1,4,8,16]
    results = []
    for bs in batch_sizes:
        t1 = run_dynamic(bs, False)
        t2 = run_dynamic(bs, True)
        if t2 is None:
            # 스킵하거나 inf 기록
            print(f"batch_size={bs} multi-GPU OOM, skipping")
            continue
        results.append({
            "batch_size": bs,
            "single_gpu_sec": t1,
            "multi_gpu_sec": t2,
            "speedup": t1 / t2
        })

    # CSV로 저장
    with open("results.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)
    print("✅ Done! See results.csv")

if __name__ == "__main__":
    main()

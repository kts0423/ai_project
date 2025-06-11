#!/usr/bin/env python3
import csv
import subprocess
import re
import os

# 비교할 모드별 실행 커맨드 템플릿
modes = {
    "single":   "CUDA_VISIBLE_DEVICES=0 python simple_bench_gpt2.py --batch_size {}",
    "dynamic":  "CUDA_VISIBLE_DEVICES=0,1 python dynamic_bench_gpt2.py --batch_size {}",
    "pipeline": "./pipeline_bench_gpt2.py --batch_size {}"
}

# 테스트할 배치 사이즈 리스트
batch_sizes = [4, 8, 16]

# 결과를 쓸 CSV 파일 열기
with open("compare_results.csv", "w", newline="") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["mode", "batch_size", "time_s"])

    # 각 모드, 각 배치 사이즈에 대해 실행
    for mode, cmd_tpl in modes.items():
        for bs in batch_sizes:
            cmd = cmd_tpl.format(bs)
            print(f">>> Running {mode} mode, batch_size={bs}")
            proc = subprocess.run(cmd, shell=True, capture_output=True, text=True, env=os.environ.copy())
            output = proc.stdout + proc.stderr

            # "All done in X.XXXs" 추출
            m = re.search(r"All done in ([0-9.]+)s", output)
            time_s = float(m.group(1)) if m else None

            writer.writerow([mode, bs, time_s])

print("✅ Done! See compare_results.csv for the summary.")

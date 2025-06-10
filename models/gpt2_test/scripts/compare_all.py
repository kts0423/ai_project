#!/usr/bin/env python3
import csv, subprocess

modes = [
    ("single", "CUDA_VISIBLE_DEVICES=0 python simple_bench_gpt2.py --batch_size {}"),
    ("dp",     "CUDA_VISIBLE_DEVICES=0,1 python dynamic_bench_gpt2.py --batch_size {}"),
    ("pp",     "./pipeline_bench_gpt2.py --batch_size {}")
]
batch_sizes = [1,4,8,16]

with open("all_results.csv","w", newline="") as f:
    writer=csv.writer(f)
    writer.writerow(["mode","batch_size","sec"])
    for name,cmd in modes:
        for bs in batch_sizes:
            full = cmd.format(bs)
            print(f">>> {name}, bs={bs}")
            p = subprocess.run(full, shell=True, capture_output=True, text=True)
            out = p.stdout
            # 마지막 줄에 “All done in X.XXXs” 찾아 파싱
            import re
            m = re.search(r"All done in ([0-9.]+)s", out)
            sec = float(m.group(1)) if m else None
            writer.writerow([name, bs, sec])
print("Completed: all_results.csv")

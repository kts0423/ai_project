import torch.multiprocessing as mp
mp.set_start_method("fork", force=True)

import json
from train_daily import trainer, model, OUTPUT_DIR

with open("best_config.json") as f:
    best = json.load(f)

trainer.args.dataloader_num_workers = best["nw"]
trainer.args.per_device_train_batch_size = best["bs"]
trainer.args.gradient_accumulation_steps = best["acc"]
model.peft_config["r"] = best["r"]
model.peft_config["lora_dropout"] = best["d"]

trainer.train()
metrics = trainer.evaluate()
for k, v in metrics.items():
    print(f"{k}: {v:.4f}")
model.save_pretrained(OUTPUT_DIR)
print(f"✅ Model saved at: {OUTPUT_DIR}")

"""LoRA SFT for v2 messages-format data (CoT + scene-anchored answers).

Same as scripts/train_lora.py but:
  - Reads synth_unified.jsonl directly (messages format)
  - Uses the v2 SYSTEM_PROMPT from toolbox/synth/prompts.py
  - Loss masked over `<thinking>...<answer>...</answer>` only

Usage
-----
    CUDA_VISIBLE_DEVICES=2 python scripts/train_lora_cot.py \\
        --train  data/cities/zurich/synth_unified.jsonl \\
        --output results/lora_zurich_cot \\
        --epochs 2
"""

import argparse
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset
from transformers import (
    AutoProcessor,
    Qwen2_5_VLForConditionalGeneration,
    Trainer,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model
from qwen_vl_utils import process_vision_info

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "toolbox"))
from synth.prompts import SYSTEM_PROMPT  # noqa: E402

MODEL_PATH = "/pub/evaluation_group/ning/test/models/Qwen2.5-VL-7B-Instruct"


class MessagesJsonlDataset(Dataset):
    """Reads {image, messages, _meta} rows. messages = [system,user,assistant]."""

    def __init__(self, path: str, max_samples: int | None = None,
                  val_frac: float = 0.0, val: bool = False, seed: int = 42):
        rows = []
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                r = json.loads(line)
                msgs = r.get("messages", [])
                user = next((m["content"] for m in msgs if m["role"] == "user"), None)
                asst = next((m["content"] for m in msgs if m["role"] == "assistant"), None)
                if not user or not asst:
                    continue
                rows.append({"image": r["image"], "user": user, "assistant": asst})

        if max_samples is not None:
            rows = rows[:max_samples]

        if val_frac > 0:
            import random as _r
            rng = _r.Random(seed)
            rng.shuffle(rows)
            n_val = max(1, int(len(rows) * val_frac))
            self.records = rows[:n_val] if val else rows[n_val:]
        else:
            self.records = rows

    def __len__(self):
        return len(self.records)

    def __getitem__(self, i):
        return self.records[i]


@dataclass
class CotCollator:
    processor: Any

    def __call__(self, batch):
        ids_list, labels_list, attn_list, img_list = [], [], [], []
        for ex in batch:
            user_content = [
                {"type": "image", "image": f"file://{ex['image']}"},
                {"type": "text", "text": ex["user"]},
            ]
            prompt_msgs = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_content},
            ]
            full_msgs = prompt_msgs + [
                {"role": "assistant", "content": ex["assistant"]},
            ]

            prompt_text = self.processor.apply_chat_template(
                prompt_msgs, tokenize=False, add_generation_prompt=True)
            full_text = self.processor.apply_chat_template(
                full_msgs, tokenize=False, add_generation_prompt=False)

            image_inputs, _ = process_vision_info(full_msgs)

            prompt_enc = self.processor(
                text=[prompt_text], images=image_inputs,
                return_tensors="pt", padding=False)
            full_enc = self.processor(
                text=[full_text], images=image_inputs,
                return_tensors="pt", padding=False)

            input_ids = full_enc["input_ids"][0]
            attn = full_enc["attention_mask"][0]
            prompt_len = prompt_enc["input_ids"].shape[1]
            labels = input_ids.clone()
            labels[:prompt_len] = -100  # train only on assistant tokens

            ids_list.append(input_ids)
            labels_list.append(labels)
            attn_list.append(attn)
            img_list.append({
                "pixel_values": full_enc["pixel_values"],
                "image_grid_thw": full_enc["image_grid_thw"],
            })

        max_len = max(x.shape[0] for x in ids_list)
        pad_id = self.processor.tokenizer.pad_token_id or 0
        def pad_to(x, v):
            o = torch.full((max_len,), v, dtype=x.dtype); o[: x.shape[0]] = x; return o

        return {
            "input_ids":   torch.stack([pad_to(x, pad_id) for x in ids_list]),
            "attention_mask": torch.stack([pad_to(x, 0) for x in attn_list]),
            "labels":       torch.stack([pad_to(x, -100) for x in labels_list]),
            "pixel_values": torch.cat([d["pixel_values"] for d in img_list], dim=0),
            "image_grid_thw": torch.cat([d["image_grid_thw"] for d in img_list], dim=0),
        }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True,
                    help="synth_unified.jsonl (messages format)")
    ap.add_argument("--val", default=None,
                    help="optional separate val jsonl; else hold out 10% of train")
    ap.add_argument("--output", required=True)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--batch_size", type=int, default=1)
    ap.add_argument("--grad_accum", type=int, default=8)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--max_samples", type=int, default=None)
    ap.add_argument("--lora_r", type=int, default=16)
    ap.add_argument("--lora_alpha", type=int, default=32)
    args = ap.parse_args()

    print(f"[train] loading model from {MODEL_PATH}")
    # Cap image tokens to keep training memory under ~60GB on shared L20X
    # (full-res Qwen2.5-VL images can exceed 1M pixels → 6k+ tokens per image
    # which is the main driver of GPU memory at backward).
    processor = AutoProcessor.from_pretrained(
        MODEL_PATH, trust_remote_code=True,
        min_pixels=256 * 256, max_pixels=448 * 448)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MODEL_PATH, torch_dtype=torch.bfloat16, attn_implementation="sdpa",
        trust_remote_code=True)
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()

    lora_cfg = LoraConfig(
        r=args.lora_r, lora_alpha=args.lora_alpha,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

    if args.val:
        train_ds = MessagesJsonlDataset(args.train, max_samples=args.max_samples)
        val_ds = MessagesJsonlDataset(args.val)
    else:
        train_ds = MessagesJsonlDataset(args.train, max_samples=args.max_samples,
                                         val_frac=0.1, val=False)
        val_ds = MessagesJsonlDataset(args.train, max_samples=args.max_samples,
                                       val_frac=0.1, val=True)
    print(f"[train] train={len(train_ds)}  val={len(val_ds)}")

    out = Path(args.output); out.mkdir(parents=True, exist_ok=True)
    targs = TrainingArguments(
        output_dir=str(out),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        bf16=True,
        warmup_ratio=0.05,
        logging_steps=10,
        save_strategy="epoch",
        eval_strategy="epoch" if len(val_ds) else "no",
        save_total_limit=2,
        report_to="none",
        remove_unused_columns=False,
    )

    trainer = Trainer(
        model=model, args=targs,
        train_dataset=train_ds,
        eval_dataset=val_ds if len(val_ds) else None,
        data_collator=CotCollator(processor),
    )
    trainer.train()
    trainer.save_model(str(out))
    processor.save_pretrained(str(out))
    print(f"[train] adapter saved → {out}")


if __name__ == "__main__":
    main()

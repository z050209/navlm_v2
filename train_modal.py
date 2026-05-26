"""Modal LoRA fine-tuning of Qwen2.5-VL-7B for NavLM — variant-aware.

One Modal app that trains the three L-* conditions of slide 4 by
swapping which dataset variant it consumes. The variant is the ONLY
difference between L-given / L-implicit / L-explicit at training time
(the prompts at eval time differ too, but that lives in eval_modal.py).

  variant = given      → user msg keeps heading; <thinking> unchanged
  variant = implicit   → user msg has no heading; <thinking> has no
                         INFERRED_HEADING step
  variant = explicit   → user msg has no heading; <thinking> still
                         spells out INFERRED_HEADING (model must learn
                         to *infer* the heading from the photo)

`src.derive_variants` produces the three .jsonl files (one row per
sample). Each row is the Qwen2.5-VL chat-template shape:

  {"image_rel": "<video>/<frame_id>.jpg",
   "messages": [{role, content:[{type:"image"|"text", ...}]}, ...]}

The `{"type": "image"}` placeholder in the user content is what tells
`processor.apply_chat_template` to splice in the image-token IDs —
without it the model trains text-only and never attends to the photo.

Usage (one variant):
    modal setup
    modal secret create huggingface HF_TOKEN=hf_xxx
    modal run train_modal.py --variant given
    modal run train_modal.py --variant explicit --epochs 3
    modal run train_modal.py --variant implicit --limit 8     # smoke
    modal volume get navlm-ckpts /lora_explicit_r16_e2 ./     # pull

For the full sweep across 3 variants + 12 eval cells, use
`python experiments.py --mode all`.
"""

import json
from pathlib import Path

import modal

app = modal.App("navlm-train")

train_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch==2.5.1", "torchvision",
        "transformers>=4.49", "peft>=0.13",
        "bitsandbytes>=0.44", "accelerate>=1.0", "datasets",
        "qwen-vl-utils", "huggingface_hub", "pillow",
    )
)

ckpts = modal.Volume.from_name("navlm-ckpts", create_if_missing=True)
data_vol = modal.Volume.from_name("navlm-data", create_if_missing=True)

BASE_MODEL = "Qwen/Qwen2.5-VL-7B-Instruct"
LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj"]
FRAMES_ROOT = "/data/frames"      # frames volume mount inside the
                                  # container (see DEV_MANUAL §4.5)


def _resolve_image(image_rel: str, frames_root: str) -> str:
    """Resolve `image_rel` (relative to <frames_root>). Accept absolute
    paths too — if the file exists locally we don't rewrite it."""
    p = Path(image_rel)
    if p.is_absolute() and p.exists():
        return str(p)
    return str(Path(frames_root) / image_rel)


@app.function(
    image=train_image,
    gpu="A100-80GB",
    timeout=6 * 3600,
    volumes={"/ckpts": ckpts, "/data": data_vol},
    secrets=[modal.Secret.from_name("huggingface")],
)
def train_lora(variant: str = "given",
               epochs: int = 2,
               lr: float = 2e-4,
               lora_r: int = 16,
               lora_alpha: int = 32,
               val_frac: float = 0.1,
               limit: int = 0) -> dict:
    """LoRA SFT of Qwen2.5-VL-7B on the chosen variant. Returns a
    summary dict with the adapter path + final val loss + history."""
    import torch
    from PIL import Image
    from peft import LoraConfig, get_peft_model
    from transformers import (AutoProcessor, BitsAndBytesConfig,
                              Qwen2_5_VLForConditionalGeneration,
                              Trainer, TrainingArguments)

    assert variant in ("given", "implicit", "explicit"), variant
    data_path = Path(f"/data/sft/{variant}.jsonl")
    assert data_path.exists(), (
        f"{data_path} not on the navlm-data volume — upload with "
        f"`modal volume put navlm-data data/sft/{variant}.jsonl "
        f"/sft/{variant}.jsonl`.")
    rows = [json.loads(l) for l in data_path.open(encoding="utf-8")
            if l.strip()]
    if limit:
        rows = rows[:limit]
    print(f"[train.{variant}] {len(rows)} samples", flush=True)

    n_val = max(1, int(len(rows) * val_frac))
    val_rows, train_rows = rows[:n_val], rows[n_val:]
    print(f"[train.{variant}] train={len(train_rows)} val={len(val_rows)} "
          f"· {epochs} epochs · lr={lr} · r={lora_r}", flush=True)

    bnb = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    processor = AutoProcessor.from_pretrained(BASE_MODEL,
                                              max_pixels=448 * 448)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        BASE_MODEL, quantization_config=bnb,
        torch_dtype=torch.bfloat16, device_map="auto",
    )
    model = get_peft_model(model, LoraConfig(
        r=lora_r, lora_alpha=lora_alpha, lora_dropout=0.05,
        target_modules=LORA_TARGETS, task_type="CAUSAL_LM",
    ))
    model.print_trainable_parameters()

    def collate(batch):
        """Apply the Qwen2.5-VL chat template + image splicing.

        Each row already has a `messages` list with a `{"type":
        "image"}` placeholder inside the user content; we feed the
        PIL via the processor's `images=` argument and it replaces
        the placeholder with the right vision-token IDs."""
        texts, images = [], []
        for row in batch:
            img_path = _resolve_image(row["image_rel"], FRAMES_ROOT)
            try:
                img = Image.open(img_path).convert("RGB")
            except (FileNotFoundError, OSError) as e:
                print(f"  skip {img_path}: {e}", flush=True)
                continue
            text = processor.apply_chat_template(
                row["messages"], tokenize=False,
                add_generation_prompt=False)
            texts.append(text); images.append(img)
        if not texts:
            raise RuntimeError("batch contained no resolvable images")
        enc = processor(text=texts, images=images, padding=True,
                        return_tensors="pt")
        labels = enc["input_ids"].clone()
        labels[labels == processor.tokenizer.pad_token_id] = -100
        # also mask Qwen2.5-VL's image-token IDs (loss only on text)
        for tok in ("<|image_pad|>", "<|vision_start|>",
                    "<|vision_end|>"):
            tid = processor.tokenizer.convert_tokens_to_ids(tok)
            if isinstance(tid, int) and tid >= 0:
                labels[labels == tid] = -100
        enc["labels"] = labels
        return enc

    out_dir = f"/ckpts/lora_{variant}_r{lora_r}_e{epochs}"
    args = TrainingArguments(
        output_dir=out_dir + "/_trainer", num_train_epochs=epochs,
        per_device_train_batch_size=1, gradient_accumulation_steps=8,
        learning_rate=lr, lr_scheduler_type="cosine", warmup_ratio=0.03,
        bf16=True,
        logging_steps=10, eval_strategy="epoch",
        per_device_eval_batch_size=1,
        save_strategy="no", report_to=[],
        remove_unused_columns=False,
    )
    trainer = Trainer(model=model, args=args,
                      train_dataset=train_rows, eval_dataset=val_rows,
                      data_collator=collate)
    trainer.train()
    val_metrics = trainer.evaluate()
    print(f"[train.{variant}] final val: {val_metrics}", flush=True)

    model.save_pretrained(out_dir)
    hist = [{"step": s.get("step"), "loss": s.get("loss"),
             "eval_loss": s.get("eval_loss"), "epoch": s.get("epoch")}
            for s in trainer.state.log_history]
    (Path(out_dir) / "history.json").write_text(json.dumps(hist, indent=2))
    (Path(out_dir) / "summary.json").write_text(json.dumps({
        "variant": variant, "epochs": epochs, "lr": lr,
        "lora_r": lora_r, "lora_alpha": lora_alpha,
        "n_train": len(train_rows), "n_val": len(val_rows),
        "final_eval_loss": val_metrics.get("eval_loss"),
    }, indent=2))
    ckpts.commit()
    print(f"[train.{variant}] adapter saved -> {out_dir}", flush=True)
    return {"adapter_path": out_dir,
            "n_train": len(train_rows), "n_val": len(val_rows),
            "final_eval_loss": val_metrics.get("eval_loss"),
            "history": hist}


@app.local_entrypoint()
def main(variant: str = "given", epochs: int = 2, lr: float = 2e-4,
         limit: int = 0):
    """Train one variant. For the full 3-variant sweep, use
    `python experiments.py --mode train`."""
    result = train_lora.remote(variant=variant, epochs=epochs, lr=lr,
                               limit=limit)
    print("=== TRAIN DONE ===")
    print(json.dumps({k: v for k, v in result.items() if k != "history"},
                     indent=2))
    print(f"Pull adapter:  modal volume get navlm-ckpts "
          f"{result['adapter_path']} ./")

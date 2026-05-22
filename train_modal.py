"""Modal LoRA fine-tuning of Qwen2.5-VL-7B for NavLM.

Trains a LoRA adapter on the NavLM instruction-tuning dataset on a Modal
A100, and saves it to a persistent volume. See logs/infra.md section 10.

    modal setup                                       # one-time auth
    modal secret create huggingface HF_TOKEN=hf_xxx   # one-time
    modal run train_modal.py                          # default 2 epochs
    modal run train_modal.py --epochs 3 --lr 1e-4
    modal run train_modal.py --limit 5                # 5-sample smoke test
    modal volume get navlm-ckpts /lora_r16_e2 ./      # pull the adapter back

The synth dataset is expected as a Hugging Face dataset repo with rows
{image: <PIL>, messages: [{role, content}]} — produced by the Phase B
annotation stage. Until that exists, run with --limit for a shape check.
"""

import modal

app = modal.App("navlm-train")

# ── container image (deps declared once; Modal caches the build) ──────
train_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch", "torchvision", "transformers>=4.49", "peft",
        "bitsandbytes", "accelerate", "datasets", "qwen-vl-utils",
        "huggingface_hub", "pillow",
    )
)

# persistent disk for adapters — survives across runs
ckpts = modal.Volume.from_name("navlm-ckpts", create_if_missing=True)

BASE_MODEL = "Qwen/Qwen2.5-VL-7B-Instruct"
DATASET_REPO = "z050209/navlm-synth"     # HF dataset, created by Phase B
LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj"]


@app.function(
    image=train_image,
    gpu="A100-80GB",                      # ~$3.73/hr
    timeout=6 * 3600,
    volumes={"/ckpts": ckpts},
    secrets=[modal.Secret.from_name("huggingface")],
)
def train_lora(epochs: int = 2, lr: float = 2e-4,
               lora_r: int = 16, lora_alpha: int = 32,
               limit: int = 0) -> str:
    """LoRA SFT of Qwen2.5-VL-7B. Returns the adapter path on the volume."""
    import torch
    from datasets import load_dataset
    from peft import LoraConfig, get_peft_model
    from transformers import (AutoProcessor, BitsAndBytesConfig,
                              Qwen2_5_VLForConditionalGeneration,
                              Trainer, TrainingArguments)

    # 4-bit NF4 base, BF16 compute
    bnb = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    processor = AutoProcessor.from_pretrained(BASE_MODEL,
                                              max_pixels=448 * 448)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        BASE_MODEL, quantization_config=bnb, torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    model = get_peft_model(model, LoraConfig(
        r=lora_r, lora_alpha=lora_alpha, lora_dropout=0.05,
        target_modules=LORA_TARGETS, task_type="CAUSAL_LM",
    ))
    model.print_trainable_parameters()

    ds = load_dataset(DATASET_REPO, split="train")
    if limit:
        ds = ds.select(range(min(limit, len(ds))))
    print(f"[train] {len(ds)} samples · {epochs} epochs · lr={lr}")

    def collate(batch):
        """Apply the Qwen2.5-VL chat template, tokenize, mask padding."""
        texts = [processor.apply_chat_template(
            row["messages"], tokenize=False, add_generation_prompt=False)
            for row in batch]
        images = [row["image"] for row in batch]
        enc = processor(text=texts, images=images, padding=True,
                        return_tensors="pt")
        labels = enc["input_ids"].clone()
        labels[labels == processor.tokenizer.pad_token_id] = -100
        enc["labels"] = labels
        return enc

    args = TrainingArguments(
        output_dir="/ckpts/_run", num_train_epochs=epochs,
        per_device_train_batch_size=1, gradient_accumulation_steps=8,
        learning_rate=lr, lr_scheduler_type="cosine", warmup_ratio=0.03,
        bf16=True, logging_steps=10, save_strategy="no", report_to=[],
    )
    Trainer(model=model, args=args, train_dataset=ds,
            data_collator=collate).train()

    out = f"/ckpts/lora_r{lora_r}_e{epochs}"
    model.save_pretrained(out)
    ckpts.commit()                        # persist before the GPU frees
    print(f"[train] adapter saved -> {out}")
    return out


@app.local_entrypoint()
def main(epochs: int = 2, lr: float = 2e-4, limit: int = 0):
    print("adapter:", train_lora.remote(epochs=epochs, lr=lr, limit=limit))

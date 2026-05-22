# NavLM — Infrastructure, Budget & Cost Tracking

Last updated: 2026-05-22. Single source of truth for accounts, credits,
API pricing, and where to check spend.

> No secret key strings are stored in this file. Keys live in the GCP
> console and in a gitignored `.env`. This file only names them.

---

## 1. Login status on this machine (2026-05-22)

| Service | Status | Account |
|---------|--------|---------|
| GitHub | ✅ logged in | `z050209` |
| gcloud | ✅ logged in | `z050209@gmail.com`, project `cs231n-navlm-2026` |
| Modal | ⚠️ CLI installed, not authenticated | workspace `z050209` — run `modal setup` |
| Hugging Face | ❌ not set up | needed for dataset/checkpoint hosting |

---

## 2. Cloud credits — total $350

| Platform | Account | Credit | Role |
|----------|---------|--------|------|
| **GCP** | `z050209@gmail.com` | **$50** (education coupon via para2046@stanford.edu) | Street View + Gemini APIs |
| **Modal** | `z050209` workspace | **$200** | GPU training & eval (primary) |
| **AWS** | not set up | **$100** | reserve / backup GPU |

---

## 3. GCP

| Item | Value |
|------|-------|
| Project | `cs231n-navlm-2026` |
| Billing account | `010F4B-3C316F-B7FE47` ("Billing Account for Education") |
| Budget alert | `navlm-50usd` — $50, emails at 50 / 75 / 90 / 100% |
| Dev VM `navlm-dev` | **DELETED 2026-05-21** (was unused; stopped ~$34/mo SSD charge) |

**API keys** (strings in the GCP console, not here):
| Key name | UID | Used for |
|----------|-----|----------|
| `streetview-test` | `d87c5c3c-4a61-414a-abea-39c9be8de870` | Street View Static + Maps |
| `gemini-navlm` | `da33c3ba-7027-4fda-ae42-a3b1d51039fc` | Gemini / Generative Language API |

---

## 4. API & GPU pricing

| Service | Price |
|---------|-------|
| Street View Static API | **$7 / 1,000 images** ($0.007 each) |
| Street View *metadata* endpoint | **FREE** |
| Gemini 2.5 Flash | ~$0.30 in / $2.50 out per 1M tokens |
| Gemini 2.5 Pro | ~$1.25 in / $10 out per 1M tokens |
| Modal A100 80GB | ~$3.73 / hr |
| Modal A10G 24GB | ~$1.10 / hr |
| Mapillary API | free |
| OSM / osmnx | free |

---

## 5. Spend so far

| Item | Cost |
|------|------|
| Street View — 830 images (712 grid test + 118 POI) | ~$5.81 |
| Gemini — 1 test call | ~$0.0001 |
| GCP VM — ran briefly 2026-05-21, deleted same day | small |
| **Total API spend** | **~$5.81 of $50 GCP credit** |

---

## 6. Cost-planning estimates (not yet spent)

| Task | Estimate |
|------|----------|
| Street View full crawl — 1,915 panos × 4 headings | ~$54 |
| Gemini POI scan — all quality-filtered frames | < $5 |
| Gemini Pro instruction annotation | depends on sample count (~$10–30) |
| Modal LoRA training | ~$22 / run, 3–6 h on A100 |
| Modal eval | ~$2 / run |

---

## 7. Where to check cost — URLs

| What | URL |
|------|-----|
| GCP billing overview | https://console.cloud.google.com/billing/010F4B-3C316F-B7FE47 |
| Daily cost report (set Group-by → Day) | https://console.cloud.google.com/billing/010F4B-3C316F-B7FE47/reports |
| **$50 credit remaining** | https://console.cloud.google.com/billing/010F4B-3C316F-B7FE47/credits |
| Budget alert `navlm-50usd` | https://console.cloud.google.com/billing/010F4B-3C316F-B7FE47/budgets |
| API keys | https://console.cloud.google.com/apis/credentials?project=cs231n-navlm-2026 |
| Modal usage | https://modal.com (workspace dashboard) |
| Local spend log (real-time, API only) | `reference/track_spend.py` → `costs/daily_spend.md` + `preview/spend.html` |

> GCP billing data lags ~24 h. The local spend log is real-time but
> covers only API spend, not GPU/VM compute.

---

## 8. Training

| Item | Value |
|------|-------|
| Base model | Qwen2.5-VL-7B-Instruct |
| Method | LoRA — r=16, alpha=32, 4-bit NF4 base, BF16 adapters |
| Training compute | Modal A100 80 GB |
| Zero-shot baselines | local — NVIDIA RTX 3060 12 GB |
| Annotation teacher VLM | Gemini Pro (planned) |

---

## 9. Tooling on this machine

| Tool | Status |
|------|--------|
| git, gh, gcloud | ✅ installed |
| Python venv (`navlm_ss/.venv`) | ✅ torch 2.5.1+cu124, transformers, torchvision, requests |
| ffmpeg | ❌ not installed (needed for frame extraction) |
| Modal CLI | ✅ installed (v1.4.3) · ⚠️ run `modal setup` to authenticate |
| Hugging Face CLI (`hf`) | ❌ not installed |
| `imagehash`, `yt-dlp`, `osmnx`, `opencv` | ❌ not in venv (needed for v2) |

---

## 10. Running model training on Modal

Modal runs Python functions on cloud GPUs, billed **per-second only while
the function runs**. You write a normal `.py`, decorate a function with
`@app.function(gpu=...)`, and `modal run` it — Modal builds the container,
provisions the GPU, runs, and tears down.

### One-time setup

```bash
modal setup                                          # browser auth → ~/.modal.toml
modal secret create huggingface HF_TOKEN=hf_xxxxx    # so jobs can pull/push HF
```

### Core concepts

| Modal object | Purpose |
|--------------|---------|
| `modal.App` | a named project |
| `modal.Image` | the container (declare pip deps in code) |
| `modal.Volume` | persistent disk — survives between runs (checkpoints, data) |
| `modal.Secret` | injects API tokens as env vars (never hard-code keys) |
| `@app.function(gpu=...)` | the GPU job; `gpu` ∈ `T4`, `L4`, `A10G`, `A100-40GB`, `A100-80GB`, `H100` |
| `@app.local_entrypoint()` | what `modal run` calls on your laptop to launch it |

### Example — LoRA SFT of Qwen2.5-VL-7B (`train_modal.py`)

```python
import modal

app = modal.App("navlm-train")

# container image — declare deps once, Modal caches the build
train_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch", "transformers", "peft", "bitsandbytes",
                 "accelerate", "datasets", "qwen-vl-utils", "huggingface_hub")
)

# persistent disk for checkpoints (survives across runs)
ckpts = modal.Volume.from_name("navlm-ckpts", create_if_missing=True)

@app.function(
    image=train_image,
    gpu="A100-80GB",                       # ~$3.73/hr
    timeout=6 * 3600,                      # kill after 6h
    volumes={"/ckpts": ckpts},
    secrets=[modal.Secret.from_name("huggingface")],   # → $HF_TOKEN
)
def train_lora(epochs: int = 2, lr: float = 2e-4, lora_r: int = 16):
    from huggingface_hub import snapshot_download
    # pull base model + our synth dataset from Hugging Face
    base = snapshot_download("Qwen/Qwen2.5-VL-7B-Instruct")
    data = snapshot_download("z050209/navlm-synth", repo_type="dataset")

    # ... load base 4-bit (NF4), attach LoRA r=lora_r alpha=32,
    #     run SFTTrainer for `epochs` at `lr` ...

    out = f"/ckpts/lora_r{lora_r}_e{epochs}"
    # trainer.save_model(out)
    ckpts.commit()                         # persist before the GPU is freed
    return out

@app.local_entrypoint()
def main(epochs: int = 2, lr: float = 2e-4):
    print("adapter saved to:", train_lora.remote(epochs=epochs, lr=lr))
```

### Run it

```bash
modal run train_modal.py                     # default args, streams logs live
modal run train_modal.py --epochs 3 --lr 1e-4 # override
modal volume ls navlm-ckpts                   # see saved checkpoints
modal volume get navlm-ckpts /lora_r16_e2 ./  # download the adapter locally
modal app logs navlm-train                    # past run logs
```

`.remote()` runs on the cloud GPU; `.local()` runs in-process (handy for
debugging). Data/checkpoints stay on the `modal.Volume` or get pushed to
Hugging Face from inside the job.

### Cost

| GPU | Rate | A typical LoRA run (3–6 h) |
|-----|------|----------------------------|
| A100-80GB | ~$3.73/hr | **~$11–22** |
| A10G 24GB | ~$1.10/hr | cheaper; fine for eval / DINOv2 embedding |

Billed per-second of function runtime only — idle time costs nothing.
Reference Modal example already in repo: `reference/run_dinov2_modal.py`.

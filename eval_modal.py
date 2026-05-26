"""Modal eval for the 6 conditions × 2 ablations of slide 4.

Six conditions:                      Two ablations:
  B-given                              video   — held-out saturday_morning
  B-implicit                           poi     — held-out POI region
  B-explicit
  L-given      (needs lora_given_*)
  L-implicit   (needs lora_implicit_*)
  L-explicit   (needs lora_explicit_*)

For each test sample it generates one assistant turn (<thinking> +
<answer>) and scores the four metrics from `src.eval_metrics`:

  format_compliance · directional_accuracy ·
  checkpoint_validity · anchor_faithfulness

`anchor_faithfulness` calls Gemini 2.5 **Flash** through the AI Studio
endpoint (free tier — works for a yes/no anchor check). Skip it with
`--no-anchor` for smoke runs.

Volumes:
  navlm-ckpts → /ckpts     (LoRA adapters from train_modal.py)
  navlm-data  → /data      (eval_test_*.jsonl, frames/)
  navlm-eval  → /eval      (per-condition per-sample.jsonl + summary)

Outputs land at:
  /eval/<run_id>/<condition>__<ablation>/per_sample.jsonl
  /eval/<run_id>/<condition>__<ablation>/summary.json

Pull with `python pull_eval.py <run_id>`.

  modal run eval_modal.py --condition B-given --ablation video --limit 5 --no-anchor
  modal run eval_modal.py --condition L-explicit --ablation poi
  modal run eval_modal.py --condition all --ablation video        # 6 cells in one call
"""

import datetime
import json
from pathlib import Path

import modal

app = modal.App("navlm-eval")

eval_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch==2.5.1", "torchvision",
        "transformers>=4.49", "peft>=0.13",
        "bitsandbytes>=0.44", "accelerate>=1.0",
        "qwen-vl-utils", "pillow", "requests",
    )
)

ckpts = modal.Volume.from_name("navlm-ckpts", create_if_missing=True)
data_vol = modal.Volume.from_name("navlm-data", create_if_missing=True)
eval_vol = modal.Volume.from_name("navlm-eval", create_if_missing=True)

BASE_MODEL = "Qwen/Qwen2.5-VL-7B-Instruct"
FRAMES_ROOT = "/data/frames"
ALL_CONDITIONS = [
    "B-given", "B-implicit", "B-explicit",
    "L-given", "L-implicit", "L-explicit",
]
ALL_ABLATIONS = ["video", "poi"]

CONDITION_SPEC = {
    "B-given":    (False, "given"),
    "B-implicit": (False, "implicit"),
    "B-explicit": (False, "explicit"),
    "L-given":    (True,  "given"),
    "L-implicit": (True,  "implicit"),
    "L-explicit": (True,  "explicit"),
}

# Inline copies of the system prompts and user-message builder — must
# match src/derive_variants.py exactly, so a trained LoRA sees the
# same prompt at eval time it saw during training.
SYS_PROMPTS = {
    "given": (
        "You are a Zurich-local walking-tour guide. You give SPOKEN "
        "directions to a tourist who has sent you a phone photo of "
        "what they currently see plus their GPS, camera heading, and "
        "destination. Reply with <thinking>...</thinking>"
        "<answer>...</answer>; the answer is 2-4 short sentences, no "
        "compass words, no numbers, no GPS, anchored to a visible "
        "object."),
    "implicit": (
        "You are a Zurich-local walking-tour guide. You give SPOKEN "
        "directions to a tourist who has sent you a phone photo of "
        "what they currently see plus their GPS and destination. They "
        "have NO compass. Reply with <thinking>...</thinking>"
        "<answer>...</answer>; the answer is 2-4 short sentences, no "
        "compass words, no numbers, no GPS, anchored to a visible "
        "object."),
    "explicit": (
        "You are a Zurich-local walking-tour guide. You give SPOKEN "
        "directions to a tourist who has sent you a phone photo of "
        "what they currently see plus their GPS and destination. They "
        "have NO compass — INFER their facing from the photo. Reply "
        "with <thinking>...</thinking><answer>...</answer>; INSIDE "
        "<thinking> include a line 'INFERRED_HEADING: <0-359>' where "
        "you commit to your best guess of which way the camera is "
        "facing, reasoning from visible landmarks. The answer is 2-4 "
        "short sentences, no compass words, no numbers, no GPS, "
        "anchored to a visible object."),
}


def _resolve_image(image_rel: str) -> str:
    p = Path(image_rel)
    if p.is_absolute() and p.exists():
        return str(p)
    return str(Path(FRAMES_ROOT) / image_rel)


def build_eval_messages(sample, variant):
    """[messages] for one eval sample, matching derive_variants exactly.
    Returns (messages, image_path)."""
    lat, lon = sample["gps"]
    lines = [f"My GPS: {lat:.5f}, {lon:.5f}"]
    if variant == "given":
        lines.append(f"My camera heading: {sample['heading']:.0f}° "
                     "(0=N, 90=E)")
    lines += [
        f"Destination: {sample['dest_name']} "
        f"(first-segment bearing {sample['route_bearing']:.0f}°)",
        f"Walking-route distance: {sample['route_distance_m']:.0f} m",
    ]
    if sample.get("nearby_pois"):
        lines.append("Nearby POIs:")
        lines += [f"- {n}" for n in sample["nearby_pois"][:8]]
    lines.append("Tell me what to do, in 2-4 spoken sentences, "
                 "anchored to something I can actually see in the photo.")
    user_text = "\n".join(lines)

    image_rel = sample.get("image_rel") or \
        f"{sample['video']}/{sample['frame_id']}.jpg"
    messages = [
        {"role": "system",
         "content": [{"type": "text", "text": SYS_PROMPTS[variant]}]},
        {"role": "user",
         "content": [{"type": "image"},
                     {"type": "text", "text": user_text}]},
    ]
    return messages, _resolve_image(image_rel)


def _bake_scoring():
    """Inline copy of the metric core (keeps the Modal container free
    of the project source tree). Mirrors src/eval_metrics.py — edit
    BOTH if the logic changes."""
    import re

    ACTION_DELTA = {"continue ahead": 0.0, "turn left": -90.0,
                    "turn right": 90.0, "turn around": 180.0}

    def angle_diff(a, b):
        return ((a - b + 540) % 360) - 180

    def closed_loop(h, verb, B):
        return abs(angle_diff(h + ACTION_DELTA[verb], B))

    COMPASS = re.compile(
        r"\b(north|south|east|west|northeast|northwest|southeast|"
        r"southwest|n\.?e\.?|n\.?w\.?|s\.?e\.?|s\.?w\.?|"
        r"\d{1,3}\s*°|degrees?)\b", re.I)
    GPS = re.compile(r"\b\d{1,2}\.\d{4,}\b")

    def parse_verb(answer):
        for v in ACTION_DELTA:
            if re.search(r"\b" + re.escape(v) + r"\b", answer or "", re.I):
                return v
        return None

    def extract_anchor(answer):
        m = re.search(
            r"(?:turn\s+(?:left|right|around)|continue\s+ahead)\s+"
            r"(?:at|near|by|past|along|next to|towards?)\s+"
            r"(?:the\s+)?([^\s.,;!?][^\.,;!?]{0,60}?)\s*[.,;!?]",
            answer or "", re.I | re.U)
        return m.group(1).strip() if m else None

    def extract_checkpoint(answer):
        m = re.search(
            r"when\s+you\s+reach\s+(?:the\s+)?"
            r"([^\s.,;!?][^\.,;!?]{0,60}?)\s*[,.]",
            answer or "", re.I | re.U)
        return m.group(1).strip() if m else None

    def format_ok(raw, answer):
        if "<thinking>" not in raw or "</thinking>" not in raw:
            return False
        if "<answer>" not in raw or "</answer>" not in raw:
            return False
        n = len([s for s in re.split(r"(?<=[.!?])\s+", answer or "")
                 if s.strip()])
        if not (2 <= n <= 4):
            return False
        if COMPASS.search(answer or ""):
            return False
        if GPS.search(answer or ""):
            return False
        return True

    def directional_ok(heading, verb, route_bearing, max_delta=30.0):
        if verb is None or verb not in ACTION_DELTA:
            return False, None
        d = closed_loop(heading, verb, route_bearing)
        return d < max_delta, d

    return (format_ok, directional_ok, parse_verb,
            extract_anchor, extract_checkpoint)


def _anchor_check(image_path, anchor, api_key, timeout=30):
    """Gemini 2.5 Flash yes/no via the AI Studio endpoint (free tier).
    Returns (ok: bool, raw_reply: str). Falls back to ok=False on any
    error so a network blip doesn't fake-inflate the metric."""
    import base64
    import requests
    if not anchor:
        return True, ""
    if not api_key:
        return False, "no GEMINI_API_KEY (set Modal secret 'gemini')"
    url = ("https://generativelanguage.googleapis.com/v1beta/models/"
           "gemini-2.5-flash:generateContent")
    try:
        b64 = base64.b64encode(Path(image_path).read_bytes()).decode()
        payload = {
            "systemInstruction": {"parts": [{
                "text": "Answer with one word: YES or NO."}]},
            "contents": [{"role": "user", "parts": [
                {"text": f"Is '{anchor}' clearly visible in this "
                 "photo? YES or NO."},
                {"inlineData": {"mimeType": "image/jpeg", "data": b64}},
            ]}],
            "generationConfig": {"maxOutputTokens": 8,
                                 "temperature": 0.0},
        }
        r = requests.post(url, json=payload, params={"key": api_key},
                          timeout=timeout)
        text = r.text[:400]
    except Exception as e:
        return False, f"ERROR {type(e).__name__}: {e}"
    return ("yes" in text.lower()), text


@app.function(
    image=eval_image,
    gpu="A100-40GB",
    timeout=4 * 3600,
    volumes={"/ckpts": ckpts, "/data": data_vol, "/eval": eval_vol},
    secrets=[
        modal.Secret.from_name("huggingface"),
        # Optional — only needed when anchor checks are on. Modal
        # tolerates a missing optional secret if `required=False` is
        # NOT used, so we just try-read os.environ at call time.
    ],
)
def evaluate_condition(condition: str, ablation: str,
                       run_id: str, limit: int = 0,
                       lora_path: str = "",
                       no_anchor: bool = False) -> dict:
    """Run one (condition, ablation). Writes per-sample jsonl +
    summary.json to /eval/<run_id>/. Returns the summary."""
    import os
    import re
    import torch
    from PIL import Image
    from transformers import (AutoProcessor, BitsAndBytesConfig,
                              Qwen2_5_VLForConditionalGeneration)

    use_lora, variant = CONDITION_SPEC[condition]
    fmt_ok, dir_ok_fn, parse_verb, extract_anchor, extract_ckpt = \
        _bake_scoring()
    api_key = os.environ.get("GEMINI_API_KEY", "")

    test_path = Path(f"/data/eval/eval_test_{ablation}.jsonl")
    assert test_path.exists(), (
        f"{test_path} missing — upload with "
        f"`modal volume put navlm-data data/cities/zurich/"
        f"eval_test_{ablation}.jsonl /eval/eval_test_{ablation}.jsonl`")
    samples = [json.loads(l) for l in test_path.open(encoding="utf-8")
               if l.strip()]
    if limit:
        samples = samples[:limit]
    print(f"[eval.{condition}.{ablation}] N={len(samples)}  "
          f"use_lora={use_lora}  variant={variant}  "
          f"anchor={'OFF' if no_anchor else 'ON'}",
          flush=True)

    bnb = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True)
    processor = AutoProcessor.from_pretrained(BASE_MODEL,
                                              max_pixels=448 * 448)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        BASE_MODEL, quantization_config=bnb,
        torch_dtype=torch.bfloat16, device_map="auto")

    if use_lora:
        from peft import PeftModel
        lp = lora_path or f"/ckpts/lora_{variant}_r16_e2"
        assert Path(lp).exists(), (
            f"LoRA adapter {lp} not on /ckpts — train it first via "
            f"`modal run train_modal.py --variant {variant}`")
        model = PeftModel.from_pretrained(model, lp)
        print(f"[eval.{condition}.{ablation}] LoRA mounted: {lp}",
              flush=True)
    model.eval()

    out_dir = Path(f"/eval/{run_id}/{condition}__{ablation}")
    out_dir.mkdir(parents=True, exist_ok=True)
    per_sample = out_dir / "per_sample.jsonl"

    counts = {"format": 0, "dir": 0, "ckpt": 0, "anchor": 0,
              "pass_strict": 0}
    deltas = []
    with per_sample.open("w", encoding="utf-8") as fout:
        for i, s in enumerate(samples):
            messages, img_path = build_eval_messages(s, variant)
            try:
                img = Image.open(img_path).convert("RGB")
            except (FileNotFoundError, OSError) as e:
                print(f"  skip {s['frame_id']}: {e}", flush=True)
                continue
            text = processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True)
            enc = processor(text=[text], images=[img], padding=True,
                            return_tensors="pt").to(model.device)
            with torch.no_grad():
                gen = model.generate(
                    **enc, max_new_tokens=512, do_sample=False,
                    pad_token_id=processor.tokenizer.eos_token_id)
            raw = processor.batch_decode(
                gen[:, enc["input_ids"].shape[1]:],
                skip_special_tokens=True)[0]

            think_m = re.search(r"<thinking>(.*?)</thinking>", raw, re.S)
            answ_m = re.search(r"<answer>(.*?)</answer>", raw, re.S)
            thinking = think_m.group(1).strip() if think_m else ""
            answer = answ_m.group(1).strip() if answ_m else ""

            f_ok = fmt_ok(raw, answer)
            verb = parse_verb(answer)
            d_ok, delta = dir_ok_fn(s["heading"], verb,
                                    s["route_bearing"])
            anchor = extract_anchor(answer)
            ckpt = extract_ckpt(answer)
            c_ok = (ckpt is None)

            if no_anchor or not anchor:
                a_ok, a_raw = True, ""
            else:
                a_ok, a_raw = _anchor_check(img_path, anchor, api_key)

            pass_strict = bool(f_ok and d_ok and c_ok and a_ok)
            counts["format"] += int(f_ok)
            counts["dir"] += int(d_ok)
            counts["ckpt"] += int(c_ok)
            counts["anchor"] += int(a_ok)
            counts["pass_strict"] += int(pass_strict)
            if delta is not None:
                deltas.append(delta)

            fout.write(json.dumps({
                "video": s["video"], "frame_id": s["frame_id"],
                "dest_name": s["dest_name"],
                "raw": raw, "thinking": thinking, "answer": answer,
                "verb": verb, "anchor": anchor, "checkpoint": ckpt,
                "format_ok": f_ok, "dir_ok": d_ok, "delta": delta,
                "ckpt_ok": c_ok, "anchor_ok": a_ok,
                "anchor_raw": a_raw, "pass_strict": pass_strict,
            }, ensure_ascii=False) + "\n")
            if (i + 1) % 10 == 0:
                fout.flush()
                print(f"  [{i+1}/{len(samples)}] "
                      f"pass={counts['pass_strict']}/{i+1}",
                      flush=True)

    n = max(1, len(samples))
    summary = {
        "run_id": run_id, "condition": condition, "ablation": ablation,
        "variant": variant, "use_lora": use_lora, "n": len(samples),
        "format_rate":      counts["format"] / n,
        "directional_rate": counts["dir"] / n,
        "checkpoint_rate":  counts["ckpt"] / n,
        "anchor_rate":      counts["anchor"] / n,
        "pass_strict_rate": counts["pass_strict"] / n,
        "median_delta": (sorted(deltas)[len(deltas) // 2]
                         if deltas else None),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    eval_vol.commit()
    print(f"[eval.{condition}.{ablation}] {summary}", flush=True)
    return summary


@app.local_entrypoint()
def main(condition: str = "B-given",
         ablation: str = "video",
         limit: int = 0,
         lora_path: str = "",
         no_anchor: bool = False,
         run_id: str = ""):
    """Evaluate ONE (condition, ablation) — or pass condition='all'
    to sweep the 6 conditions for the chosen ablation in one call."""
    run_id = run_id or datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    conds = ALL_CONDITIONS if condition.lower() == "all" else [condition]
    assert ablation in ALL_ABLATIONS, ablation

    results = []
    for c in conds:
        print(f"=== {c} × {ablation}  (run {run_id}) ===")
        s = evaluate_condition.remote(
            condition=c, ablation=ablation, run_id=run_id, limit=limit,
            lora_path=lora_path, no_anchor=no_anchor)
        results.append(s)

    print()
    print(f"=== SUMMARY  ablation={ablation}  run_id={run_id} ===")
    print(f"{'condition':<12} {'N':>4}  {'format':>7} {'dir':>6} "
          f"{'ckpt':>6} {'anchor':>7} {'PASS':>6}")
    for s in results:
        print(f"{s['condition']:<12} {s['n']:>4d}  "
              f"{s['format_rate']:>7.2%} {s['directional_rate']:>6.2%} "
              f"{s['checkpoint_rate']:>6.2%} {s['anchor_rate']:>7.2%} "
              f"{s['pass_strict_rate']:>6.2%}")
    print(f"\nPull results:  python pull_eval.py {run_id}")

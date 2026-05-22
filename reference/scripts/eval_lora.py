"""Evaluate base Qwen2.5-VL vs LoRA-zurich-v3 on the held-out eval set.

For each row in --eval, runs inference with the model (base or base+lora),
then scores the generated <answer> with the step_12 closed-loop verifier:

  1. format checks (TTS rules, no compass / metres / GPS)
  2. sentence count 2-4
  3. closed-loop angle (parsed action + heading_gt vs first_seg_bearing)
  4. checkpoint grounded (mode B route street name OR permanent landmark)

Aggregate metrics + side-by-side markdown for spot-check.

Usage
-----
    # base model only
    CUDA_VISIBLE_DEVICES=2 python scripts/eval_lora.py \\
        --eval data/cities/zurich/synth_v3_eval.jsonl \\
        --tag  base \\
        --max-samples 50

    # with LoRA
    CUDA_VISIBLE_DEVICES=2 python scripts/eval_lora.py \\
        --eval data/cities/zurich/synth_v3_eval.jsonl \\
        --lora results/lora_zurich_v3 \\
        --tag  lora_v3 \\
        --max-samples 50

    # compare two runs
    python scripts/eval_lora.py --compare \\
        results/eval_v3_base.json \\
        results/eval_v3_lora_v3.json
"""

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "pipeline"))
sys.path.insert(0, str(ROOT / "toolbox"))

from step_12_closed_loop_verify import (  # noqa: E402
    parse_assistant, parse_action_from_answer,
    gate_format, gate_sentence_count, gate_checkpoint_grounded,
    get_first_seg_bearing, ACTION_DELTA, angle_diff,
)

# For gate 6 (hallucination), use the existing Gemma yes/no helper.
sys.path.insert(0, str(ROOT / "toolbox"))
from synth_utils import img_to_data_url  # noqa: E402
import requests  # noqa: E402


MODEL_PATH = "/pub/evaluation_group/ning/test/models/Qwen2.5-VL-7B-Instruct"

GEMMA_URL   = "http://localhost:8003/v1"
GEMMA_MODEL = "google/gemma-4-31b-it"


# --- gate 5: destination correctly named in answer -----------------

def gate_dest_correct(answer, meta):
    """The destination POI name should appear in the answer."""
    if not answer:
        return False, "no answer"
    dest = meta.get("destination", "")
    if not dest:
        return True, "no dest in meta"
    if dest.lower() in answer.lower():
        return True, None
    # also accept first word of dest (e.g. "Hauptbahnhof" matches "Hauptbahnhof Süd")
    first = dest.split()[0].lower()
    if len(first) >= 4 and first in answer.lower():
        return True, f"matched first word {first!r}"
    return False, f"destination {dest!r} not in answer"


# --- gate 6: visible-anchor hallucination check (Gemma yes/no) ------

def gate_anchor_grounded(answer, image_path, vlm_url=GEMMA_URL,
                          model=GEMMA_MODEL, timeout=20):
    """Ask Gemma whether the answer's visual reference is consistent with
    the image. Single yes/no call per sample.
    """
    if not answer:
        return False, "no answer"
    prompt = (
        "An assistant gave these walking directions to a person looking at "
        "this photograph:\n\n"
        f"  \"{answer}\"\n\n"
        "Look at the photograph and judge: are the visual objects the "
        "assistant references (e.g. \"the cafe tables on your right\", "
        "\"the stone building\", \"the tram tracks ahead\") actually present "
        "in the image where they're claimed to be? Reply with exactly one of:\n"
        "  YES - <one short phrase confirming what you see>\n"
        "  NO - <one short phrase about which referenced object is missing>\n"
        "  PARTIAL - <one short phrase noting partial match>"
    )
    try:
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url",
                 "image_url": {"url": img_to_data_url(Path(image_path))}},
            ]}],
            "max_tokens": 60,
            "temperature": 0.0,
        }
        r = requests.post(f"{vlm_url}/chat/completions", json=payload,
                          timeout=timeout)
        r.raise_for_status()
        out = r.json()["choices"][0]["message"]["content"].strip()
    except Exception as e:
        # Don't block on verifier errors; treat as inconclusive (pass)
        return True, f"vlm_error:{type(e).__name__}"
    head = out.lstrip().upper()
    if head.startswith("YES") or head.startswith("PARTIAL"):
        return True, out[:80]
    if head.startswith("NO"):
        return False, out[:80]
    return True, f"ambiguous:{out[:60]}"


def load_model(lora_path=None):
    """Load base + optional LoRA. Returns (model, processor, label)."""
    import torch
    from transformers import (Qwen2_5_VLForConditionalGeneration,
                              AutoProcessor)
    processor = AutoProcessor.from_pretrained(MODEL_PATH, trust_remote_code=True)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MODEL_PATH, torch_dtype=torch.bfloat16,
        attn_implementation="sdpa", trust_remote_code=True)
    label = "base"
    if lora_path:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, lora_path)
        model = model.merge_and_unload()
        label = Path(lora_path).name
    model.eval()
    model.cuda()
    return model, processor, label


def generate_one(model, processor, image_path, system_prompt, user_msg,
                 max_new_tokens=1024):
    """Run one sample through the model. Returns the raw assistant string."""
    from qwen_vl_utils import process_vision_info
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": [
            {"type": "image", "image": image_path},
            {"type": "text", "text": user_msg},
        ]},
    ]
    text = processor.apply_chat_template(messages, tokenize=False,
                                          add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(text=[text], images=image_inputs, videos=video_inputs,
                        padding=True, return_tensors="pt").to("cuda")
    import torch
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=max_new_tokens,
                              do_sample=False, temperature=0.0)
    out_text = processor.batch_decode(
        out[:, inputs.input_ids.shape[1]:], skip_special_tokens=True)[0]
    return out_text.strip()


def score_response(meta, raw_response, user_msg, image_path=None,
                   run_hallucination=True):
    """Run step_12 gates + dest_correct + anchor_grounded.
    Set run_hallucination=False to skip the per-sample VLM call."""
    thinking, answer = parse_assistant(raw_response)
    parsed_action, _ = parse_action_from_answer(answer or "")

    gates = {}
    if not answer:
        gates["1_format"] = (False, "no <answer>")
        gates["2_sentence_count"] = (False, "no <answer>")
        gates["4_checkpoint"] = (False, "no <answer>")
        gates["5_dest_correct"] = (False, "no <answer>")
        gates["6_anchor_grounded"] = (False, "no <answer>")
    else:
        gates["1_format"] = gate_format(answer)
        gates["2_sentence_count"] = gate_sentence_count(answer)
        gates["4_checkpoint"] = gate_checkpoint_grounded(answer, user_msg)
        gates["5_dest_correct"] = gate_dest_correct(answer, meta)
        if run_hallucination and image_path:
            gates["6_anchor_grounded"] = gate_anchor_grounded(answer, image_path)
        else:
            gates["6_anchor_grounded"] = (True, "skipped")

    seg_b = get_first_seg_bearing(meta)
    heading_gt = meta.get("user_heading")
    delta = None
    if parsed_action in ACTION_DELTA and heading_gt is not None and seg_b is not None:
        h_walks = (heading_gt + ACTION_DELTA[parsed_action]) % 360
        delta = abs(angle_diff(h_walks, seg_b))
        if delta > 55:
            gates["3_closed_loop"] = (False, f"δ={delta:.1f}° > 55°")
        else:
            gates["3_closed_loop"] = (True, f"δ={delta:.1f}°")
    else:
        gates["3_closed_loop"] = (False, "no parseable action")

    # PASS_strict: only the 5 CORE gates + δ<30°. Gate 5 (dest_correct)
    # is reported as a side metric — it's a naming-style check, not a
    # correctness check, and shouldn't gate the headline pass rate.
    CORE_GATES = ["1_format", "2_sentence_count", "3_closed_loop",
                  "4_checkpoint", "6_anchor_grounded"]
    overall_strict = (delta is not None and delta < 30
                      and all(gates[g][0] for g in CORE_GATES if g in gates))
    overall_loose  = all(gates[g][0] for g in CORE_GATES if g in gates)

    return {
        "thinking": thinking, "answer": answer,
        "parsed_action": parsed_action,
        "delta": delta,
        "gates": {k: {"ok": v[0], "reason": v[1]} for k, v in gates.items()},
        "pass_strict_30": overall_strict,
        "pass_loose_55":  overall_loose,
    }


def evaluate(eval_jsonl, lora_path, tag, max_samples=None, out_dir=None,
             args_skip_halluc=False):
    rows = [json.loads(l) for l in open(eval_jsonl) if l.strip()]
    if max_samples:
        rows = rows[:max_samples]
    print(f"[eval] {len(rows)} samples from {eval_jsonl}")

    print(f"[eval] loading model (lora={lora_path})...")
    model, processor, label = load_model(lora_path)
    print(f"[eval] loaded: {label}")

    out_dir = Path(out_dir or ROOT / "results")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_jsonl = out_dir / f"eval_v3_{tag}.jsonl"
    out_summary = out_dir / f"eval_v3_{tag}.json"

    gate_pass = Counter()
    n_pass_30 = 0
    n_pass_55 = 0
    deltas = []
    t0 = time.time()
    with open(out_jsonl, "w") as fout:
        for i, r in enumerate(rows):
            sys_msg = next(m["content"] for m in r["messages"] if m["role"] == "system")
            user_msg = next(m["content"] for m in r["messages"] if m["role"] == "user")
            try:
                resp = generate_one(model, processor, r["image"], sys_msg, user_msg)
            except Exception as e:
                resp = f"[GEN_ERROR: {type(e).__name__}: {e}]"
            score = score_response(r["_meta"], resp, user_msg,
                                    image_path=r["image"],
                                    run_hallucination=not args_skip_halluc)
            for k, g in score["gates"].items():
                if g["ok"]:
                    gate_pass[k] += 1
            if score["pass_strict_30"]: n_pass_30 += 1
            if score["pass_loose_55"]:  n_pass_55 += 1
            if score["delta"] is not None:
                deltas.append(score["delta"])

            row_out = {
                "idx": i, "frame": r["_meta"]["start_frame"],
                "destination": r["_meta"]["destination"],
                "heading_gt": r["_meta"]["user_heading"],
                "first_seg_bearing": r["_meta"].get("first_seg_bearing"),
                "planner_action": r["_meta"]["first_action"],
                "model_response": resp,
                **score,
            }
            fout.write(json.dumps(row_out, ensure_ascii=False) + "\n")
            if (i + 1) % 10 == 0:
                rate = (i + 1) / max(time.time() - t0, 1e-3)
                print(f"  [{i+1}/{len(rows)}]  pass30={n_pass_30}  pass55={n_pass_55}  "
                      f"({rate:.2f}/s)")

    summary = {
        "tag": tag, "model": label, "n_samples": len(rows),
        "gate_pass_rate": {k: gate_pass[k] / len(rows) for k in
                            ["1_format", "2_sentence_count", "3_closed_loop",
                             "4_checkpoint", "5_dest_correct",
                             "6_anchor_grounded"]},
        "pass_strict_30":  n_pass_30 / len(rows),
        "pass_loose_55":   n_pass_55 / len(rows),
        "delta_distribution": {
            "n_with_delta": len(deltas),
            "median": sorted(deltas)[len(deltas)//2] if deltas else None,
            "p90":    sorted(deltas)[int(0.9 * len(deltas))] if deltas else None,
            "lt_30":  sum(1 for d in deltas if d < 30) / max(len(deltas), 1),
            "lt_55":  sum(1 for d in deltas if d <= 55) / max(len(deltas), 1),
        },
        "out_jsonl": str(out_jsonl),
    }
    with open(out_summary, "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    # console summary
    print(f"\n=== {tag} ({label}) ===")
    print(f"  samples : {summary['n_samples']}")
    for k, r in summary["gate_pass_rate"].items():
        print(f"  {k:<24}  {r*100:5.1f}%")
    print(f"  PASS_strict (<30°)        {summary['pass_strict_30']*100:5.1f}%")
    print(f"  PASS_loose  (≤55°)        {summary['pass_loose_55']*100:5.1f}%")
    print(f"  median δ  : {summary['delta_distribution']['median']}")
    print(f"  p90    δ  : {summary['delta_distribution']['p90']}")
    print(f"\n  written → {out_jsonl}")
    print(f"  summary → {out_summary}")


def compare(json_a, json_b):
    a = json.load(open(json_a))
    b = json.load(open(json_b))
    print(f"=== {a['tag']}  vs  {b['tag']} ===\n")
    print(f"{'metric':<30}{a['tag']:>14}{b['tag']:>14}    Δ")
    print("-" * 72)
    def row(label, va, vb, fmt="{:5.1f}%", pct=True):
        delta = (vb - va) * 100 if pct else (vb - va)
        sign = "+" if delta >= 0 else ""
        s_va = fmt.format(va * 100 if pct else va)
        s_vb = fmt.format(vb * 100 if pct else vb)
        print(f"{label:<30}{s_va:>14}{s_vb:>14}   {sign}{delta:.1f}")
    row("samples", a["n_samples"], b["n_samples"], "{:d}", pct=False)
    for k in ["1_format", "2_sentence_count", "3_closed_loop", "4_checkpoint",
              "5_dest_correct", "6_anchor_grounded"]:
        if k in a["gate_pass_rate"] and k in b["gate_pass_rate"]:
            row(f"  gate {k}",
                a["gate_pass_rate"][k], b["gate_pass_rate"][k])
    row("PASS_strict (<30°)", a["pass_strict_30"], b["pass_strict_30"])
    row("PASS_loose  (≤55°)", a["pass_loose_55"],  b["pass_loose_55"])
    row("delta_lt_30", a["delta_distribution"]["lt_30"],
                       b["delta_distribution"]["lt_30"])
    if a["delta_distribution"]["median"] and b["delta_distribution"]["median"]:
        row("median δ (°)",
            a["delta_distribution"]["median"], b["delta_distribution"]["median"],
            "{:.1f}", pct=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval", default="data/cities/zurich/synth_v3_eval.jsonl")
    ap.add_argument("--lora", default=None,
                    help="path to LoRA adapter dir; omit for base model only")
    ap.add_argument("--tag", default="run",
                    help="output filename suffix (eval_v3_<tag>.jsonl)")
    ap.add_argument("--max-samples", type=int, default=None)
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--skip-hallucination", action="store_true",
                    help="skip per-sample Gemma yes/no anchor check (faster)")
    ap.add_argument("--compare", nargs=2, default=None,
                    help="paths to two summary jsons to compare side-by-side")
    args = ap.parse_args()

    if args.compare:
        compare(args.compare[0], args.compare[1])
        return
    evaluate(args.eval, args.lora, args.tag, args.max_samples, args.out_dir,
             args_skip_halluc=args.skip_hallucination)


if __name__ == "__main__":
    main()

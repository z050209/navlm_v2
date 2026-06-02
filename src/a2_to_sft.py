"""Convert annotations_a2_{given,derived,implicit}.jsonl into
Qwen2.5-VL chat-template format for `src/a2_train_modal.py`.

This script processes ALL THREE variants in one run, splitting on a
single shared `(video, frame_id, destination)` key list — so the three
variants' train / val / test splits contain IDENTICAL instances, just
with each variant's own student prompt and teacher response.

That alignment lets you compare the 6 conditions head-to-head per row
(zs-given vs trained-given vs zs-derived vs trained-derived, etc.) on
the same test instances.

Input:
  data/cities/zurich/a2/annotations_a2_given.jsonl
  data/cities/zurich/a2/annotations_a2_derived.jsonl
  data/cities/zurich/a2/annotations_a2_implicit.jsonl

Output (6 files total):
  data/sft/a2_given_train.jsonl     a2_given_val.jsonl     a2_given_test.jsonl
  data/sft/a2_derived_train.jsonl   a2_derived_val.jsonl   a2_derived_test.jsonl
  data/sft/a2_implicit_train.jsonl  a2_implicit_val.jsonl  a2_implicit_test.jsonl

Per-row output shape (same as before):
  {
    "image_rel": "<video>/<frame_id>.jpg",
    "messages": [
      {"role": "system",    "content": [{"type":"text", "text": <variant-specific SYSTEM>}]},
      {"role": "user",      "content": [{"type":"image"},
                                         {"type":"text", "text": <variant-specific student_prompt>}]},
      {"role": "assistant", "content": [{"type":"text", "text": <variant-specific teacher_response>}]}
    ],
    "video": ..., "frame_id": ..., "destination": ...,
    "gt_verb": ..., "first_verb": ..., "heading": ...
  }

Key-alignment rule:
  - Build the key set = (video, frame_id, destination) triples present
    in ALL THREE annotation files (intersection).
  - Filter to format_pass==True in ALL THREE variants. A row only enters
    the cohort if every variant labelled it cleanly.
  - With --only-pass: additionally require direction_pass==True in ALL
    THREE (cleanest but smallest).
  - Shuffle the survivor keys with seed=42 and split 80/10/10.
  - Apply that single split to each variant's row list.

  python -m src.a2_to_sft
  python -m src.a2_to_sft --only-pass
  python -m src.a2_to_sft --seed 7 --val-frac 0.10 --test-frac 0.10
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config                                       # noqa: E402
from src.a2_annotate import system_prompt          # noqa: E402

VARIANTS = ("given", "derived", "implicit")


def _key(r):
    return (r["video"], r["frame_id"], r["destination"])


def _load_variant(in_dir, variant, only_pass):
    p = in_dir / f"annotations_a2_{variant}.jsonl"
    if not p.exists():
        sys.exit(f"input not found: {p}")
    rows = [json.loads(l) for l in p.open(encoding="utf-8") if l.strip()]
    n0 = len(rows)
    rows = [r for r in rows if r.get("format_pass")]
    n1 = len(rows)
    if only_pass:
        rows = [r for r in rows if r.get("direction_pass")]
    n2 = len(rows)
    print(f"[to_sft] {variant:<9s}: loaded {n0:,} → fmt {n1:,}"
          f"{(' → dir ' + format(n2, ',')) if only_pass else ''}")
    return {_key(r): r for r in rows}


def _to_qwen_row(r, sys_prompt_text):
    return {
        "image_rel": f"{r['video']}/{r['frame_id']}.jpg",
        "messages": [
            {"role": "system",
             "content": [{"type": "text", "text": sys_prompt_text}]},
            {"role": "user",
             "content": [
                 {"type": "image"},
                 {"type": "text", "text": r["student_prompt"]},
             ]},
            {"role": "assistant",
             "content": [{"type": "text", "text": r["response"]}]},
        ],
        "video": r["video"], "frame_id": r["frame_id"],
        "destination": r["destination"],
        "destination_zh": r.get("destination_zh", ""),
        "gt_verb": r["gt_verb"], "first_verb": r["first_verb"],
        "heading": r.get("heading"),
        "direction_pass": r.get("direction_pass"),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("--in-dir", default=str(config.CITY_DIR / "a2"),
                    help="dir containing annotations_a2_{variant}.jsonl")
    ap.add_argument("--out-dir", default="data/sft",
                    help="where to write the 9 split files")
    ap.add_argument("--val-frac", type=float, default=0.10)
    ap.add_argument("--test-frac", type=float, default=0.10)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--only-pass", action="store_true",
                    help="additionally require direction_pass==True in "
                         "all three variants (cleaner but smaller)")
    args = ap.parse_args()

    in_dir = Path(args.in_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── load all 3 variants, build per-variant key→row maps ───────────
    by_variant = {v: _load_variant(in_dir, v, args.only_pass)
                  for v in VARIANTS}

    # ── intersect keys: a row must pass filters in ALL 3 variants ─────
    common = set.intersection(*[set(by_variant[v].keys())
                                 for v in VARIANTS])
    print()
    print(f"[to_sft] per-variant survivor counts:")
    for v in VARIANTS:
        n = len(by_variant[v])
        only_here = len(set(by_variant[v]) - common)
        print(f"   {v:<9s}: {n:,} kept · {only_here:,} dropped "
              f"(not in all-3 intersection)")
    print(f"[to_sft] aligned cohort (in all 3): {len(common):,}")

    # ── single 80/10/10 split on the shared key list ──────────────────
    keys = sorted(common)                       # deterministic ordering
    rng = random.Random(args.seed)
    rng.shuffle(keys)
    n = len(keys)
    n_test = int(n * args.test_frac)
    n_val = int(n * args.val_frac)
    split_keys = {
        "test":  set(keys[:n_test]),
        "val":   set(keys[n_test:n_test + n_val]),
        "train": set(keys[n_test + n_val:]),
    }
    print()
    print(f"[to_sft] split (seed={args.seed}, "
          f"val={args.val_frac}, test={args.test_frac}):")
    print(f"   train: {len(split_keys['train']):,}")
    print(f"   val:   {len(split_keys['val']):,}")
    print(f"   test:  {len(split_keys['test']):,}")

    # ── write 9 files (3 variants × 3 splits) ─────────────────────────
    print()
    for v in VARIANTS:
        sys_prompt_text = system_prompt(v)
        rows_map = by_variant[v]
        for split_name in ("train", "val", "test"):
            out = out_dir / f"a2_{v}_{split_name}.jsonl"
            with out.open("w", encoding="utf-8") as f:
                for k in sorted(split_keys[split_name]):
                    f.write(json.dumps(_to_qwen_row(rows_map[k],
                                                      sys_prompt_text),
                                        ensure_ascii=False) + "\n")
            print(f"   wrote {out}  ({len(split_keys[split_name])} rows)")

    # ── write the shared split manifest (auditing aid) ────────────────
    manifest = {
        "seed": args.seed,
        "only_pass": args.only_pass,
        "val_frac": args.val_frac,
        "test_frac": args.test_frac,
        "n_aligned": len(common),
        "n_train": len(split_keys["train"]),
        "n_val": len(split_keys["val"]),
        "n_test": len(split_keys["test"]),
        "test_keys":
            [list(k) for k in sorted(split_keys["test"])],
    }
    (out_dir / "a2_split_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False))
    print(f"\n[to_sft] wrote {out_dir / 'a2_split_manifest.json'}")
    print(f"[to_sft] ALL 3 VARIANTS use the SAME (video, frame_id, "
          f"destination) keys per split — apples-to-apples cross-condition "
          f"comparison.")


if __name__ == "__main__":
    main()

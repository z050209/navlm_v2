"""Pull 2 success + 2 failure example rows per headline condition.

Headline conditions:
  - zs-heading-given / trained-heading-given_r16_e5
  - zs-heading-derived / trained-heading-derived_r16_e5
  - zs-heading-implicit / trained-heading-implicit_r16_e5

For each, prints the row's video, frame_id, destination, GT verb, model's
first_verb, and the full <thinking> + <answer> body. Writes to
docs/qualitative_examples.md (markdown for inclusion in the report).
"""
import argparse
import io
import json
import sys
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

HEADLINE_CONDITIONS = [
    "zs-heading-given",
    "trained-heading-given_r16_e5",
    "zs-heading-derived",
    "trained-heading-derived_r16_e5",
    "zs-heading-implicit",
    "trained-heading-implicit_r16_e5",
]


def pick_examples(rows, n_each=2):
    """Pick n_each PASS + n_each FAIL rows, varied across gt_verbs."""
    passes = [r for r in rows if r.get("PASS")]
    fails = [r for r in rows if not r.get("PASS")]
    # diversify by GT verb
    def diversify(pool):
        seen, out = set(), []
        for r in pool:
            v = r.get("gt_verb")
            if v not in seen:
                out.append(r); seen.add(v)
            if len(out) >= n_each:
                break
        # If not enough variety, just take first
        while len(out) < n_each and len(pool) > len(out):
            for r in pool:
                if r not in out:
                    out.append(r); break
        return out[:n_each]
    return diversify(passes), diversify(fails)


def fmt_row(r):
    resp = r.get("model_response", "")
    t1 = resp.find("<thinking>"); t2 = resp.find("</thinking>", t1)
    thinking = ""
    if t1 >= 0:
        thinking = resp[t1+10 : t2 if t2 > t1 else None].strip()
    a1 = resp.find("<answer>")
    answer = ""
    if a1 >= 0:
        a2 = resp.find("</answer>", a1)
        answer = resp[a1+8 : a2 if a2 > a1 else None].strip()
    return (
        f"- **frame**: `{r['video']}/{r['frame_id']}.jpg`\n"
        f"- **destination**: {r['destination']}  "
        f"(GT heading={r.get('heading', 0):.0f}°)\n"
        f"- **GT verb**: `{r.get('gt_verb')}` · "
        f"**model verb**: `{r.get('first_verb') or '(none)'}` · "
        f"**PASS**: {'✓' if r.get('PASS') else '✗'}\n"
        f"- **<thinking>** (truncated to 400 chars):\n"
        f"  ```\n  {thinking[:400]}\n  ```\n"
        f"- **<answer>**: `{answer[:200]}`\n"
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir",
                    default="eval_pull/ablation_20260602_054707")
    ap.add_argument("--out",
                    default="docs/qualitative_examples.md")
    args = ap.parse_args()

    run = Path(args.run_dir)
    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)

    buf = []
    buf.append("# Qualitative success/failure examples\n")
    buf.append(f"Source: `{args.run_dir}/` (2026-06-02/-03 ablation)\n\n")
    buf.append(f"For each headline condition, 2 success rows (`PASS=True`) + "
                f"2 failure rows (`PASS=False`). Examples diversified across "
                f"GT verbs where possible.\n\n")

    for cond in HEADLINE_CONDITIONS:
        d = run / cond / "per_sample_scored.jsonl"
        if not d.exists():
            buf.append(f"## `{cond}`\n\n*(missing scored file)*\n\n")
            continue
        rows = [json.loads(l) for l in d.open(encoding="utf-8")
                if l.strip()]
        if not rows:
            buf.append(f"## `{cond}`\n\n*(empty)*\n\n")
            continue
        passes, fails = pick_examples(rows, n_each=2)

        buf.append(f"## `{cond}`  (n={len(rows)}, "
                    f"PASS={sum(1 for r in rows if r.get('PASS'))/len(rows)*100:.1f}%)\n\n")

        buf.append("### Successes (model verb == GT verb)\n\n")
        for i, r in enumerate(passes, 1):
            buf.append(f"#### Success #{i}\n\n{fmt_row(r)}\n")

        buf.append("### Failures (model verb != GT verb)\n\n")
        for i, r in enumerate(fails, 1):
            buf.append(f"#### Failure #{i}\n\n{fmt_row(r)}\n")

        buf.append("---\n\n")

    out.write_text("".join(buf), encoding="utf-8")
    print(f"wrote {out}  ({sum(len(s) for s in buf):,} chars)")


if __name__ == "__main__":
    main()

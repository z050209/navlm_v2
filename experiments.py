"""Slide 4/5 orchestrator — runs all 6 conditions × 2 ablations.

This is the LOCAL entrypoint that drives the full milestone matrix on
Modal:

  • For each L-* condition (L-given, L-implicit, L-explicit):
      `modal run train_modal.py --variant <v>`        (one A100, ~3-6 h)
  • For each (condition, ablation) pair (12 total):
      `modal run eval_modal.py --condition <c> --ablation <a>`
  • Pull all results back: `python pull_eval.py <run_id>`

The B-* conditions skip training (zero-shot baselines on base Qwen);
the L-* conditions depend on the matching adapter being on the
`navlm-ckpts` volume before eval starts. Order: train_given,
train_implicit, train_explicit → then all 12 evals.

Modes:
  --mode train        train the 3 LoRA adapters only
  --mode eval         run all 12 evals (assumes adapters exist)
  --mode all          (default) train then eval
  --mode smoke        --limit 5 across the board (~$2, ~20 min)

Examples:
  python experiments.py --mode smoke
  python experiments.py --mode all
  python experiments.py --mode eval --conditions L-explicit --ablations video
"""

import argparse
import datetime
import subprocess
import sys


ALL_CONDS = ["B-given", "B-implicit", "B-explicit",
             "L-given", "L-implicit", "L-explicit"]
ALL_ABLS = ["video", "poi"]
SFT_VARIANTS = ["given", "implicit", "explicit"]


def _run(cmd, **kw):
    """Run `cmd` (list) — stream output, raise on non-zero exit."""
    print(f"\n$ {' '.join(cmd)}\n", flush=True)
    r = subprocess.run(cmd, **kw)
    if r.returncode != 0:
        sys.exit(f"[experiments] command failed (exit {r.returncode}): "
                 f"{' '.join(cmd)}")
    return r


def train_all(epochs: int, lr: float, limit: int):
    """Train the three LoRA adapters in sequence (serial — Modal can
    queue them; one A100 each)."""
    for v in SFT_VARIANTS:
        cmd = ["modal", "run", "train_modal.py",
               "--variant", v, "--epochs", str(epochs), "--lr", str(lr)]
        if limit:
            cmd += ["--limit", str(limit)]
        _run(cmd)


def eval_all(conditions, ablations, limit, no_anchor, run_id):
    """Run the requested (condition × ablation) eval cells."""
    for a in ablations:
        # Use `--condition all` when the full 6 are requested (one
        # local call, six remote functions — same total cost but
        # tidier output table).
        if set(conditions) == set(ALL_CONDS):
            cmd = ["modal", "run", "eval_modal.py",
                   "--condition", "all", "--ablation", a,
                   "--run-id", run_id]
            if limit:
                cmd += ["--limit", str(limit)]
            if no_anchor:
                cmd += ["--no-anchor"]
            _run(cmd)
        else:
            for c in conditions:
                cmd = ["modal", "run", "eval_modal.py",
                       "--condition", c, "--ablation", a,
                       "--run-id", run_id]
                if limit:
                    cmd += ["--limit", str(limit)]
                if no_anchor:
                    cmd += ["--no-anchor"]
                _run(cmd)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("--mode", choices=["train", "eval", "all", "smoke"],
                    default="all")
    ap.add_argument("--conditions", default=",".join(ALL_CONDS),
                    help=f"comma-separated (default: all 6)")
    ap.add_argument("--ablations", default=",".join(ALL_ABLS),
                    help="comma-separated (default: both)")
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--limit", type=int, default=0,
                    help="cap rows (per stage) — for smoke")
    ap.add_argument("--no-anchor", action="store_true",
                    help="skip the Gemini anchor-faithfulness check")
    ap.add_argument("--run-id", default=None)
    args = ap.parse_args()

    run_id = args.run_id or datetime.datetime.now().strftime(
        "%Y%m%d_%H%M%S")
    conds = [c.strip() for c in args.conditions.split(",") if c.strip()]
    abls = [a.strip() for a in args.ablations.split(",") if a.strip()]
    assert all(c in ALL_CONDS for c in conds), conds
    assert all(a in ALL_ABLS for a in abls), abls

    mode = args.mode
    if mode == "smoke":
        mode = "all"
        args.limit = args.limit or 5
        print(f"[experiments] SMOKE — limit={args.limit}, run_id={run_id}")

    print(f"[experiments] mode={mode}  conds={conds}  ablations={abls}  "
          f"limit={args.limit}  run_id={run_id}")

    if mode in ("train", "all"):
        # Only train the variants we actually need for L-* in conds
        needed = sorted({c.split("-", 1)[1] for c in conds
                         if c.startswith("L-")})
        for v in needed:
            cmd = ["modal", "run", "train_modal.py",
                   "--variant", v, "--epochs", str(args.epochs),
                   "--lr", str(args.lr)]
            if args.limit:
                cmd += ["--limit", str(args.limit)]
            _run(cmd)

    if mode in ("eval", "all"):
        eval_all(conds, abls, args.limit, args.no_anchor, run_id)

    print(f"\n=== ALL DONE — run_id={run_id} ===")
    print(f"Pull results:  python pull_eval.py {run_id}")


if __name__ == "__main__":
    main()

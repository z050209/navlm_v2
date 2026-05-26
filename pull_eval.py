"""Pull a Modal eval run's outputs back to the local disk.

  python pull_eval.py 20260601_223110           # one run
  python pull_eval.py 20260601_223110 --dest ./my_runs

Underneath: `modal volume get navlm-eval /<run_id> ./eval_results/<run_id>`,
followed by an aggregation pass that prints the slide-4 matrix:

  condition    × ablation video    × ablation poi
  B-given                  PASS=...                PASS=...
  B-implicit               PASS=...                PASS=...
  ...

If a (condition, ablation) cell is missing, it shows "—".
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

ALL_CONDS = ["B-given", "B-implicit", "B-explicit",
             "L-given", "L-implicit", "L-explicit"]
ALL_ABLS = ["video", "poi"]


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("run_id", help="run_id printed by experiments.py")
    ap.add_argument("--dest", default="./eval_results",
                    help="local destination root (default eval_results/)")
    ap.add_argument("--no-download", action="store_true",
                    help="skip modal volume get, aggregate only")
    args = ap.parse_args()

    dest_root = Path(args.dest)
    dest = dest_root / args.run_id
    dest.parent.mkdir(parents=True, exist_ok=True)

    if not args.no_download:
        cmd = ["modal", "volume", "get", "navlm-eval",
               f"/{args.run_id}", str(dest_root)]
        print("$", " ".join(cmd))
        r = subprocess.run(cmd)
        if r.returncode != 0:
            sys.exit(f"[pull_eval] modal volume get failed "
                     f"({r.returncode})")

    # ── aggregate the per-cell summary.json files ────────────────
    summaries = {}
    for cell_dir in sorted(dest.glob("*__*")):
        s_path = cell_dir / "summary.json"
        if not s_path.exists():
            continue
        s = json.loads(s_path.read_text(encoding="utf-8"))
        summaries[(s["condition"], s["ablation"])] = s

    if not summaries:
        sys.exit(f"[pull_eval] no summaries under {dest}")

    # write the 6×2 matrix
    matrix_path = dest / "_matrix.json"
    matrix = {c: {a: summaries.get((c, a)) for a in ALL_ABLS}
              for c in ALL_CONDS}
    matrix_path.write_text(json.dumps(matrix, indent=2))
    print(f"[pull_eval] wrote {matrix_path}")
    print()

    # print the headline table
    print(f"=== run {args.run_id} — PASS_strict matrix ===\n")
    print(f"{'condition':<12}  {'N (vid|poi)':>13}  "
          f"{'video PASS':>11}  {'poi PASS':>9}  {'video dir':>9}  "
          f"{'poi dir':>9}")
    for c in ALL_CONDS:
        v = summaries.get((c, "video"))
        p = summaries.get((c, "poi"))
        nv = v["n"] if v else 0
        np_ = p["n"] if p else 0
        vp = f"{v['pass_strict_rate']:.2%}" if v else "—"
        pp = f"{p['pass_strict_rate']:.2%}" if p else "—"
        vd = f"{v['directional_rate']:.2%}" if v else "—"
        pd = f"{p['directional_rate']:.2%}" if p else "—"
        print(f"{c:<12}  {nv:>5}|{np_:<5}  {vp:>11}  {pp:>9}  "
              f"{vd:>9}  {pd:>9}")

    print()
    print(f"Per-sample jsonl files live under: {dest}")


if __name__ == "__main__":
    main()

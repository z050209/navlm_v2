"""Print a summary of the unified pipeline output.

Reads `data/cities/zurich/frame_starts_trusted_all.jsonl` and breaks down
by video × gps_source × verdict. Used after run_all completes.
"""

import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pipeline.config import DATA, VIDEO_NAMES  # noqa: E402


def main():
    p = DATA / "frame_starts_trusted_all.jsonl"
    if not p.exists():
        sys.exit(f"missing {p}")

    rows = [json.loads(ln) for ln in open(p)]
    print(f"=== unified trusted_starts: {len(rows)} frames ===\n")

    by_video = Counter()
    by_source = Counter()
    by_verdict = Counter()
    by_video_source = {}
    for r in rows:
        v = r.get("video", "?")
        s = r.get("gps_source", "?")
        vd = r.get("evidence", {}).get("verdict", "?")
        by_video[v] += 1
        by_source[s] += 1
        by_verdict[vd] += 1
        by_video_source.setdefault(v, Counter())[s] += 1

    print(f"{'video':<22} {'total':>6}  {'ocr':>4} {'v_high':>7} {'v_med':>6} {'v_low':>6}")
    print("-" * 60)
    for v in VIDEO_NAMES:
        c = by_video_source.get(v, Counter())
        total = by_video[v]
        print(f"{v:<22} {total:>6}  {c.get('ocr', 0):>4} "
              f"{c.get('visual_high', 0):>7} {c.get('visual_medium', 0):>6} "
              f"{c.get('visual_low', 0):>6}")

    print(f"\nby gps_source:  {dict(by_source.most_common())}")
    print(f"by verdict:     {dict(by_verdict.most_common())}")

    # Compare against legacy
    legacy_orig = DATA / "frame_starts_trusted.jsonl"
    legacy_extra = DATA / "frame_starts_trusted_extra.jsonl"
    n_legacy = 0
    for f in (legacy_orig, legacy_extra):
        if f.exists():
            n_legacy += sum(1 for _ in open(f))
    print(f"\ncomparison:")
    print(f"  legacy trusted_starts (orig+extra): {n_legacy}")
    print(f"  new pipeline trusted_starts_all:    {len(rows)}")


if __name__ == "__main__":
    main()

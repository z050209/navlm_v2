"""Count how often the VLM mentioned each of the 21 attractions across
its scan output.

Works on either VLM scan file (or both merged):
  poi_scan.jsonl             — the every-10 baseline (872 rows)
  poi_scan_cos0.75.jsonl     — the cos>=0.75 expansion (4,101 rows)

For each of the 21 attractions, reports:
  - n_visible  : frames with the attraction in `visible[]`
  - n_guess    : frames where the attraction IS the `guess`
  - n_either   : frames where it appears in either field

Aliases are applied so "Kirche St. Peter", "St. Peter's Church",
"St. Peter Church" all count as St. Peter; "Zürichsee" counts as Lake
Zurich; bare "Limmat" counts as Limmat river; etc.

Output to stdout only — this is a coverage diagnostic, not a data
artifact.

  python -m src.a2_vlm_coverage                  # both files merged
  python -m src.a2_vlm_coverage --input poi_scan.jsonl
  python -m src.a2_vlm_coverage --input poi_scan_cos0.75.jsonl
"""

from __future__ import annotations

import argparse
import collections
import json
import re
import sys
import unicodedata
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config                                       # noqa: E402
from src.a2_attraction_slots import (               # noqa: E402
    ATTRACTIONS_21, ALIASES,
)


def fold(s: str) -> str:
    s = unicodedata.normalize("NFKD", str(s or "").lower())
    s = "".join(c for c in s if not unicodedata.combining(c))
    s = re.sub(r"[^a-z0-9 ]+", " ", s)
    return re.sub(r"\s+", " ", s).strip()


NAME_TO_CANON = {}
for en, *_ in ATTRACTIONS_21:
    NAME_TO_CANON[fold(en)] = en
    for a in ALIASES.get(en, set()):
        NAME_TO_CANON[fold(a)] = en


def _flat(vis):
    out = []
    for v in (vis or []):
        if isinstance(v, list):
            out.extend(str(x) for x in v)
        else:
            out.append(str(v))
    return out


def _canon_hits(strings):
    hits = set()
    for s in strings:
        for p in re.split(r"\s*(?:/|,|\||\bam\b|\bat\b|\bnear\b)\s*",
                           s or ""):
            f = fold(p)
            if f and f in NAME_TO_CANON:
                hits.add(NAME_TO_CANON[f])
    return hits


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("--input", action="append", default=None,
                    help="poi_scan file(s) to count — repeat for multi. "
                         "Default: poi_scan.jsonl + poi_scan_cos0.75.jsonl "
                         "(both merged, dedup by (video, frame_id))")
    args = ap.parse_args()

    sources = args.input or [
        str(config.CITY_DIR / "poi_scan.jsonl"),
        str(config.CITY_DIR / "poi_scan_cos0.75.jsonl"),
    ]
    rows_by_key = {}
    src_counts = []
    for sf in sources:
        if not Path(sf).exists():
            print(f"WARN: {sf} not found, skipping")
            continue
        n = 0
        for line in Path(sf).open(encoding="utf-8"):
            if not line.strip():
                continue
            d = json.loads(line)
            rows_by_key[(d["video"], d["frame_id"])] = d
            n += 1
        src_counts.append((Path(sf).name, n))

    for name, n in src_counts:
        print(f"  {name}: {n:,} rows")
    print(f"unique VLM-scanned frames (deduped):  {len(rows_by_key):,}")

    n_visible = collections.Counter()
    n_guess = collections.Counter()
    n_either = collections.Counter()
    n_with_any = 0
    for r in rows_by_key.values():
        vis = _flat(r.get("visible"))
        guess = (r.get("guess") or "").strip()
        in_visible = _canon_hits(vis)
        in_guess = _canon_hits([guess])
        if in_visible or in_guess:
            n_with_any += 1
        for en in in_visible:
            n_visible[en] += 1
        for en in in_guess:
            n_guess[en] += 1
        for en in in_visible | in_guess:
            n_either[en] += 1

    print()
    print("=" * 96)
    print("PER-ATTRACTION — how often the VLM named each (after aliases)")
    print("=" * 96)
    print(f"\n{'#':>2}  {'attraction':<22s} {'中文':<14s} "
          f"{'kind':<8s} {'visible':>8s} {'guess':>8s} {'either':>8s}")
    print("-" * 96)
    for i, (en, zh, _lat, _lon, kind) in enumerate(ATTRACTIONS_21, 1):
        print(f"{i:>2}  {en:<22s} {zh:<14s} {kind:<8s} "
              f"{n_visible[en]:>8d} {n_guess[en]:>8d} "
              f"{n_either[en]:>8d}")
    print()
    print(f"frames with >=1 attraction in EITHER field: "
          f"{n_with_any:,}  ({100*n_with_any/max(1,len(rows_by_key)):.1f} %)")


if __name__ == "__main__":
    main()

"""Dump the RAW name strings the VLM produced — no OSM resolution, no
folding, no aliases. So we can see exactly what tokens it returns and
whether the 27 canonical landmark names (`src/poi.py:CANDIDATE_POIS`)
are in that raw vocabulary at all.

Three sections:
  (1) every distinct `guess` string + count
  (2) every distinct `visible` string + count
  (3) per-candidate-landmark cross-tab: for each of the 27 canonical
      names, list the raw VLM strings that *fold to* that name, with
      counts in `guess` vs `visible`.

  python -m src.poi_scan_raw_names                # everything
  python -m src.poi_scan_raw_names --head 60      # truncate (1) and (2)
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
from src.poi import CANDIDATE_POIS                  # noqa: E402


def fold(s: str) -> str:
    """Lowercase + strip diacritics — same as src/pois.py:fold."""
    s = unicodedata.normalize("NFKD", str(s or "").lower())
    return "".join(c for c in s if not unicodedata.combining(c))


def normalise(s: str) -> str:
    """For substring containment — fold + collapse non-alnum to space."""
    s = fold(s)
    s = re.sub(r"[^a-z0-9 ]+", " ", s)
    return re.sub(r"\s+", " ", s).strip()


CANDIDATE_FOLDED = [(en, fold(en)) for en, _, _, _, _ in CANDIDATE_POIS]
CANDIDATE_NORM = [(en, normalise(en)) for en, _, _, _, _ in CANDIDATE_POIS]


def contains_candidate(s: str):
    """Return list of canonical EN names whose folded form is a substring
    of `s`'s normalised form. Order-preserving."""
    n = normalise(s)
    return [en for en, ndl in CANDIDATE_NORM if ndl and ndl in n]


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("--input",
                    default=str(config.CITY_DIR / "poi_scan.jsonl"))
    ap.add_argument("--head", type=int, default=0,
                    help="truncate the long all-strings dumps to top N "
                         "(0 = show all)")
    args = ap.parse_args()

    in_path = Path(args.input)
    if not in_path.exists():
        sys.exit(f"input not found: {in_path}")

    rows = [json.loads(l) for l in in_path.open(encoding="utf-8")
            if l.strip()]
    print(f"[poi_scan_raw] poi_scan.jsonl rows: {len(rows):,}")

    # ── collect raw strings ──────────────────────────────────────────
    guess_strings = collections.Counter()
    visible_strings = collections.Counter()
    for r in rows:
        g = (r.get("guess") or "").strip()
        if g:
            # `guess` is sometimes "German | English" — keep both halves
            for s in re.split(r"\s*\|\s*", g):
                if s:
                    guess_strings[s] += 1
        for v in (r.get("visible") or []):
            if isinstance(v, list):
                for s in v:
                    if str(s).strip():
                        visible_strings[str(s).strip()] += 1
            elif str(v).strip():
                visible_strings[str(v).strip()] += 1

    print(f"[poi_scan_raw] distinct `guess` strings:   "
          f"{len(guess_strings):,}")
    print(f"[poi_scan_raw] distinct `visible` strings: "
          f"{len(visible_strings):,}")

    # ── (1) every distinct guess ─────────────────────────────────────
    print()
    print("=" * 96)
    print(f"(1) ALL DISTINCT `guess` STRINGS  ({len(guess_strings)} unique"
          + (f", showing top {args.head}" if args.head else "")
          + ")")
    print("=" * 96)
    items = guess_strings.most_common(args.head or None)
    for s, c in items:
        hits = contains_candidate(s)
        marker = "★" if hits else " "
        hit_str = f"  [→ {', '.join(hits)}]" if hits else ""
        print(f"  {marker} {c:>5d}  {s}{hit_str}")

    # ── (2) every distinct visible ───────────────────────────────────
    print()
    print("=" * 96)
    print(f"(2) ALL DISTINCT `visible` STRINGS  ({len(visible_strings)} "
          f"unique"
          + (f", showing top {args.head}" if args.head else "")
          + ")")
    print("=" * 96)
    items = visible_strings.most_common(args.head or None)
    for s, c in items:
        hits = contains_candidate(s)
        marker = "★" if hits else " "
        hit_str = f"  [→ {', '.join(hits)}]" if hits else ""
        print(f"  {marker} {c:>5d}  {s}{hit_str}")

    # ── (3) per-candidate cross-tab ──────────────────────────────────
    # for each of the 27 canonical names, list every distinct raw VLM
    # string that contains it (folded), with counts.
    print()
    print("=" * 96)
    print("(3) FOR EACH OF THE 27 CANONICAL LANDMARKS — RAW VLM STRINGS "
          "THAT CONTAIN IT")
    print("=" * 96)
    for i, (en, zh, _lat, _lon, kind) in enumerate(CANDIDATE_POIS, 1):
        in_guess = {s: c for s, c in guess_strings.items()
                    if en in contains_candidate(s)}
        in_visible = {s: c for s, c in visible_strings.items()
                      if en in contains_candidate(s)}
        total_g = sum(in_guess.values())
        total_v = sum(in_visible.values())
        print(f"\n{i:>2}. {en} ({zh}) — kind={kind}")
        print(f"    guess   total = {total_g}  ·  "
              f"distinct = {len(in_guess)}")
        for s, c in sorted(in_guess.items(), key=lambda x: -x[1]):
            print(f"        {c:>4d}  {s}")
        print(f"    visible total = {total_v}  ·  "
              f"distinct = {len(in_visible)}")
        for s, c in sorted(in_visible.items(), key=lambda x: -x[1]):
            print(f"        {c:>4d}  {s}")


if __name__ == "__main__":
    main()

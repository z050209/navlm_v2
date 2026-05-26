"""Throwaway — run VLM (poi_scan-style) on tier-2 frames at cos>=0.75
to expand the tier-1 set. Output schema matches poi_scan.jsonl so the
file can be concatenated for the next gps_recovery run.

  python _vlm_test.py                 # smoke test: 3 random frames
  python _vlm_test.py --limit 50      # run on 50 random frames
  python _vlm_test.py --limit 4019    # full extrapolation

Append-mode + skip-already-done: safe to re-run with larger --limit.
"""

import argparse, collections, json, random, sys, time
from pathlib import Path

from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config
from src import poi_scan

ap = argparse.ArgumentParser()
ap.add_argument("--limit", type=int, default=3,
                help="frames to PROCESS (scan + copy). 0 = no cap.")
ap.add_argument("--seed", type=int, default=42,
                help="random seed for reproducibility")
ap.add_argument("--min-sim", type=float, default=0.75,
                help="cosine threshold for the tier-2 candidate pool")
ap.add_argument("--output", type=str,
                default="poi_scan_cos0.75.jsonl",
                help="output filename under data/cities/zurich/ "
                     "(append-mode; skips already-scanned frames)")
args = ap.parse_args()

# 1) load candidates — ALL accepted at cos>=min-sim (both tiers).
# Tier-1 frames are already in poi_scan.jsonl, so we'll *copy* them
# (free); only tier-2 frames need fresh Pro calls.
rows = [json.loads(l) for l in
        (config.CITY_DIR / "gps_recovery_all.jsonl").open(encoding="utf-8")
        if l.strip()]
candidates = [r for r in rows
              if r.get("accepted") and r["s_dino"] >= args.min_sim]
random.seed(args.seed)
random.shuffle(candidates)
t1 = sum(1 for c in candidates if c.get("tier") == 1)
t2 = sum(1 for c in candidates if c.get("tier") == 2)
print(f"candidates at cos>={args.min_sim}: {len(candidates):,} "
      f"({t1} tier-1 to COPY, {t2} tier-2 to SCAN)", flush=True)

# load existing poi_scan.jsonl rows (the tier-1 source) into a lookup
poi_scan_index = {}
poi_scan_path = config.CITY_DIR / "poi_scan.jsonl"
for line in poi_scan_path.open(encoding="utf-8"):
    if not line.strip():
        continue
    d = json.loads(line)
    poi_scan_index[(d["video"], d["frame_id"])] = d
print(f"original poi_scan.jsonl rows loaded: {len(poi_scan_index)}",
      flush=True)

# 2) skip frames already in the output (resume support)
out_path = config.CITY_DIR / args.output
done = set()
if out_path.exists():
    for line in out_path.open(encoding="utf-8"):
        if not line.strip():
            continue
        d = json.loads(line)
        done.add((d["video"], d["frame_id"]))
    print(f"already done in {args.output}: {len(done)}", flush=True)

# 3) take the next un-done candidates (capped by --limit if > 0)
todo = [c for c in candidates if (c["video"], c["frame_id"]) not in done]
if args.limit > 0:
    todo = todo[:args.limit]
todo_copy  = [c for c in todo
              if (c["video"], c["frame_id"]) in poi_scan_index]
todo_scan  = [c for c in todo
              if (c["video"], c["frame_id"]) not in poi_scan_index]
print(f"processing {len(todo)} frames: "
      f"{len(todo_copy)} copy + {len(todo_scan)} scan "
      f"(scan cost ~${len(todo_scan)*0.012:.2f}, "
      f"~{len(todo_scan)*11/3600:.1f} h)", flush=True)

# 4) load OSM POIs for resolve
pois_list = json.loads(
    (config.CITY_DIR / "pois.json").read_text(encoding="utf-8"))

# 5) run — copy tier-1 rows first (instant), then scan tier-2 rows.
t0 = time.time()
n_copied, n_scanned, n_errored = 0, 0, 0
with out_path.open("a", encoding="utf-8") as fout:
    # Phase A: copy from poi_scan.jsonl (free)
    for c in todo_copy:
        key = (c["video"], c["frame_id"])
        src = poi_scan_index[key]
        # write the original row verbatim
        fout.write(json.dumps(src, ensure_ascii=False) + "\n")
        n_copied += 1
    if n_copied:
        fout.flush()
        print(f"  [copy] {n_copied} tier-1 rows copied from "
              f"poi_scan.jsonl (free)", flush=True)
    # Phase B: scan tier-2 with Pro — tqdm progress bar with live ETA
    pbar = tqdm(todo_scan, desc="[Pro VLM]", unit="f",
                mininterval=2.0, dynamic_ncols=True)
    for frame in pbar:
        frame_path = (config.FRAMES_DIR / frame["video"] /
                      f"{frame['frame_id']}.jpg")
        try:
            parsed = poi_scan.scan_frame(frame_path)
        except Exception as e:
            n_errored += 1
            tqdm.write(f"  ERROR {frame['video']}/{frame['frame_id']}: "
                       f"{type(e).__name__}: {e}")
            continue
        matched, unmatched = poi_scan.match_scan(parsed, pois_list)
        row = {
            "video": frame["video"],
            "frame_id": frame["frame_id"],
            "guess": " | ".join(parsed["guess"]),
            "confidence": parsed["confidence"],
            "reasoning": parsed["reasoning"],
            "visible": parsed["visible"],
            "matched": matched,
            "unmatched": unmatched,
        }
        fout.write(json.dumps(row, ensure_ascii=False) + "\n")
        fout.flush()
        n_scanned += 1
        # postfix with running cost + last match
        guess = parsed["guess"][0][:14] if parsed["guess"] else "-"
        dino_n = (frame.get("dino_nearest_name") or "-")[:14]
        pbar.set_postfix_str(
            f"${n_scanned * 0.012:.2f} · "
            f"{dino_n}<>{guess}", refresh=False)
    pbar.close()

# 6) summary
print(flush=True)
print("=== SUMMARY ===", flush=True)
print(f"copied (tier-1 from poi_scan): {n_copied}", flush=True)
print(f"scanned (tier-2 Pro): {n_scanned}", flush=True)
print(f"errored: {n_errored}", flush=True)
print(f"output total now: {len(done) + n_copied + n_scanned} "
      f"rows in {args.output}", flush=True)
print(f"wall time: {time.time()-t0:.0f} s "
      f"({(time.time()-t0)/3600:.2f} h)", flush=True)
print(f"cost (scan only, $0.012/call): ~${n_scanned*0.012:.2f}",
      flush=True)

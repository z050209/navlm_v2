"""Step 13: independent VLM geo-localization cross-check.

WHY
---
DINOv2 visual matching always returns a nearest neighbour — cosine
similarity is relative, there is no "I don't know". In a city of
repetitive cobblestone lanes and old-town facades it will confidently
pair two *different* places that merely look alike, and step 10 only
catches that when a named POI is visible. On a plain street the wrong
match passes through.

This step adds a SECOND, INDEPENDENT location hypothesis: a VLM looks
at the frame and reasons about where in Zurich it is, with no knowledge
of the DINOv2 result. We then compare the two:

    variance_m = haversine(vlm_gps, match_gps)

    GEO_PASS     variance <= --variance-m            (two methods agree)
    GEO_FAIL     variance >  --variance-m AND the VLM was confident
                 -> the DINOv2/visual match is likely wrong; drop it
    GEO_UNKNOWN  VLM could not localize (low confidence / null)
                 -> no evidence either way; keep but flag

Backend-agnostic: dispatches through synth.backends.call_teacher, so it
runs on the Gemma vLLM teacher or any wired cloud VLM.

Run:
    python -m pipeline.step_13_vlm_geocheck \\
        --match  results/match_zurich.jsonl \\
        --frames data/cities/zurich/frame_starts_trusted_all.jsonl \\
        --out    results/geocheck_zurich.jsonl \\
        --backend gemma --vllm-url http://localhost:8003/v1 --model gemma
"""

import argparse
import json
import math
import re
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "toolbox"))

from synth.backends import call_teacher           # noqa: E402
from zurich_landmarks_gps import ZURICH_LANDMARKS  # noqa: E402
from scenery_pois import SCENERY_POIS              # noqa: E402

OSM_TABLE = _ROOT / "data/cities/zurich/landmarks_zurich_osm.json"
ZURICH_BBOX = (8.50, 47.35, 8.58, 47.40)   # sanity box for VLM coords

SYS_PROMPT = (
    "You are an expert at geo-localizing street-level photographs taken "
    "in central Zurich, Switzerland. You are given ONE photo and must "
    "decide where it was taken, using visible architecture, signage, "
    "streets, churches, the Limmat river and other cues."
)

USER_MSG = (
    "Where in central Zurich was this photo taken? Reason from what you "
    "see, then answer with STRICT JSON on the last line:\n"
    '{"landmark": "<nearest named street/square/landmark, or null>", '
    '"lat": <decimal degrees or null>, "lon": <decimal degrees or null>, '
    '"confidence": "high|medium|low", "reasoning": "<one sentence>"}\n'
    "Use confidence 'high' only if you recognise a specific place. If you "
    "genuinely cannot tell, use confidence 'low' and null for the rest."
)


def haversine_m(la1, lo1, la2, lo2):
    R = 6371000.0
    p1, p2 = math.radians(la1), math.radians(la2)
    dphi = math.radians(la2 - la1)
    dlam = math.radians(lo2 - lo1)
    a = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlam / 2) ** 2
    return 2 * R * math.asin(math.sqrt(a))


def build_poi_index():
    """name/alias (lowercased) -> (lat, lon). Curated tables + OSM."""
    idx = {}
    for name, (lat, lon, aliases) in ZURICH_LANDMARKS.items():
        idx[name.lower()] = (lat, lon)
        for a in aliases:
            idx[a.lower()] = (lat, lon)
    for name, s in SCENERY_POIS.items():
        idx[name.lower()] = (s["lat"], s["lon"])
        for a in s.get("aliases", []):
            idx[a.lower()] = (s["lat"], s["lon"])
    if OSM_TABLE.exists():
        for name, p in json.loads(OSM_TABLE.read_text(encoding="utf-8")).items():
            idx.setdefault(name.lower(), (p["lat"], p["lon"]))
    return idx


def parse_vlm_json(text):
    """Pull the last {...} blob out of the VLM response."""
    blobs = re.findall(r"\{[^{}]*\}", text, re.DOTALL)
    for b in reversed(blobs):
        try:
            return json.loads(b)
        except Exception:
            continue
    return None


def resolve_vlm_gps(parsed, poi_idx):
    """Return (lat, lon, source) or (None, None, None)."""
    if not parsed:
        return None, None, None
    name = (parsed.get("landmark") or "").strip().lower()
    if name and name in poi_idx:
        lat, lon = poi_idx[name]
        return lat, lon, "table"
    if name:  # loose substring match against table
        hit = next((k for k in poi_idx if name in k or k in name), None)
        if hit:
            lat, lon = poi_idx[hit]
            return lat, lon, "table~"
    lat, lon = parsed.get("lat"), parsed.get("lon")
    w, s, e, n = ZURICH_BBOX
    if isinstance(lat, (int, float)) and isinstance(lon, (int, float)) \
            and s <= lat <= n and w <= lon <= e:
        return float(lat), float(lon), "vlm_coords"
    return None, None, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--match", required=True,
                    help="visual_match_gps output jsonl (frame_id, gps, ...)")
    ap.add_argument("--frames", required=True,
                    help="jsonl mapping frame_id -> image path (field 'image')")
    ap.add_argument("--out", required=True)
    ap.add_argument("--variance-m", type=float, default=150.0,
                    help="VLM vs visual-match disagreement that triggers a drop")
    ap.add_argument("--backend", default="gemini",
                    choices=["gemini", "gemma", "anthropic", "openai"])
    ap.add_argument("--vllm-url", default="http://localhost:8003/v1")
    ap.add_argument("--model", default="gemini-2.5-flash")
    ap.add_argument("--limit", type=int, default=0, help="cap frames (0=all)")
    args = ap.parse_args()

    img_of = {}
    for ln in open(args.frames, encoding="utf-8"):
        r = json.loads(ln)
        if r.get("image"):
            img_of[r["frame_id"]] = r["image"]

    rows = [json.loads(l) for l in open(args.match, encoding="utf-8")]
    if args.limit:
        rows = rows[:args.limit]
    poi_idx = build_poi_index()
    print(f"[step13] {len(rows)} frames  backend={args.backend}  "
          f"poi_index={len(poi_idx)}  variance_thresh={args.variance_m}m")

    kw = {}
    if args.backend == "gemma":
        kw = {"vllm_url": args.vllm_url, "model": args.model}
    elif args.backend == "gemini":
        kw = {"model": args.model}

    counts = {"GEO_PASS": 0, "GEO_FAIL": 0, "GEO_UNKNOWN": 0, "ERROR": 0}
    with open(args.out, "w", encoding="utf-8") as fout:
        for i, r in enumerate(rows, 1):
            fid = r["frame_id"]
            match_gps = r.get("gps")
            img = img_of.get(fid)
            if not img or not Path(img).exists() or not match_gps:
                rec = {"frame_id": fid, "verdict": "ERROR",
                       "reason": "missing image or visual-match gps"}
                counts["ERROR"] += 1
                fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
                continue
            try:
                resp = call_teacher(args.backend, img, SYS_PROMPT, USER_MSG, **kw)
            except Exception as e:
                rec = {"frame_id": fid, "verdict": "ERROR", "reason": str(e)[:160]}
                counts["ERROR"] += 1
                fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
                continue

            parsed = parse_vlm_json(resp)
            conf = (parsed or {}).get("confidence", "low")
            vlat, vlon, vsrc = resolve_vlm_gps(parsed, poi_idx)

            if vlat is None:
                verdict, variance = "GEO_UNKNOWN", None
            else:
                variance = haversine_m(match_gps[0], match_gps[1], vlat, vlon)
                if variance <= args.variance_m:
                    verdict = "GEO_PASS"
                elif conf in ("high", "medium"):
                    verdict = "GEO_FAIL"     # VLM confidently disagrees -> drop
                else:
                    verdict = "GEO_UNKNOWN"  # VLM unsure; don't drop on a guess
            counts[verdict] += 1

            fout.write(json.dumps({
                "frame_id": fid,
                "verdict": verdict,
                "variance_m": (round(variance, 1) if variance is not None else None),
                "match_gps": match_gps,
                "vlm_gps": ([vlat, vlon] if vlat is not None else None),
                "vlm_gps_source": vsrc,
                "vlm_landmark": (parsed or {}).get("landmark"),
                "vlm_confidence": conf,
                "vlm_reasoning": (parsed or {}).get("reasoning"),
            }, ensure_ascii=False) + "\n")
            if i % 50 == 0:
                print(f"  [{i}/{len(rows)}] " + " ".join(f"{k}={v}" for k, v in counts.items()))

    print(f"[step13] done -> {args.out}")
    print("  " + "  ".join(f"{k}={v}" for k, v in counts.items()))
    dropped = counts["GEO_FAIL"]
    print(f"  {dropped} frames flagged GEO_FAIL "
          f"(DINOv2 match contradicted by VLM geo-localization)")


if __name__ == "__main__":
    main()

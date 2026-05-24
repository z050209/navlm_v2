"""Stage 6 — GPS recovery for video frames (DEV_MANUAL §2.5).

For each video frame:
  1. DINOv2 embed -> cosine-match against the Street View index (top-k)
  2. VLM (Gemini Pro) names the place -> resolved to GPS
  3. src.reconcile combines the two into an accept/reject + GPS
  4. heading = circular mean of the matched crops' rendered headings

Pure functions (cosine_topk, circular_mean) are unit-tested; embedding
needs DINOv2/GPU and the VLM step needs a GEMINI_API_KEY.

    python -m src.gps_recovery        # runs the full stage (heavy)
"""

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config                       # noqa: E402
from src import reconcile           # noqa: E402


def cosine_topk(query_emb, ref_embs, k=config.DINOV2_TOPK):
    """Top-k reference indices + cosine similarities for one query.

    query_emb: (D,) ; ref_embs: (N, D). Both assumed L2-normalised, so
    cosine = dot product. Returns (indices, sims), best-first.
    """
    import numpy as np
    sims = ref_embs @ query_emb
    order = np.argsort(-sims)[:k]
    return order, sims[order]


def circular_mean(degrees):
    """Circular mean of a list of headings (degrees), in [0, 360)."""
    if not degrees:
        return None
    x = sum(math.sin(math.radians(d)) for d in degrees)
    y = sum(math.cos(math.radians(d)) for d in degrees)
    return (math.degrees(math.atan2(x, y)) + 360) % 360


def circular_spread(degrees):
    """Circular standard deviation (degrees) — heading confidence proxy."""
    if not degrees:
        return 360.0
    n = len(degrees)
    x = sum(math.sin(math.radians(d)) for d in degrees) / n
    y = sum(math.cos(math.radians(d)) for d in degrees) / n
    r = math.sqrt(x * x + y * y)
    if r >= 0.9999:
        return 0.0
    if r < 1e-9:                       # fully dispersed -> max spread
        return 360.0
    return math.degrees(math.sqrt(-2 * math.log(r)))


def recover_frame(query_emb, ref):
    """Recover (gps, heading) for one frame.

    ref: dict with embs (N,D), gps [(lat,lon)...], headings [deg...],
         and `vlm` = a callable frame->(place_gps, confidence).
    Returns a reconcile result dict + heading fields.
    """
    idx, sims = cosine_topk(query_emb, ref["embs"])
    dino_gps = ref["gps"][int(idx[0])]
    heading = circular_mean([ref["headings"][int(i)] for i in idx])
    spread = circular_spread([ref["headings"][int(i)] for i in idx])

    vlm_gps, vlm_conf = ref["vlm"]()           # VLM place-naming
    result = reconcile.reconcile(dino_gps, float(sims[0]), vlm_gps, vlm_conf)
    result["heading"] = heading
    result["heading_spread"] = spread
    return result


def main():
    """End-to-end GPS recovery for the frames in `poi_scan.jsonl`.

    For every scan row:
      DINOv2 top-K SV matches      ->  (g_dino, s_dino) + heading
      geo_check_from_scan(row)     ->  (g_vlm, vlm_conf, place_name)
      reconcile.reconcile(...)     ->  accepted? + blended GPS + score
    Writes one JSON line per frame to
    `data/cities/zurich/gps_recovery.jsonl` (flushed live, crash-safe).
    """
    import argparse
    import json
    import numpy as np
    from tqdm import tqdm
    from src import reconcile
    from src.geo_check import geo_check_from_scan

    ap = argparse.ArgumentParser(
        description="GPS recovery — DINOv2 + VLM, weighted reconcile")
    ap.add_argument("-k", type=int, default=config.DINOV2_TOPK,
                    help="top-K SV matches per frame")
    ap.add_argument("--min-sim", type=float, default=0.60,
                    help="reject DINOv2 match below this cosine "
                         "(default 0.60, the pilot threshold)")
    ap.add_argument("--frame-cache", type=str, default="frames_n30_l0",
                    help="DINOv2 frame cache name "
                         "(under data/cities/zurich/dinov2/)")
    args = ap.parse_args()

    # OSM POIs — VLM guess -> GPS lookup
    pois = json.loads(
        (config.CITY_DIR / "pois.json").read_text(encoding="utf-8"))
    pois_map = {p["name"]: p for p in pois}

    # SV per-image meta — id -> (lat, lon, heading)
    sv_meta_path = config.STREETVIEW_DIR / "meta.jsonl"
    if not sv_meta_path.exists():
        sys.exit(f"no SV meta at {sv_meta_path} — copy it from the v1 "
                 f"streetview folder.")
    sv_meta = {}
    for line in sv_meta_path.open(encoding="utf-8"):
        m = json.loads(line)
        sv_meta[m["id"]] = (m["lat"], m["lon"], m.get("compass_angle", 0))

    # DINOv2 caches (sv refs + video frames)
    cdir = config.CITY_DIR / "dinov2"
    sv_cache = np.load(cdir / "sv_v1.npz", allow_pickle=True)
    sv_embs = sv_cache["embs"]
    sv_ids = [Path(p).stem for p in sv_cache["paths"]]   # image filename stem

    fpath = cdir / f"{args.frame_cache}.npz"
    if not fpath.exists():
        sys.exit(f"no DINOv2 frame cache at {fpath} — "
                 f"run `python -m src.dinov2_match --every-n 30` first")
    fcache = np.load(fpath, allow_pickle=True)
    frame_embs = fcache["embs"]
    frame_paths = [Path(p) for p in fcache["paths"]]
    # (video, frame_id) -> index into frame_embs
    frame_idx = {(p.parent.name, p.stem): i
                 for i, p in enumerate(frame_paths)}

    # POI-scan rows (the VLM signal, already computed)
    scan_path = config.CITY_DIR / "poi_scan.jsonl"
    scan_rows = [json.loads(l) for l in scan_path.open(encoding="utf-8")
                 if l.strip()]

    out_path = config.CITY_DIR / "gps_recovery.jsonl"
    counts = {"accepted": 0, "rejected_low_score": 0,
              "dino_weak": 0, "vlm_unresolved": 0,
              "no_frame_emb": 0}

    with out_path.open("w", encoding="utf-8") as fout:
        for row in tqdm(scan_rows, desc="[gps_recovery]", unit="frame"):
            key = (row["video"], row["frame_id"])
            i = frame_idx.get(key)
            if i is None:
                counts["no_frame_emb"] += 1
                continue

            # ── DINOv2 ─────────────────────────────────────────────
            idx, sims = cosine_topk(frame_embs[i], sv_embs, k=args.k)
            top_id = sv_ids[int(idx[0])]
            dino_loc = sv_meta.get(top_id)              # (lat,lon,head)
            s_dino = float(sims[0])
            headings = [sv_meta.get(sv_ids[int(j)], (0, 0, 0))[2]
                        for j in idx]
            heading = circular_mean(headings)
            spread = circular_spread(headings)

            # ── VLM ───────────────────────────────────────────────
            vlm = geo_check_from_scan(row, pois_map)

            rec = {
                "video": row["video"], "frame_id": row["frame_id"],
                "g_dino": [dino_loc[0], dino_loc[1]] if dino_loc else None,
                "s_dino": s_dino,
                "g_vlm": list(vlm["gps"]) if vlm["gps"] else None,
                "vlm_conf": vlm["confidence"],
                "place_guess": vlm["place_name"],
                "top_sv_id": top_id,
                "heading": heading, "heading_spread": spread,
            }

            # ── reconcile decision ────────────────────────────────
            if not dino_loc or s_dino < args.min_sim:
                rec.update({"accepted": False, "score": s_dino,
                            "variance_m": None, "gps": None,
                            "reject_reason": "dino_weak"})
                counts["dino_weak"] += 1
            elif not vlm["gps"]:
                rec.update({"accepted": False, "score": None,
                            "variance_m": None, "gps": None,
                            "reject_reason": "vlm_unresolved"})
                counts["vlm_unresolved"] += 1
            else:
                r = reconcile.reconcile(
                    (dino_loc[0], dino_loc[1]), s_dino,
                    vlm["gps"], vlm["confidence"])
                rec.update({"accepted": r["accepted"],
                            "score": r["score"],
                            "variance_m": r["variance_m"],
                            "gps": list(r["gps"]) if r["gps"] else None,
                            "reject_reason": (""
                                              if r["accepted"]
                                              else "low_score")})
                counts["accepted" if r["accepted"]
                       else "rejected_low_score"] += 1

            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
            fout.flush()

    print(f"[gps_recovery] wrote {out_path}")
    total = sum(counts.values())
    for k, v in counts.items():
        print(f"  {k:22s} {v:4d}  ({100 * v / max(1, total):.0f}%)")


if __name__ == "__main__":
    main()

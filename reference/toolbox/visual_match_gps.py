"""Match query (video-frame) embeddings to Mapillary reference and estimate GPS.

Usage
-----
    python toolbox/visual_match_gps.py \
        --reference data/mapillary/zurich/embeddings.npz \
        --query data/cities/zurich/frames/zurich_embeddings.npz \
        --out data/cities/zurich/frame_gps.jsonl \
        --topk 5 --max-dispersion-km 0.3

Output JSONL, one row per query frame:
    {
      "frame_id": "frame_00123",
      "top_matches": [ {"id":"...","lat":..,"lon":..,"sim":..}, ... ],
      "gps": [lat, lon],
      "dispersion_km": 0.12,
      "confidence": "high" | "medium" | "low"
    }
"""

import argparse
import json
import math
from pathlib import Path

import numpy as np


def haversine_km(lat1, lon1, lat2, lon2):
    """Great-circle distance in km."""
    R = 6371.0
    p1 = math.radians(lat1); p2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlam = math.radians(lon2 - lon1)
    a = math.sin(dphi/2)**2 + math.cos(p1)*math.cos(p2)*math.sin(dlam/2)**2
    return 2*R*math.asin(math.sqrt(a))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reference", required=True, help="Mapillary embeddings .npz (must have lat/lon in meta)")
    ap.add_argument("--query", required=True, help="Frame embeddings .npz")
    ap.add_argument("--out", required=True)
    ap.add_argument("--topk", type=int, default=5)
    ap.add_argument("--max-dispersion-km", type=float, default=0.3,
                    help="if top-k GPS spread > this, mark confidence=low")
    ap.add_argument("--min-sim", type=float, default=0.0,
                    help="absolute cosine-similarity floor: if the best match "
                         "scores below this, reject (confidence=low) regardless "
                         "of dispersion. DINOv2 always returns an argmax, so "
                         "without a floor a confident-but-wrong match passes.")
    ap.add_argument("--use-faiss", action="store_true")
    args = ap.parse_args()

    ref = np.load(args.reference, allow_pickle=True)
    q = np.load(args.query, allow_pickle=True)
    ref_embs = ref["embs"]
    ref_ids = ref["ids"]
    ref_meta = [json.loads(m) for m in ref["meta"]]
    q_embs = q["embs"]
    q_ids = q["ids"]

    print(f"[match] ref={ref_embs.shape}  query={q_embs.shape}  topk={args.topk}")

    # Similarity search (embeddings are L2-normalised, so cosine = dot)
    if args.use_faiss:
        import faiss
        d = ref_embs.shape[1]
        index = faiss.IndexFlatIP(d)
        index.add(ref_embs)
        sims, idxs = index.search(q_embs, args.topk)
    else:
        # Simple numpy: for <50k refs it's fast enough
        # q_embs (Nq, d) · ref_embs.T (d, Nr) → (Nq, Nr)
        sim_mat = q_embs @ ref_embs.T
        idxs = np.argsort(-sim_mat, axis=1)[:, :args.topk]
        sims = np.take_along_axis(sim_mat, idxs, axis=1)

    n_high = n_med = n_low = 0
    with open(args.out, "w") as fout:
        for qi, q_id in enumerate(q_ids):
            matches = []
            for rank in range(args.topk):
                j = int(idxs[qi, rank])
                m = ref_meta[j]
                matches.append({
                    "id": str(ref_ids[j]),
                    "lat": m.get("lat"),
                    "lon": m.get("lon"),
                    "sim": float(sims[qi, rank]),
                })
            # Aggregate: median of top-k
            lats = [m["lat"] for m in matches if m["lat"] is not None]
            lons = [m["lon"] for m in matches if m["lon"] is not None]
            if not lats:
                gps = None
                disp = None
                conf = "low"
            else:
                lat_med = float(np.median(lats))
                lon_med = float(np.median(lons))
                # dispersion = max pairwise distance
                disp = 0.0
                for i in range(len(lats)):
                    for j in range(i + 1, len(lats)):
                        d = haversine_km(lats[i], lons[i], lats[j], lons[j])
                        if d > disp:
                            disp = d
                gps = [lat_med, lon_med]
                best_sim = float(sims[qi, 0])
                if best_sim < args.min_sim:
                    # absolute floor fails → don't trust this match at all
                    conf = "low"; n_low += 1
                elif disp <= args.max_dispersion_km * 0.5:
                    conf = "high"; n_high += 1
                elif disp <= args.max_dispersion_km:
                    conf = "medium"; n_med += 1
                else:
                    conf = "low"; n_low += 1

            fout.write(json.dumps({
                "frame_id": str(q_id),
                "top_matches": matches,
                "gps": gps,
                "dispersion_km": disp,
                "best_sim": (float(sims[qi, 0]) if len(matches) else None),
                "confidence": conf,
            }) + "\n")

    total = len(q_ids)
    print(f"[match] done: high={n_high}  medium={n_med}  low={n_low}  "
          f"(of {total} query frames; {100*(n_high+n_med)/total:.1f}% usable)")


if __name__ == "__main__":
    main()

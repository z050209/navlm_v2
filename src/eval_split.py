"""Build the held-out evaluation set for the single POI-region ablation.

Reads annotations_<variant>.jsonl (the verifier-passed Phase B output)
and produces TWO files:

  eval_test_poi.jsonl     — rows whose destination POI is in the
                            held-out POI region (= 20 % spatially-
                            outermost of the destinations actually
                            used). Tests "POI-region generalisation":
                            can the model route to destinations it
                            never saw labelled in training?
  eval_train.jsonl        — everything else (LoRA SFT input)

**Design note — only one ablation.** The earlier two-ablation design
(adding a saturday_morning video hold-out) was dropped because the
video-hold-out accuracy was confounded by a single failure mode
(the L-given LoRA never emits "turn around" verbs, and the video
hold-out has 2× more turn-around ground-truth labels than the POI
hold-out — see DEV_MANUAL §4.7.4). The POI-region hold-out is the
cleaner test of generalisation: novel destinations on visually-
familiar scenery, isolates the routing-learning signal from visual
novelty. saturday_morning frames now join the training pool.

For backwards compatibility the script still accepts `--keep-video`
to also produce eval_test_video.jsonl + exclude saturday_morning
from training, but this is OFF by default.

  python -m src.eval_split                           # default annotations
  python -m src.eval_split --input annotations_strict.jsonl
  python -m src.eval_split --keep-video              # legacy two-ablation
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config                                  # noqa: E402

DEFAULT_HOLDOUT_POIS = getattr(config, "HOLDOUT_POIS", set())


def _holdout_pois(point_pois, frac=0.2):
    """Spatial holdout: the 20 % furthest POIs from the visited centroid.
    Deterministic — same input, same output."""
    if not point_pois:
        return set()
    lats = [p["lat"] for p in point_pois]
    lons = [p["lon"] for p in point_pois]
    clat, clon = sum(lats) / len(lats), sum(lons) / len(lons)
    scored = sorted(point_pois,
                    key=lambda p: (p["lat"] - clat) ** 2
                                 + (p["lon"] - clon) ** 2,
                    reverse=True)
    k = max(1, int(len(scored) * frac))
    return {p["name"] for p in scored[:k]}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("--input", default=None,
                    help="annotations*.jsonl; default = latest in CITY_DIR")
    ap.add_argument("--video", default=config.HOLDOUT_VIDEO,
                    help=f"video to hold out IF --keep-video is set "
                         f"(default {config.HOLDOUT_VIDEO})")
    ap.add_argument("--holdout-frac", type=float, default=0.2,
                    help="POI fraction for the POI ablation (default 0.2)")
    ap.add_argument("--keep-video", action="store_true",
                    help="legacy two-ablation mode — also write "
                         "eval_test_video.jsonl and exclude --video "
                         "rows from training. OFF by default; the "
                         "current pipeline is single-ablation "
                         "(POI hold-out only). See eval_split.py "
                         "docstring for rationale.")
    args = ap.parse_args()

    if args.input:
        in_path = Path(args.input)
    else:
        cand = sorted(config.CITY_DIR.glob("annotations*.jsonl"))
        if not cand:
            sys.exit("[eval_split] no annotations*.jsonl found in "
                     f"{config.CITY_DIR}")
        in_path = cand[-1]
    print(f"[eval_split] in:  {in_path.name}", flush=True)

    records = [json.loads(l) for l in in_path.open(encoding="utf-8")
               if l.strip()]
    records = [r for r in records if r.get("accepted")]
    print(f"[eval_split] verifier-passed annotations: {len(records)}",
          flush=True)

    # Restrict the holdout-POI computation to the POIs that ACTUALLY
    # appear as destinations in the data (typically the top-30 from
    # the production pool — see §2.7.2). Picking 20 % of those gives
    # 6 held-out destinations, vs. 257 obscure POIs you get if you
    # compute over all 1,289 pois.json entries (most of which never
    # appear in any sample). Centroid GPS of each used POI is its
    # median (lat, lon) across samples where it was the destination.
    dest_centroids = {}
    for r in records:
        dest_centroids.setdefault(r["dest_name"], []).append(
            tuple(r["dest_gps"]))
    used_dest_pois = []
    for name, pts in dest_centroids.items():
        lats = sorted(p[0] for p in pts)
        lons = sorted(p[1] for p in pts)
        used_dest_pois.append({"name": name,
                                "lat": lats[len(lats) // 2],
                                "lon": lons[len(lons) // 2]})
    holdout = DEFAULT_HOLDOUT_POIS or _holdout_pois(
        used_dest_pois, args.holdout_frac)
    print(f"[eval_split] destinations used in data: {len(used_dest_pois)}",
          flush=True)
    print(f"[eval_split] holdout POIs ({len(holdout)} = "
          f"{100*args.holdout_frac:.0f}% spatial outliers): "
          f"{sorted(holdout)}", flush=True)

    video_path = config.CITY_DIR / "eval_test_video.jsonl"
    poi_path = config.CITY_DIR / "eval_test_poi.jsonl"
    train_path = config.CITY_DIR / "eval_train.jsonl"

    n_v = n_p = n_t = 0
    write_video = bool(args.keep_video)
    fv = video_path.open("w", encoding="utf-8") if write_video else None
    try:
        with poi_path.open("w", encoding="utf-8") as fp, \
                train_path.open("w", encoding="utf-8") as ft:
            for r in records:
                line = json.dumps(r, ensure_ascii=False) + "\n"
                in_video = (r["video"] == args.video) if write_video else False
                in_poi = r["dest_name"] in holdout
                if in_video and fv is not None:
                    fv.write(line); n_v += 1
                if in_poi:
                    fp.write(line); n_p += 1
                # train set = neither the holdout video nor the holdout POI
                if not in_video and not in_poi:
                    ft.write(line); n_t += 1
    finally:
        if fv is not None:
            fv.close()

    if write_video:
        print(f"[eval_split] eval_test_video: {n_v} -> {video_path.name} "
              f"(legacy --keep-video mode)", flush=True)
    else:
        print(f"[eval_split] eval_test_video: SKIPPED (single-ablation mode; "
              f"saturday_morning rows kept in training)", flush=True)
    print(f"[eval_split] eval_test_poi:   {n_p} -> {poi_path.name}",
          flush=True)
    print(f"[eval_split] eval_train:      {n_t} -> {train_path.name}",
          flush=True)


if __name__ == "__main__":
    main()

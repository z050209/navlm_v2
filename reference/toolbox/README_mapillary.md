# Mapillary visual-matching GPS pipeline (v2, multi-signal)

Give our GPS-less walking-tour frames an estimated lat/lon by
**combining three signals**:

1. **Visual similarity** to crowd-sourced Mapillary images (have GPS)
   → `embed_images.py` (DINOv2-avg / MixVPR) + `visual_match_gps.py`
2. **Temporal consistency**: adjacent frames can't be 500m apart
   → `temporal_smooth.py` (window-median outlier replacement)
3. **OCR landmark override**: if a sign names a known Zurich landmark,
   use that landmark's GPS directly (precision ~10–50 m)
   → `ocr_augment.py` + `landmark_match.py`
4. **Priority merge** with `merge_gps_sources.py`: OCR > smoothed-visual > raw-visual.

## Prereq

1. Register at <https://www.mapillary.com/developer>
2. Create an app, copy the **client token** (starts `MLY|`)
3. `export MAPILLARY_TOKEN="MLY|xxx"`

`fetch_mapillary.py` must run on a host that can reach
`graph.mapillary.com` — the server behind this repo cannot, so fetch
on your laptop and rsync the result up.

## Pipeline (v2)

```bash
# (1) Download ~5k Mapillary images covering Zurich centre  [LAPTOP]
python toolbox/fetch_mapillary.py --city zurich \
  --bbox 8.528,47.366,8.557,47.385 \
  --limit 5000
rsync -avz data/mapillary/zurich/ user@server:/pub/.../navlm/data/mapillary/zurich/

# (2) Visual embedding — method=avg is AnyLoc-Lite (beats CLS for VPR)
CUDA_VISIBLE_DEVICES=3 python toolbox/embed_images.py \
  --method avg --images data/mapillary/zurich/images \
  --meta data/mapillary/zurich/meta.jsonl \
  --out data/mapillary/zurich/embeddings.npz

CUDA_VISIBLE_DEVICES=3 python toolbox/embed_images.py \
  --method avg --images data/cities/zurich/frames/zurich \
  --out data/cities/zurich/frames/zurich_embeddings.npz

# (3) Visual match (top-k median GPS)
python toolbox/visual_match_gps.py \
  --reference data/mapillary/zurich/embeddings.npz \
  --query data/cities/zurich/frames/zurich_embeddings.npz \
  --out data/cities/zurich/frame_gps.jsonl \
  --topk 5 --max-dispersion-km 0.3

# (4) Temporal smoothing — fix outliers
python toolbox/temporal_smooth.py \
  --in data/cities/zurich/frame_gps.jsonl \
  --out data/cities/zurich/frame_gps_smoothed.jsonl \
  --window 5 --outlier-km 0.3

# (5) OCR every frame with Qwen3-VL-32B (vLLM port 8001)
python toolbox/ocr_augment.py \
  --frames data/cities/zurich/frames/zurich \
  --out data/cities/zurich/frame_ocr.jsonl \
  --vllm-url http://localhost:8001/v1 \
  --model Qwen/Qwen3-VL-32B-Instruct

# (6) Match OCR text → hard-coded Zurich landmark GPS
python toolbox/landmark_match.py \
  --ocr data/cities/zurich/frame_ocr.jsonl \
  --out data/cities/zurich/frame_gps_ocr.jsonl

# (7) Priority-merge the 3 sources
python toolbox/merge_gps_sources.py \
  --dino data/cities/zurich/frame_gps.jsonl \
  --dino-smoothed data/cities/zurich/frame_gps_smoothed.jsonl \
  --ocr data/cities/zurich/frame_gps_ocr.jsonl \
  --out data/cities/zurich/frame_gps_final.jsonl \
  --disagreement-km 0.5

# (8) Inject into existing annotations
python toolbox/inject_gps_annotations.py \
  --annotations data/cities/zurich/annotations_gemma.jsonl \
  --frame-gps data/cities/zurich/frame_gps_final.jsonl \
  --out data/cities/zurich/annotations_gemma_geo.jsonl \
  --min-confidence medium
```

## Expected timings (on the current cluster)

| Step | Expected |
|---|---|
| fetch_mapillary (5k thumb_256 images) | ~20 min on laptop |
| rsync to server (~1.2 GB) | ~2–5 min |
| embed_images ×2 (DINOv2-base on one L20X) | ~5–8 min each |
| visual_match_gps (numpy matmul) | <1 min |
| temporal_smooth | <5 s |
| ocr_augment (vLLM Qwen3-VL-32B, 2224 frames @ 0.9 fps) | ~40 min |
| landmark_match | <2 s |
| merge_gps_sources | <5 s |
| inject_gps_annotations | <2 s |
| **Total (parallelising embed with OCR)** | **~50–60 min** |

## Expected accuracy per source

| Source | Median error | Useful for |
|---|---|---|
| DINOv2-avg + Mapillary top-5 median | 100–300 m (worse on repetitive streets) | coarse city-region |
| + temporal smoothing | 80–200 m (outliers cleaned) | coarse but more robust |
| OCR landmark match (only when a sign visible) | 10–50 m | precise anchor points |
| Merged (OCR where we have it, dino elsewhere) | 10–300 m | downstream map_context |

## Method choice for `embed_images.py`

| `--method` | Notes |
|---|---|
| `cls` | DINOv2 CLS token — simplest, used in v1 pipeline. Baseline. |
| `avg` ⭐ | DINOv2 patch-token average (AnyLoc-Lite). **Recommended default**: ~5–10 % VPR gain over `cls`, no extra dependency. |
| `mixvpr` | MixVPR checkpoint via torch.hub — strongest VPR accuracy but needs an extra clone/checkpoint download; may fail on a firewalled host. |

## Troubleshooting

- **`graph.mapillary.com` unreachable** — fetch on your laptop, rsync to server.
- **MixVPR load fails** — fall back to `--method avg`, which is 95% of the way there and has zero extra dependencies.
- **OCR returns `[]` for many frames** — normal for residential streets with no signage. Just means the merge step will fall back to the DINO/smoothed source for those frames.
- **`landmark_match.py` gives 0 matches** — extend `zurich_landmarks_gps.py` with more aliases / landmarks. It's just a plain dict you can add entries to.

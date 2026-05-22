# NavLM Toolbox

Tools to turn raw walking-tour videos + Mapillary street-view + OSM map data
into instruction-tuning samples for a Qwen2.5-VL-7B + LoRA model that
translates **(GPS + map context + camera frame)** → **scene-anchored
walking directions** (e.g. "turn right past the glass-fronted building").

## Two paths through the toolbox

### v2 — current path (used since 2026-04)
Tool-augmented inference architecture with Chain-of-Thought training data.
Main entry: `synth_unified.py`. Strict per-frame visual verification.

```
extract_frames.py / extract_extra_videos.py     ── raw video → frames
fetch_mapillary[_tiled].py                      ── Mapillary refs (laptop)
embed_images.py                                 ── DINOv2 embeddings

visual_match_gps.py                             ── frame → estimated GPS
refine_visual_match.py                          ── consensus filter (NEW)
ocr_augment.py + landmark_match.py              ── OCR landmark → GPS
merge_gps_sources.py                            ── priority merge
compute_frame_heading.py                        ── compass median
build_trusted_starts.py                         ── high-confidence starts pool
extract_osm_pois.py                             ── auto OSM POI table
scan_mapillary_landmarks.py                     ── Gemma yes/no Mapillary scan
way_planner.py                                  ── osmnx + networkx routing tool

synth_unified.py    ★ MAIN                       ── synth → messages JSONL
synth_utils.py                                   ── shared geom / VLM-call

train_city_lora.py                              ── train Qwen2.5-VL + LoRA
```

### v1 — legacy path (`_legacy/`)
The original three-algorithm synthesizer. Still functional but produces
the older non-CoT annotations format. See `_legacy/README.md`.

`run_city.sh` is the v1 one-shot orchestrator that calls v1 modules.

---

## Typical v2 invocation

```bash
# 1. (LAPTOP) download Mapillary, rsync to server
python toolbox/fetch_mapillary_tiled.py --city zurich \
       --bbox 8.520,47.360,8.570,47.395 --tile-size-m 200

# 2. extract video frames
python toolbox/extract_frames.py --city zurich --dedup
python toolbox/extract_extra_videos.py   # for additional videos in Zurich_extra/

# 3. visual matching for video frame GPS
python toolbox/embed_images.py --method avg \
       --images data/cities/mapillary/zurich/images \
       --meta   data/cities/mapillary/zurich/meta.jsonl \
       --out    data/cities/mapillary/zurich/embeddings.npz

python toolbox/embed_images.py --method avg \
       --images data/cities/zurich/frames/zurich \
       --out    data/cities/zurich/frames/zurich_embeddings.npz

python toolbox/visual_match_gps.py \
       --reference data/cities/mapillary/zurich/embeddings.npz \
       --query     data/cities/zurich/frames/zurich_embeddings.npz \
       --out       data/cities/zurich/frame_gps.jsonl --topk 5

python toolbox/refine_visual_match.py     # GPS+compass consensus filter

# 4. OCR + landmark grounding
python toolbox/ocr_augment.py
python toolbox/landmark_match.py
python toolbox/merge_gps_sources.py

# 5. heading per frame
python toolbox/compute_frame_heading.py

# 6. trusted starts pool (OCR ∪ visual_consensus)
python toolbox/build_trusted_starts.py

# 7. OSM POI table (one-shot per city)
python toolbox/extract_osm_pois.py \
       --bbox 8.520,47.360,8.570,47.395

# 8. (optional) extend trusted starts via Mapillary scan
python toolbox/scan_mapillary_landmarks.py \
       --max-distance-m 80

# 9. ★ synthesize training data
python toolbox/synth_unified.py \
       --backend gemma \
       --vllm-url http://localhost:8003/v1 \
       --model    google/gemma-4-31b-it

# 10. fine-tune
python toolbox/train_city_lora.py \
       --train data/cities/zurich/synth_unified.jsonl \
       --output_dir results/lora_zurich_cot
```

---

## Layout

```
toolbox/
├── README.md                       ← this file
├── synth_unified.py    ★           ← v2 main synthesizer
├── synth_utils.py                  ← shared utilities
├── way_planner.py                  ── osmnx routing tool
│
├── extract_frames.py               ── ingest
├── extract_extra_videos.py
├── expand_frames.py                ── (one-shot data expansion)
├── fetch_videos.py
├── fetch_mapillary.py
├── fetch_mapillary_tiled.py
│
├── embed_images.py                 ── DINOv2
├── visual_match_gps.py             ── GPS estimation
├── refine_visual_match.py
├── temporal_smooth.py
├── ocr_augment.py
├── landmark_match.py
├── merge_gps_sources.py
├── compute_frame_heading.py
├── build_trusted_starts.py
│
├── extract_osm_pois.py             ── OSM POI table
├── zurich_landmarks_gps.py         ── hand-written landmark table (OCR)
├── scan_mapillary_landmarks.py     ── extend trusted starts
│
├── train_city_lora.py              ── LoRA fine-tune
├── auto_annotate.py                ── (v1 annotation helper, still used)
├── build_sft.py
├── build_eval_map_context.py
├── scene_inventory.py
├── build_ood_from_photos.py
├── config.py
│
├── run_city.sh                     ── v1 one-shot orchestrator
├── README_mapillary.md             ── Mapillary-specific cheat sheet
└── _legacy/                        ── deprecated, don't use for new work
    ├── README.md
    ├── synth_algo1_video_routes.py
    ├── synth_algo2_visible_scene.py
    ├── synth_algo3b_osm_routed.py
    └── inject_gps_annotations.py
```

---

## Output artifacts (per city)

```
data/cities/<city>/
├── videos/                          ── source MP4 + Zurich_extra/
├── frames/<video>/*.jpg             ── scene-deduped keyframes
├── frame_gps.jsonl                  ── visual-match raw + top_matches
├── frame_gps_refined.jsonl          ── + GPS/compass consensus columns
├── frame_gps_smoothed.jsonl
├── frame_gps_ocr.jsonl              ── OCR landmark anchors
├── frame_gps_final.jsonl            ── priority-merged
├── frame_heading.jsonl              ── per-frame compass
├── frame_starts_trusted.jsonl       ★ start pool
├── landmarks_zurich_osm.json        ── auto OSM POI table
├── osm_walking.pkl                  ── pickled walking graph
├── synth_unified.jsonl              ★ training data
└── frame_ocr.jsonl                  ── raw OCR text per frame

results/lora_<city>_cot/
└── adapter_model.safetensors        ★ trained LoRA adapter
```

---

See `docs/pipeline.md` for the full end-to-end walkthrough including the
research thesis and inference-time architecture.

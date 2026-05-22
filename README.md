# NavLM v2

Self-supervised navigation VLM — fine-tune a VLM to give spoken walking
directions from a phone photo + GPS, *without a compass*. From-scratch
rebuild; see **`DEV_MANUAL.md`** for the full design and **`logs/infra.md`**
for accounts, budget and cost tracking.

## Layout

```
config.py        paths, bbox, thresholds, model names (relative paths only)
src/             the pipeline — run as `python -m src.<module>`
reference/       old navlm_ss code, read-only
logs/            daily logs + infra.md
viz/             generated HTML visualizations
DEV_MANUAL.md    design doc
```

## Setup

- Reuses the `navlm_ss/.venv` (torch 2.5.1+cu124, transformers, modal).
- Extra deps: `pip install yt-dlp imagehash opencv-python` ; `ffmpeg` on PATH.
- Copy `.env.example` → `.env` and fill in the API keys.
- Data dir defaults to `./data`; override with the `NAVLM_DATA` env var.

## Phase A — run order

```bash
python config.py                    # sanity-check resolved paths
python -m src.download_videos        # 8 YouTube videos -> DATA_ROOT/.../videos
python -m src.extract_frames         # videos -> quality-filtered frames
```

More stages (Street View crawl, GPS recovery, OSM/HMM) land as they're
built — tracked in `DEV_MANUAL.md` §9.

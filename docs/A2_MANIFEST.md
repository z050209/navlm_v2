# Attempt 2 — File Manifest

Single index for every file created during Attempt 2 (landmark-named
destinations re-annotation). See `DEV_MANUAL.md` §6 for the design
rationale; this doc just lists files.

## Naming convention

- **Scripts**: `src/a2_*.py`
- **Data outputs**: `data/cities/zurich/a2/*.jsonl`
- **Visualisations**: `viz/a2_*.html`
- **Terminology**: use **"attraction"** consistently for the 21-name
  curated destination vocabulary. Avoid mixing "landmark" / "POI" /
  "famous-X" terms.

---

## 1. Source-of-truth constants

The 21 attractions and their aliases are defined ONCE in:

| File | Purpose |
|---|---|
| `src/a2_attraction_slots.py` (top of file) | Defines `ATTRACTIONS_21` (EN, ZH, lat, lon, kind) and `ALIASES` (variant spellings the VLM produces). Every other `a2_*` script imports from here. |

If you need to add/remove an attraction or alias, **edit this one place** and rerun the downstream scripts.

---

## 2. Scripts

All scripts read upstream files (e.g. `gps_recovery_full.jsonl`,
`poi_scan*.jsonl`) and write only into `data/cities/zurich/a2/` (or
stdout). They DO NOT mutate upstream pipeline files.

### `src/a2_proximity_tag.py`
- **Purpose**: For every VLM-confirmed accepted frame, tag the nearest of the 21 attractions by GPS distance + lists of attractions within 50/100/150 m.
- **Input**: `gps_recovery_full.jsonl` (default)
- **Output**: `data/cities/zurich/a2/proximity_tag.jsonl`
- **Run**: `python -m src.a2_proximity_tag --tier 1 --radius 100`
- **Schema** (per row): `{video, frame_id, gps, place_guess, nearest_landmark, nearest_dist_m, landmarks_within_50m, landmarks_within_100m, landmarks_within_150m}`
- **Status**: Superseded by `a2_attraction_slots.py` for the slot map. Kept for spot-checking individual frames.

### `src/a2_vlm_coverage.py`
- **Purpose**: Diagnostic. For each of the 21 attractions, how often the VLM mentioned it in `visible[]` or `guess[]` across the merged scan files.
- **Input**: `poi_scan.jsonl` + `poi_scan_cos0.75.jsonl` (both merged by default)
- **Output**: stdout only (table)
- **Run**: `python -m src.a2_vlm_coverage`
- **Use**: Answers "did the VLM ever see attraction X?" — found the 3 attractions with zero coverage (Polyterrasse, Bürkliplatz, Sechseläutenplatz in the older 27-list).

### `src/a2_raw_vlm_strings.py`
- **Purpose**: Diagnostic. Dump the raw VLM `guess` and `visible[]` strings, ranked by frequency, with markers showing which contain a canonical attraction name. Reveals spelling variance.
- **Input**: `poi_scan.jsonl` (default) or `poi_scan_cos0.75.jsonl` via `--input`
- **Output**: stdout only (long table)
- **Run**: `python -m src.a2_raw_vlm_strings --head 60`
- **Use**: How we discovered that the VLM uses "Kirche St. Peter", "St. Peter's Church", etc. as variants of "St. Peter" → drove the alias table.

### `src/a2_join_3way.py`
- **Purpose**: 3-way join of (proximity tag) × (gps_recovery row) × (VLM scan row) per frame. Classifies each frame into BOTH-agree / DIFFERENT-attraction / CANONICAL-only / no-VLM-data quadrants.
- **Inputs**: `a2/proximity_tag.jsonl`, `gps_recovery_full.jsonl`, `poi_scan*.jsonl` (both)
- **Outputs**: `data/cities/zurich/a2/join_3way.jsonl`, `data/cities/zurich/a2/join_3way.tsv`
- **Run**: `python -m src.a2_join_3way --radius 100`
- **Use**: Per-frame audit table. The TSV is openable in Excel for spot-checking.

### `src/a2_sanity_check.py`
- **Purpose**: Verify the VLM↔DINOv2 mapping is sound. Reports how many VLM-confirmed frames have an actual VLM scan row, how many of those resolve, how the two place names agree.
- **Input**: `gps_recovery_full.jsonl`
- **Output**: stdout only
- **Run**: `python -m src.a2_sanity_check`
- **Status**: Created but never run yet. Run before any commit that depends on the gps_recovery output.

### `src/a2_match_strict.py`
- **Purpose**: Strict per-frame filter. Keep a frame iff the VLM (in `visible[]` or `guess`) EXACTLY names one of the top-N nearest attractions by GPS. No 250 m neighborhood slack.
- **Input**: `gps_recovery_full.jsonl` + both `poi_scan*` files
- **Output**: `data/cities/zurich/a2/match_strict.jsonl`
- **Run**: `python -m src.a2_match_strict --top-n 3 --max-dist 300`
- **Use**: The "trusted-by-exact-match" cohort. Currently produces ~752 frames at top-3/300m.

### `src/a2_attraction_slots.py`
- **Purpose**: Per-attraction slot map. For each of the 21 attractions, list the set of `(video, frame_id, SV slot)` tuples that represent it, with evidence type (E1 visible, E2 guess, E3 proximity).
- **Input**: `gps_recovery_full.jsonl` + both `poi_scan*` files
- **Output**: `data/cities/zurich/a2/attraction_slots.jsonl`
- **Run**: `python -m src.a2_attraction_slots --radius 250`
- **Caveat**: At R=250 m the proximity gate is very generous (every old-town frame is within range of 5-8 attractions). For meaningful per-attraction counts, use R=50-80 m or drop E3 and use only E1+E2.

### `src/a2_sv_pano_attractions.py`
- **Purpose**: Map each SV crop (the ground-truth GPS reference set DINOv2 matches frames to) to one consensus attraction. Combines proximity to the crop's GT GPS + VLM evidence from frames matched to it.
- **Input**: `data/cities/streetview/zurich/meta.jsonl` (the 4,431 SV crops) + `gps_recovery_full.jsonl` + both `poi_scan*` files
- **Output**: `data/cities/zurich/a2/sv_attractions.jsonl`
- **Run**: `python -m src.a2_sv_pano_attractions --proximity-radius 100`
- **Result**: 1,023 of 4,431 crops matched by ≥1 frame; 809 get an attraction tag; the highest-confidence set is 23 crops where vlm_visible + vlm_guess + proximity all agree.

---

## 3. Data outputs

All in `data/cities/zurich/a2/`.

### `a2/proximity_tag.jsonl`
- **Source**: `a2_proximity_tag.py`
- **Rows**: ~2,470 (one per VLM-confirmed accepted frame)
- **Purpose**: Per-frame "what attractions are near this GPS" lookup table.

### `a2/join_3way.jsonl` + `join_3way.tsv`
- **Source**: `a2_join_3way.py`
- **Rows**: 1,858 (frames within 100 m of a canonical attraction)
- **Purpose**: Audit table showing how proximity-tag, gps_recovery's resolved name, and raw VLM evidence agree per frame. TSV format for Excel spot-checking.

### `a2/match_strict.jsonl`
- **Source**: `a2_match_strict.py`
- **Rows**: 752 (frames whose VLM exact-mentions one of top-3 nearest attractions within 300 m)
- **Purpose**: High-precision "trusted attraction" cohort for re-annotation.

### `a2/attraction_slots.jsonl`
- **Source**: `a2_attraction_slots.py`
- **Rows**: 21 (one per attraction)
- **Purpose**: Per-attraction list of frames + SV slots with evidence breakdown. Source-of-truth for "which frames represent attraction X".

### `a2/sv_attractions.jsonl`
- **Source**: `a2_sv_pano_attractions.py`
- **Rows**: 4,431 (one per SV crop)
- **Purpose**: GT-GPS anchor mapping. For every SV crop in our reference set, the best-guess attraction it represents (with consensus_source labeling whether it's proximity / VLM-visible / VLM-guess / combo).

---

## 4. Upstream files we READ (not in a2/, do not modify)

| File | Origin | Why we read it |
|---|---|---|
| `data/cities/zurich/gps_recovery_full.jsonl` | `src/gps_recovery.py` (Attempt 1) | Frame-level DINOv2 GPS + VLM-resolved place name + tier markers |
| `data/cities/zurich/poi_scan.jsonl` | `src/poi_scan.py` (every-10 baseline) | Raw VLM `visible[]` + `guess` for 872 frames |
| `data/cities/zurich/poi_scan_cos0.75.jsonl` | `_vlm_test.py` (cos≥0.75 expansion) | Same schema, additional 4,101 frames |
| `data/cities/streetview/zurich/meta.jsonl` | `src/streetview.py` (Attempt 1) | The 4,431 SV crops' GT GPS coordinates |
| `data/cities/zurich/pois.json` | `src/pois.py` (Attempt 1) | The ~1,289-entry OSM POI table |

## 4a. Supplementary data files (manually-curated, in a2/)

| File | Purpose |
|---|---|
| `data/cities/zurich/a2/extra_pois.json` | Hand-curated supplementary OSM entries for landmarks that OSM has but `src/pois.py:POINT_TAGS` filter dropped (Paradeplatz tagged `place=square`, Rathaus tagged `building=public`). Same schema as `pois.json`. Auto-merged by `a2_step1_gps_geo.py`. Coords verified against Nominatim. |

---

## 5. Files deleted (cleanup record)

| File | Reason deleted |
|---|---|
| `data/cities/zurich/gps_recovery_dedup_best_per_slot.jsonl` | One-off exploration (slot dedup test) |
| `data/cities/zurich/gps_recovery_full_exact.jsonl` | Filter A output, superseded by `match_strict.jsonl` |
| `data/cities/zurich/landmark_audit.jsonl` (older 27-POI version) | Renamed → `a2/proximity_tag.jsonl` |
| `_watchdog_annotate_incremental.py` | Stale watchdog from earlier session |
| `src/landmark_audit.py` | Renamed → `src/a2_proximity_tag.py` |
| `src/poi_scan_audit.py` + `src/cos75_visible_audit.py` | Merged → `src/a2_vlm_coverage.py` |
| `src/poi_scan_raw_names.py` | Renamed → `src/a2_raw_vlm_strings.py` |
| `src/landmark_vs_vlm.py` | Renamed → `src/a2_join_3way.py` |
| `src/vlm_dino_sanity.py` | Renamed → `src/a2_sanity_check.py` |
| `src/attraction_match.py` | Renamed → `src/a2_match_strict.py` |
| `src/landmark_slots.py` | Renamed → `src/a2_attraction_slots.py` |

---

## 6. What's still to build

| Step | Planned script | Purpose |
|---|---|---|
| Per-frame attraction-via-SV-pano | `src/a2_frame_attractions.py` | Use `sv_attractions.jsonl` to inherit the SV crop's consensus attraction to every matched video frame. Replaces ad-hoc per-frame proximity tagging. |
| Re-annotation prompt v2 | `src/a2_annotate.py` | Run Gemini Pro 2.5 with landmark destinations (drop checkpoints, use 21-attraction vocabulary). |
| Eval split for landmark holdout | `src/a2_eval_split.py` | 20% landmark-region holdout from the 21 attractions. |

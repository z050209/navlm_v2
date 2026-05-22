"""Pipeline configuration: video list, paths, filter thresholds.

All canonical names live here. Other modules import from this file rather
than hard-coding paths. Paths are resolved relative to the repo root
(detected from this file's location).
"""

from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
DATA = REPO / "data" / "cities" / "zurich"
PIPELINE_OUT = DATA / "pipeline"
LOGS = REPO / "pipeline" / "logs"

# Mapillary reference index (used by visual_match_gps + refine_visual_match).
# Two indices supported via the `variant` parameter to paths_for():
#   ""       → original 5k thumb index   data/cities/mapillary/zurich/
#   "_hq"    → 22.6k 4K HQ index         data/mapillary/zurich_altstadt_hq/
MAPILLARY_INDEX = {
    "":    {"meta": REPO / "data/cities/mapillary/zurich/meta.jsonl",
            "embeddings": REPO / "data/cities/mapillary/zurich/embeddings.npz",
            "images": REPO / "data/cities/mapillary/zurich/images"},
    "_hq": {"meta": REPO / "data/mapillary/zurich_altstadt_hq/metadata.jsonl",
            "embeddings": REPO / "data/mapillary/zurich_altstadt_hq/embeddings.npz",
            "images": REPO / "data/mapillary/zurich_altstadt_hq/images"},
}

# (canonical_name, frames_dir_name, raw_prefix_in_existing_files)
# raw_prefix = how the existing scripts named the per-video output files
VIDEOS = [
    # canonical          frames dir                                                 raw_prefix (without leading 'extra_')
    ("zurich_main",      "zurich",                                                  None),
    ("bahnhofstrasse",   "extra_Switzerland_Zurich_Bahnhofstrasse_Walking_tour_Cit",
                         "Switzerland_Zurich_Bahnhofstrasse_Walking_tour_Cit"),
    ("most_famous",      "extra_Walking_Tour_of_Switzerland_s_Most_Famous_City_Zur",
                         "Walking_Tour_of_Switzerland_s_Most_Famous_City_Zur"),
    ("most_elegant",     "extra_ZURICH_Switzerland_The_Most_Elegant_City_in_Europe",
                         "ZURICH_Switzerland_The_Most_Elegant_City_in_Europe"),
    ("looks_perfect",    "extra_Zurich_Switzerland_A_City_That_Looks_Too_Perfect_t",
                         "Zurich_Switzerland_A_City_That_Looks_Too_Perfect_t"),
    ("old_town_limmat",  "extra_Zurich_Switzerland_Old_Town_Limmat_River_Walking_T",
                         "Zurich_Switzerland_Old_Town_Limmat_River_Walking_T"),
    ("hidden_streets",   "extra_Zurich_in_Summer_Hidden_Streets_River_Views_Swiss_",
                         "Zurich_in_Summer_Hidden_Streets_River_Views_Swiss_"),
    ("saturday_morning", "extra_Zurich_looks_STUNNING_on_Saturday_Morning_Switzerl",
                         "Zurich_looks_STUNNING_on_Saturday_Morning_Switzerl"),
]

VIDEO_NAMES = [v[0] for v in VIDEOS]


def paths_for(canonical, variant=""):
    """Return all canonical paths for one video as a dict.

    variant:  ""      → original 5k Mapillary index, outputs without suffix
              "_hq"   → 22.6k HQ Mapillary index, outputs with `_hq` suffix
    """
    rec = next(v for v in VIDEOS if v[0] == canonical)
    _, frames_name, prefix = rec
    pipeline_dir = DATA / f"pipeline{variant}"
    out = pipeline_dir / canonical
    # `prefix` is the full filename prefix (already starts with 'extra_'
    # for the 7 extras, or None for the original zurich_main video).
    s = variant  # suffix on per-video legacy filenames
    return {
        "frames_dir":          DATA / "frames" / frames_name,
        "embeddings":          DATA / "frames" / f"{frames_name}_embeddings.npz",
        "out_dir":             out,
        # per-video phase A outputs (visual_match + refine + ocr + heading)
        "legacy_visual_refined": (DATA / f"extra_{prefix}_frame_gps_refined{s}.jsonl"
                                  if prefix else DATA / f"frame_gps_refined{s}.jsonl"),
        "legacy_ocr_match":      (DATA / f"extra_{prefix}_frame_gps_ocr.jsonl"
                                  if prefix else DATA / "frame_gps_ocr.jsonl"),
        "legacy_heading":        (DATA / f"extra_{prefix}_frame_heading{s}.jsonl"
                                  if prefix else DATA / f"frame_heading{s}.jsonl"),
        # new pipeline outputs (under pipeline_hq/ when variant=_hq)
        "step_07_merged":   out / "step_07_merged.jsonl",
        "step_08_hmm":      out / "step_08_hmm.jsonl",
        "step_09_vlm_poi":  out / "step_09_vlm_poi.jsonl",
        "step_10_verify":   out / "step_10_verify.jsonl",
        "step_11_trusted":  out / "step_11_trusted.jsonl",
    }


def trusted_starts_path(variant=""):
    """Combined trusted_starts file path for a given variant."""
    return DATA / f"frame_starts_trusted_all{variant}.jsonl"


# Shared resources
OSM_GRAPH      = DATA / "osm_walking.pkl"
POI_OSM        = DATA / "landmarks_zurich_osm.json"
VLM_POI_SCAN   = DATA / "_video_poi_multi.jsonl"   # already produced by scan_video_pois_multi.py


# Thresholds for the verifier (step 10) and trusted filter (step 11)
GPS_NEAR_RADIUS_M     = 50      # POI considered "right next to" GPS
GPS_VISIBLE_RADIUS_M  = 200     # POI plausibly within camera frame
VLM_MISMATCH_FAIL_M   = 500     # VLM-seen POI farther than this from GPS → FAIL


# Filter matrix used by step 11. Rows = gps_source, cols = verdict.
# Value = True if the (source, verdict) pair makes the frame trusted.
# Step 7 keeps only visual_high rows (top-5 Mapillary within 50m + compass <30°).
# visual_medium / visual_low / OCR-alone are all dropped earlier.
TRUSTED_MATRIX = {
    "visual_high":   {"PASS_LANDMARK": True,  "PASS_STREET": True,  "INCONCLUSIVE": True,  "FAIL": False},
}


def lang_label():
    """Human-readable summary used by the run_all driver banner."""
    return (f"pipeline: {len(VIDEOS)} videos, "
            f"out → {PIPELINE_OUT.relative_to(REPO)}/")

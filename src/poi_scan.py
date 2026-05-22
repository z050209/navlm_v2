"""POI scan — name the places visible in video frames (Gemini Flash),
then match the names against the OSM POI table.

Replaces the v1 closed 27-candidate Gemma scan. This is **open-set**: the
VLM freely names whatever landmarks / squares / churches / streets /
bridges it sees; we record the raw names and resolve each against the
OSM POI table (`src/pois.py` output) via `resolve_poi()`. Each match is
tagged with its tier — L1 (iconic) / L2 (mid) / L3 (other) — so the
dataset can be filtered to **L1 + L2**.

    python -m src.poi_scan --limit 5       # 5-frame trial first
    python -m src.poi_scan --every-n 10    # every 10th extracted frame
    python -m src.poi_scan                 # all extracted frames

Needs GEMINI_API_KEY (.env). Model: `gemini-2.5-flash` ("gemini fast").
Output: data/cities/zurich/poi_scan.jsonl

Tiering is OSM-tag based (poi_tier, below) — edit the TIER_BY_* maps.
"""

import argparse
import json
import sys
from pathlib import Path

from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config                       # noqa: E402
from src.pois import resolve_poi    # noqa: E402

# ── tiers, by OSM tag — derived from OpenStreetMap's own categories.
#    Edit these maps to re-tier. A specific 'key=value' wins over the
#    key-only fallback. (OSM gives the *category*; true prominence would
#    need another signal — see DEV_MANUAL §2.3.)
TIER_BY_TAG = {
    # L1 — landmark categories people navigate TO
    "tourism=attraction": 1, "tourism=viewpoint": 1, "tourism=zoo": 1,
    "historic=castle": 1, "historic=monument": 1, "historic=memorial": 1,
    "railway=station": 1, "amenity=place_of_worship": 1,
    "amenity=townhall": 1,
    # L2 — supporting POIs
    "tourism=museum": 2, "tourism=gallery": 2, "tourism=artwork": 2,
    "tourism=hotel": 2, "tourism=theme_park": 2,
    "amenity=theatre": 2, "amenity=cinema": 2, "amenity=marketplace": 2,
    "amenity=library": 2, "amenity=university": 2, "amenity=college": 2,
    "leisure=park": 2, "leisure=garden": 2, "leisure=stadium": 2,
    "man_made=bridge": 2, "waterway=river": 2, "natural=water": 2,
    "place=square": 2,
}
# key-only fallback when the exact key=value is not mapped above
TIER_BY_KEY = {"tourism": 2, "historic": 2, "man_made": 2, "waterway": 2,
               "highway": 2, "amenity": 3, "leisure": 3, "natural": 3,
               "railway": 3, "place": 3}

GEMINI_SYS = "You name visible places in street-level photos of Zurich."
GEMINI_PROMPT = (
    "This is a street-level photo taken in central Zurich, Switzerland. "
    "List every named place clearly visible — landmarks, churches, "
    "squares, streets, bridges, the river, the lake, notable buildings.\n"
    "Name each place as it appears on a map / OpenStreetMap — prefer the "
    "official local German name (e.g. 'Hauptbahnhof', 'Zürichsee', "
    "'Grossmünster'). If the place is also widely known by a different "
    "English or common name, add that after a ' | '.\n"
    "One place per line, for example:\n"
    "  Hauptbahnhof | Zurich Main Station\n"
    "  Limmat\n"
    "If nothing nameable is visible, reply with the single word: none"
)


def poi_tier(osm_kind):
    """Tier a POI from its OSM tag 'key=value' (`src/pois.py:osm_kind`):
    1 = iconic landmark, 2 = mid, 3 = other. A specific tag value wins;
    else a key-level fallback; else L3. Pure — unit-tested."""
    if not osm_kind:
        return 3
    if osm_kind in TIER_BY_TAG:
        return TIER_BY_TAG[osm_kind]
    return TIER_BY_KEY.get(osm_kind.split("=", 1)[0], 3)


def parse_names(text):
    """Parse the VLM reply — one place per line, name variants split on
    '|'. Returns a list of variant-lists, e.g.
    `[['Hauptbahnhof', 'Zurich Main Station'], ['Limmat']]`. Pure."""
    out = []
    for line in (text or "").splitlines():
        line = line.strip().lstrip("-*•").strip()
        if not line or line.lower() == "none":
            continue
        variants = [v.strip() for v in line.split("|") if v.strip()]
        if variants:
            out.append(variants)
    return out


def match_names(places, osm_pois):
    """Resolve each place against the OSM POI table.

    `places`: a list of variant-lists (from `parse_names`). For each
    place the first variant that resolves wins — so a miss on Gemini's
    English name can still hit on the German one. Returns
    (matched, unmatched): matched is `[{variants, matched_name,
    osm_name, osm_kind, kind_label, tier}]`, unmatched is
    `[variants, ...]`. Pure — unit-tested.
    """
    matched, unmatched = [], []
    for variants in places:
        hit = used = None
        for v in variants:
            hit = resolve_poi(v, osm_pois)
            if hit:
                used = v
                break
        if hit:
            kind = hit.get("osm_kind", "")
            matched.append({
                "variants": variants, "matched_name": used,
                "osm_name": hit["name"],
                "osm_kind": kind,
                "kind_label": hit.get("kind_label", ""),
                "tier": poi_tier(kind),
            })
        else:
            unmatched.append(variants)
    return matched, unmatched


def load_osm_pois():
    """Load the OSM POI table (src/pois.py output). Exits if absent."""
    p = config.CITY_DIR / "pois.json"
    if not p.exists():
        sys.exit(f"OSM POI table not found: {p}\n"
                 f"run `python -m src.pois` first.")
    return json.loads(p.read_text(encoding="utf-8"))


def _downscaled(image_path, max_px=config.POI_SCAN_MAX_PX):
    """Write a downscaled (<= max_px) JPEG copy of the frame to a temp
    file; return its path. The scan only needs recognition, not 4K
    detail — this roughly halves the Gemini image-token cost."""
    import tempfile
    from PIL import Image
    img = Image.open(image_path).convert("RGB")
    img.thumbnail((max_px, max_px))
    tmp = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
    tmp.close()
    img.save(tmp.name, "JPEG", quality=85)
    return tmp.name


def scan_frame(image_path, retries=3):
    """One Gemini-Flash open-set naming call → variant-lists. The frame
    is downscaled first (cost). Retries with backoff on transient API
    errors (e.g. 503/429)."""
    import os
    import time
    sys.path.insert(0, str(config.REPO_ROOT / "reference" / "toolbox"))
    from synth.backends import call_gemini
    small = _downscaled(image_path)
    try:
        for attempt in range(retries):
            try:
                resp = call_gemini(small, GEMINI_SYS, GEMINI_PROMPT,
                                   model=config.GEMINI_SCAN)
                return parse_names(resp)
            except Exception:
                if attempt == retries - 1:
                    raise
                time.sleep(3 * (attempt + 1))    # 3 s, 6 s backoff
    finally:
        try:
            os.unlink(small)
        except OSError:
            pass
    return []


def discover_frames(every_n=1):
    """Extracted frames -> [(video, frame_id, path), ...]."""
    out = []
    if not config.FRAMES_DIR.exists():
        return out
    for vdir in sorted(config.FRAMES_DIR.iterdir()):
        if not vdir.is_dir() or vdir.name.endswith("_dense"):
            continue
        for f in sorted(vdir.glob("frame_*.jpg"))[::every_n]:
            out.append((vdir.name, f.stem, f))
    return out


def main():
    ap = argparse.ArgumentParser(description="POI scan — Gemini Flash")
    ap.add_argument("--every-n", type=int, default=10,
                    help="scan every Nth extracted frame")
    ap.add_argument("--limit", type=int, default=0,
                    help="cap total frames (trial runs)")
    args = ap.parse_args()

    osm_pois = load_osm_pois()
    frames = discover_frames(args.every_n)
    if args.limit:
        frames = frames[:args.limit]
    if not frames:
        sys.exit(f"no extracted frames under {config.FRAMES_DIR} — "
                 f"run `python -m src.extract_frames` first")

    out_path = config.CITY_DIR / "poi_scan.jsonl"
    print(f"[poi_scan] {len(frames)} frames · model={config.GEMINI_SCAN} · "
          f"OSM table={len(osm_pois)} POIs -> {out_path}")

    n_l12 = 0
    with out_path.open("w", encoding="utf-8") as fout:
        for video, frame_id, path in tqdm(frames, desc="[poi_scan]",
                                          unit="frame"):
            try:
                raw = scan_frame(path)
            except Exception as e:
                raw = []
                tqdm.write(f"  {frame_id}: {type(e).__name__}: {e}")
            matched, unmatched = match_names(raw, osm_pois)
            n_l12 += sum(1 for m in matched if m["tier"] in (1, 2))
            fout.write(json.dumps({
                "video": video, "frame_id": frame_id,
                "places": raw, "matched": matched, "unmatched": unmatched,
            }, ensure_ascii=False) + "\n")
    print(f"[poi_scan] done — {n_l12} L1/L2 POI sightings recorded.")


if __name__ == "__main__":
    main()

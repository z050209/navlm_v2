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

The L1 / L2 keyword sets below are meant to be edited — tune the tiers.
"""

import argparse
import json
import sys
from pathlib import Path

from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config                       # noqa: E402
from src.pois import resolve_poi    # noqa: E402

# ── tier keyword sets — EDIT to tune which POIs count as L1 / L2 ──────
L1_KEYWORDS = {
    "hauptbahnhof", "grossmünster", "grossmunster", "fraumünster",
    "fraumunster", "paradeplatz", "bahnhofstrasse", "lindenhof",
    "bellevue", "opernhaus", "kunsthaus", "st. peter", "limmat",
    "lake zurich", "zürichsee", "niederdorf", "rathaus", "landesmuseum",
    "sechseläutenplatz", "münsterhof", "polyterrasse", "eth", "jelmoli",
    "globus", "bürkliplatz", "stadthaus", "quaibrücke", "münsterbrücke",
}
L2_KEYWORDS = {
    "museum", "kirche", "church", "platz", "square", "brücke", "bridge",
    "park", "garten", "garden", "quai", "promenade", "gasse", "strasse",
    "universität", "university", "theater", "theatre", "hotel", "markt",
    "tower", "turm", "monument", "fountain", "brunnen", "hof",
}

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


def poi_tier(name, kind_label=""):
    """Classify a POI into tier 1 (iconic) / 2 (mid) / 3 (other). Pure."""
    t = f"{name} {kind_label}".lower()
    if any(k in t for k in L1_KEYWORDS):
        return 1
    if any(k in t for k in L2_KEYWORDS):
        return 2
    return 3


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
    (matched, unmatched): matched is
    `[{variants, matched_name, osm_name, tier}]`, unmatched is
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
            matched.append({
                "variants": variants, "matched_name": used,
                "osm_name": hit["name"],
                "tier": poi_tier(hit["name"], hit.get("kind_label", "")),
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


def scan_frame(image_path):
    """One Gemini-Flash open-set naming call. Returns the raw name list."""
    sys.path.insert(0, str(config.REPO_ROOT / "reference" / "toolbox"))
    from synth.backends import call_gemini
    resp = call_gemini(str(image_path), GEMINI_SYS, GEMINI_PROMPT,
                       model=config.GEMINI_SCAN)
    return parse_names(resp)


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

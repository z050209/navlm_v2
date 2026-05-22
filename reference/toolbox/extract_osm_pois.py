"""Auto-extract Zurich POIs from OSM via Overpass API.

Outputs a JSON file: dict mapping name → {lat, lon, aliases, kinds}.
The aliases list is consumed by landmark_match.py for OCR matching, and the
(name, lat, lon) tuples are consumed by way_planner.py / synth_unified.py.

Run once per city:
    python toolbox/extract_osm_pois.py \\
        --bbox 8.520,47.360,8.570,47.395 \\
        --out  data/cities/zurich/landmarks_zurich_osm.json
"""

import argparse
import json
from pathlib import Path

import osmnx as ox

# Tag-value allowlist. Each entry: top-level tag → set of acceptable values
# (or True for "any value"). Stricter than v1 — drops decorative items,
# tiny memorials, and amenity=fountain (which OSM uses for both plaza
# fountains AND wall-mounted drinking-water taps, which are useless as
# navigation landmarks).
ACCEPT_TAG_GROUPS = [
    {"tourism": ["museum", "attraction", "viewpoint", "hotel", "gallery",
                 "theme_park", "zoo", "aquarium"]},
    {"historic": ["castle", "monument", "ruins", "archaeological_site",
                  "fort", "tower", "manor"]},
    {"amenity": ["theatre", "museum", "library", "cinema", "place_of_worship",
                 "townhall", "marketplace", "university", "college"]},
    {"public_transport": ["station"]},
    {"railway": ["station"]},
    {"leisure": ["park", "garden", "stadium"]},
    {"place": ["square", "neighbourhood", "suburb"]},
]

# Drop POIs with these meaningless names
NAME_BLOCKLIST = {"unknown", "unnamed", "n/a", ""}

# Drop POIs whose names are obviously descriptive sentences, not landmarks.
# (e.g. "Wandbrunnen mit wasserspeiender Bronzemaske" → 41 chars → dropped)
MAX_NAME_LEN = 30
MIN_NAME_LEN = 3


# Map (top-level tag, value) → human-readable "what this is" label.
# Used so the synth prompt can say "Hauptbahnhof (a train station)".
KIND_LABELS = {
    ("tourism", "museum"): "a museum",
    ("tourism", "attraction"): "a tourist attraction",
    ("tourism", "viewpoint"): "a viewpoint",
    ("tourism", "hotel"): "a hotel",
    ("tourism", "gallery"): "an art gallery",
    ("tourism", "theme_park"): "a theme park",
    ("tourism", "zoo"): "a zoo",
    ("tourism", "aquarium"): "an aquarium",

    ("historic", "castle"): "a historic castle",
    ("historic", "monument"): "a monument",
    ("historic", "ruins"): "historic ruins",
    ("historic", "archaeological_site"): "an archaeological site",
    ("historic", "fort"): "a fort",
    ("historic", "tower"): "a historic tower",
    ("historic", "manor"): "a manor house",

    ("amenity", "theatre"): "a theatre",
    ("amenity", "museum"): "a museum",
    ("amenity", "library"): "a library",
    ("amenity", "cinema"): "a cinema",
    ("amenity", "place_of_worship"): "a church",
    ("amenity", "townhall"): "a town hall",
    ("amenity", "marketplace"): "a marketplace",
    ("amenity", "university"): "a university",
    ("amenity", "college"): "a college",

    ("public_transport", "station"): "a public transport station",
    ("railway", "station"): "a train station",

    ("leisure", "park"): "a park",
    ("leisure", "garden"): "a garden",
    ("leisure", "stadium"): "a stadium",

    ("place", "square"): "a public square",
    ("place", "neighbourhood"): "a neighbourhood",
    ("place", "suburb"): "a suburb",
}


def kind_label_for(tag, value):
    """Return human label, or '' if not in our table."""
    return KIND_LABELS.get((tag, value), "")


def normalize_aliases(name, max_aliases=4):
    out = {name.strip(), name.strip().lower()}
    repl = str.maketrans({"ä":"a","ö":"o","ü":"u","ß":"ss",
                          "Ä":"A","Ö":"O","Ü":"U"})
    out.add(name.translate(repl))
    out.add(name.translate(repl).lower())
    out = sorted(a for a in out if a and len(a) >= 2)
    return out[:max_aliases]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bbox", default="8.520,47.360,8.570,47.395",
                    help="W,S,E,N comma-separated")
    ap.add_argument("--out", default="data/cities/zurich/landmarks_zurich_osm.json")
    args = ap.parse_args()

    bbox = tuple(map(float, args.bbox.split(",")))

    pois = {}  # name → {lat, lon, aliases, kinds}
    for tags in ACCEPT_TAG_GROUPS:
        try:
            gdf = ox.features_from_bbox(bbox=bbox, tags=tags)
        except Exception as e:
            print(f"  skip {tags}: {type(e).__name__}: {str(e)[:80]}")
            continue
        if "name" not in gdf.columns:
            continue
        gdf = gdf[gdf["name"].notna()]
        for _, row in gdf.iterrows():
            name = str(row["name"]).strip()
            if name.lower() in NAME_BLOCKLIST:
                continue
            if not (MIN_NAME_LEN <= len(name) <= MAX_NAME_LEN):
                continue
            # Prefer names that look like proper nouns (start with capital,
            # contain at least one alphabetic char). Reject all-lowercase
            # blobs, all-digits, etc.
            if not (name[0].isupper() and any(c.isalpha() for c in name)):
                continue
            try:
                pt = row.geometry.centroid
                lat, lon = float(pt.y), float(pt.x)
            except Exception:
                continue
            kinds = list(tags.keys())
            # Resolve a human-readable kind_label (e.g. "a museum") from the
            # actual OSM tag values on this row.
            kind_label = ""
            for tag in kinds:
                v = row.get(tag)
                if isinstance(v, str):
                    label = kind_label_for(tag, v.strip())
                    if label:
                        kind_label = label
                        break
            if name in pois:
                # merge tag kinds, prefer most-specific kind_label
                pois[name]["kinds"] = sorted(set(pois[name]["kinds"]) | set(kinds))
                if kind_label and not pois[name].get("kind_label"):
                    pois[name]["kind_label"] = kind_label
            else:
                pois[name] = {
                    "lat": round(lat, 6),
                    "lon": round(lon, 6),
                    "aliases": normalize_aliases(name),
                    "kinds": sorted(set(kinds)),
                    "kind_label": kind_label,
                }

    print(f"[poi] {len(pois)} unique named POIs")

    # Category breakdown
    by_kind = {}
    for n, p in pois.items():
        for k in p["kinds"]:
            by_kind[k] = by_kind.get(k, 0) + 1
    for k, c in sorted(by_kind.items(), key=lambda x: -x[1]):
        print(f"  {k:20s}  {c}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(pois, f, ensure_ascii=False, indent=1)
    print(f"[poi] wrote → {args.out}")


if __name__ == "__main__":
    main()

"""POI table extraction from OpenStreetMap (DEV_MANUAL §2.3).

Builds the single v2 POI table — point landmarks + way/area features
(streets, river, lake, bridges) — with real geometry, via osmnx
Overpass over the project bbox.

  clean_name()  — validate / normalise an OSM name (pure, unit-tested)
  extract()     — the osmnx query; writes data/cities/zurich/pois.json

    python -m src.pois            # extract the POI table
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config  # noqa: E402

NAME_BLOCKLIST = {"", "unknown", "unnamed", "n/a", "none"}
MIN_NAME_LEN, MAX_NAME_LEN = 3, 40

# point landmarks
POINT_TAGS = {
    "tourism": True, "historic": True, "railway": "station",
    "amenity": ["theatre", "museum", "place_of_worship", "townhall",
                "marketplace", "library"],
}
# way / area features — kept WITH geometry
WAY_TAGS = {
    "highway": ["primary", "secondary", "tertiary", "residential",
                "pedestrian", "living_street"],
    "waterway": "river", "natural": "water", "man_made": "bridge",
}


def clean_name(name):
    """Return a normalised POI name, or None if it should be dropped.
    Drops blocklisted / too-short / too-long / non-capitalised names."""
    if not name or not isinstance(name, str):
        return None
    n = " ".join(name.split())
    if n.lower() in NAME_BLOCKLIST:
        return None
    if not (MIN_NAME_LEN <= len(n) <= MAX_NAME_LEN):
        return None
    if not n[0].isupper():
        return None
    return n


def extract(bbox=config.POI_BBOX):
    """Query OSM for point + way/area POIs; write pois.json. Needs osmnx."""
    import osmnx as ox

    w, s, e, n = bbox
    rows = []
    for tags, kind_group in ((POINT_TAGS, "point"), (WAY_TAGS, "way")):
        gdf = ox.features_from_bbox((w, s, e, n), tags=tags)
        for _, row in gdf.iterrows():
            name = clean_name(row.get("name"))
            if not name:
                continue
            geom = row.geometry
            rows.append({
                "name": name,
                "kind_group": kind_group,
                "lat": geom.centroid.y, "lon": geom.centroid.x,
                "geometry": geom.wkt,
            })

    # dedupe by name (keep first)
    seen, uniq = set(), []
    for r in rows:
        if r["name"] not in seen:
            seen.add(r["name"])
            uniq.append(r)

    out = config.CITY_DIR / "pois.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(uniq, ensure_ascii=False, indent=1),
                   encoding="utf-8")
    print(f"[pois] {len(uniq)} POIs -> {out}")
    return uniq


def main():
    extract()


if __name__ == "__main__":
    main()

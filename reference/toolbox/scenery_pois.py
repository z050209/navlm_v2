"""Hand-written supplementary table of scenery / area / way POIs that
OSM tags as ways or polygons rather than point nodes.

Each entry: name → {center_lat, center_lon, kind, kind_label, radius_m,
                     aliases}

Used by:
  - POI proximity check (use entry-specific radius, NOT global 50m)
  - way_planner (target a representative centre point)
  - synth_unified prompts (kind_label gives natural-language description)
  - viewer maps (different style for scenery POIs)
"""

SCENERY_POIS = {
    "Bahnhofstrasse": {
        "lat": 47.37367, "lon": 8.53924,
        "kind": "street", "kind_label": "the main shopping street with tram tracks",
        "radius_m": 300, "aliases": ["bahnhofstrasse", "bahnhofstraße"]},
    "Limmatquai": {
        "lat": 47.37200, "lon": 8.54330,
        "kind": "promenade", "kind_label": "a riverside promenade with arcaded buildings",
        "radius_m": 300, "aliases": ["limmatquai"]},
    "Niederdorfstrasse": {
        "lat": 47.37318, "lon": 8.54417,
        "kind": "street", "kind_label": "an old-town pedestrian shopping lane",
        "radius_m": 200, "aliases": ["niederdorfstrasse", "niederdorf"]},
    "Rennweg": {
        "lat": 47.37326, "lon": 8.54000,
        "kind": "street", "kind_label": "a downhill medieval shopping street",
        "radius_m": 180, "aliases": ["rennweg"]},
    "Storchengasse": {
        "lat": 47.37072, "lon": 8.54066,
        "kind": "lane", "kind_label": "a narrow medieval lane with arcades",
        "radius_m": 100, "aliases": ["storchengasse"]},
    "Münstergasse": {
        "lat": 47.37150, "lon": 8.54330,
        "kind": "lane", "kind_label": "a cafe-lined lane near the cathedral",
        "radius_m": 100, "aliases": ["münstergasse", "munstergasse", "muenstergasse"]},

    "Limmat river": {
        "lat": 47.37100, "lon": 8.54200,
        "kind": "river", "kind_label": "a river running through the old town",
        "radius_m": 500, "aliases": ["limmat", "limmat river"]},
    "Lake Zurich": {
        "lat": 47.36500, "lon": 8.54500,
        "kind": "lake", "kind_label": "the northern shore of Lake Zurich",
        "radius_m": 600, "aliases": ["lake zurich", "zürichsee", "zuerichsee"]},

    "Hauptbahnhof": {
        "lat": 47.37802, "lon": 8.54023,
        "kind": "station", "kind_label": "the main railway station with its arched stone facade",
        "radius_m": 200, "aliases": ["hauptbahnhof", "main station", "zürich hb"]},

    "Münsterbrücke": {
        "lat": 47.36970, "lon": 8.54200,
        "kind": "bridge", "kind_label": "a stone bridge crossing the Limmat at the Grossmünster",
        "radius_m": 80, "aliases": ["münsterbrücke", "muensterbruecke", "munsterbrucke"]},
    "Quaibrücke": {
        "lat": 47.36593, "lon": 8.54367,
        "kind": "bridge", "kind_label": "the lakeside bridge linking Bellevue and Bürkliplatz",
        "radius_m": 100, "aliases": ["quaibrücke", "quaibruecke"]},
    "Rathausbrücke": {
        "lat": 47.37117, "lon": 8.54158,
        "kind": "bridge", "kind_label": "the bridge in front of the city hall",
        "radius_m": 60, "aliases": ["rathausbrücke", "rathausbruecke"]},
}


def in_scenery_radius(gps_lat, gps_lon):
    """Return list of (name, distance_m) for scenery POIs whose
    custom radius contains (gps_lat, gps_lon). Sorted closest first."""
    import math
    out = []
    for name, info in SCENERY_POIS.items():
        # haversine
        R = 6371000.0
        p1 = math.radians(gps_lat); p2 = math.radians(info["lat"])
        dphi = math.radians(info["lat"] - gps_lat)
        dlam = math.radians(info["lon"] - gps_lon)
        a = math.sin(dphi/2)**2 + math.cos(p1)*math.cos(p2)*math.sin(dlam/2)**2
        d = 2 * R * math.asin(math.sqrt(a))
        if d <= info["radius_m"]:
            out.append((name, d))
    out.sort(key=lambda x: x[1])
    return out

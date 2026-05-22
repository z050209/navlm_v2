"""Hard-coded Zurich landmark names + lat/lon.

Used by landmark_match.py to resolve OCR-detected sign text to exact GPS.
Coordinates rounded to ~10m precision. Covers the ~2km core of the walking
tour we annotated (Hauptbahnhof → Altstadt → Grossmünster → Bellevue and
back). Extend this table with more entries if the OCR signal suggests
areas we haven't covered.

Each entry: landmark-name → (lat, lon, [alias strings...])
Aliases are substrings we look for in OCR text (case-insensitive).
"""

# (lat, lon, aliases)  — lat/lon are manually verified from OSM
ZURICH_LANDMARKS = {
    # Train station + square
    "Hauptbahnhof":        (47.37802, 8.54023, ["hauptbahnhof", "bahnhofplatz", "main station"]),
    "Bahnhofstrasse":      (47.37367, 8.53924, ["bahnhofstrasse", "bahnhofstraße"]),

    # Old town churches
    "Grossmünster":        (47.37018, 8.54425, ["grossmünster", "grossmunster", "grossmuenster"]),
    "Fraumünster":         (47.37005, 8.54148, ["fraumünster", "fraumunster", "fraumuenster"]),
    "St. Peter":           (47.37154, 8.54126, ["st. peter", "st peter", "peterskirche"]),

    # Old town squares / streets
    "Lindenhof":           (47.37280, 8.54149, ["lindenhof"]),
    "Paradeplatz":         (47.36953, 8.53866, ["paradeplatz"]),
    "Münsterhof":          (47.37072, 8.54128, ["münsterhof", "muensterhof", "munsterhof"]),
    "Rennweg":             (47.37326, 8.54000, ["rennweg"]),
    "Storchengasse":       (47.37072, 8.54066, ["storchengasse"]),
    "Niederdorfstrasse":   (47.37318, 8.54417, ["niederdorfstrasse", "niederdorf"]),
    "Limmatquai":          (47.37200, 8.54330, ["limmatquai"]),
    "Münstergasse":        (47.37150, 8.54330, ["münstergasse", "muenstergasse", "munstergasse"]),

    # River + bridges
    "Limmat":              (47.37100, 8.54200, ["limmat"]),
    "Münsterbrücke":       (47.36970, 8.54200, ["münsterbrücke", "muensterbruecke", "munsterbrucke"]),
    "Quaibrücke":          (47.36593, 8.54367, ["quaibrücke", "quaibruecke"]),

    # Bellevue / lake area
    "Bellevue":            (47.36684, 8.54517, ["bellevue", "bellevueplatz"]),
    "Opernhaus":           (47.36548, 8.54683, ["opernhaus", "opera house"]),
    "Sechseläutenplatz":   (47.36620, 8.54615, ["sechseläutenplatz", "sechselaeutenplatz"]),

    # Museums / cultural
    "Helmhaus":            (47.37052, 8.54303, ["helmhaus"]),
    "Kunsthaus":           (47.37037, 8.54834, ["kunsthaus"]),

    # Transport / banks often visible as signs
    "UBS Paradeplatz":     (47.36942, 8.53837, ["ubs"]),
    "Credit Suisse":       (47.36963, 8.53878, ["credit suisse"]),
    "Zürich Airport":      (47.45008, 8.56194, ["zurich airport", "zürich airport"]),

    # Shops seen on Bahnhofstrasse (chains → coarse GPS on Bahnhofstrasse)
    "Globus":              (47.37563, 8.54058, ["globus"]),
    "Jelmoli":             (47.37480, 8.53846, ["jelmoli"]),

    # Watch shops (Paradeplatz / Bahnhofstrasse area)
    "Tudor":               (47.37150, 8.53920, ["tudor"]),
    "Tissot":              (47.37050, 8.53890, ["tissot"]),
    "Omega":               (47.37220, 8.53940, ["omega"]),
    "Breitling":           (47.37120, 8.53910, ["breitling"]),
}


def build_alias_index():
    """Return dict: alias_lowercase → (canonical_name, lat, lon)."""
    idx = {}
    for name, (lat, lon, aliases) in ZURICH_LANDMARKS.items():
        for a in aliases:
            idx[a.lower()] = (name, lat, lon)
    return idx

"""Unit tests for src/pois.py — OSM name cleaning."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src import pois  # noqa: E402


def test_clean_name_keeps_valid():
    assert pois.clean_name("Grossmünster") == "Grossmünster"
    assert pois.clean_name("  Bahnhofstrasse  ") == "Bahnhofstrasse"


def test_clean_name_drops_invalid():
    assert pois.clean_name("") is None
    assert pois.clean_name(None) is None
    assert pois.clean_name("unknown") is None          # blocklisted
    assert pois.clean_name("ab") is None               # too short
    assert pois.clean_name("x" * 50) is None           # too long
    assert pois.clean_name("lowercase start") is None  # not capitalised


def test_collect_aliases_includes_en_de_not_zh():
    tags = {"alt_name": "ETH Zürich", "short_name": "ETH",
            "name:de": "Eidgenössische Technische Hochschule",
            "name:zh": "苏黎世联邦理工学院"}
    al = pois.collect_aliases("Polyterrasse", tags)
    assert "Polyterrasse" in al and "ETH" in al and "ETH Zürich" in al
    assert "苏黎世联邦理工学院" not in al          # Chinese not collected


def test_collect_aliases_dedupes_and_splits_semicolons():
    al = pois.collect_aliases("A", {"alt_name": "B;A;b"})
    lows = [a.lower() for a in al]
    assert len(lows) == len(set(lows))             # case-insensitive unique


def test_resolve_poi_exact_and_alias():
    pp = [{"name": "Polyterrasse", "aliases": ["ETH", "ETH Zürich"]},
          {"name": "Grossmünster", "aliases": []}]
    assert pois.resolve_poi("Polyterrasse", pp)["name"] == "Polyterrasse"
    assert pois.resolve_poi("eth", pp)["name"] == "Polyterrasse"   # alias
    assert pois.resolve_poi("Grossmünster", pp)["name"] == "Grossmünster"


def test_resolve_poi_substring_and_miss():
    pp = [{"name": "Bahnhofstrasse", "aliases": []}]
    assert pois.resolve_poi("bahnhofstrasse street", pp)["name"] == "Bahnhofstrasse"
    assert pois.resolve_poi("Nowhere at all", pp) is None
    assert pois.resolve_poi("", pp) is None


def test_resolve_poi_diacritic_fold():
    pp = [{"name": "Grossmünster", "aliases": []}]
    assert pois.resolve_poi("Grossmunster", pp)["name"] == "Grossmünster"
    assert pois.resolve_poi("GROSSMUNSTER", pp)["name"] == "Grossmünster"


def test_fold_strips_diacritics():
    assert pois.fold("Grossmünster") == "grossmunster"
    assert pois.fold("Zürichsee") == "zurichsee"


def test_osm_kind():
    assert pois.osm_kind({"tourism": "museum"}) == "tourism=museum"
    assert pois.osm_kind({"amenity": "place_of_worship",
                          "name": "X"}) == "amenity=place_of_worship"
    assert pois.osm_kind({"highway": "primary"}) == "highway=primary"
    assert pois.osm_kind({"name": "X"}) == ""              # no kind tag
    # priority order: tourism before highway
    assert pois.osm_kind({"tourism": "attraction",
                          "highway": "primary"}) == "tourism=attraction"

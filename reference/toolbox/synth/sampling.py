"""POI tier classification + destination / location sampling.

Tier classification determines what fraction of training samples target
iconic landmarks (most touristically relevant) vs obscure ones.
Reverse-lookup converts a raw GPS to a human-readable USER_LOCATION for
the synth prompt.
"""

import sys
from pathlib import Path

# Allow imports either as a package or via sys.path
_PARENT = str(Path(__file__).resolve().parent.parent)
if _PARENT not in sys.path:
    sys.path.insert(0, _PARENT)
from synth_utils import haversine_m  # noqa: E402
try:
    from scenery_pois import SCENERY_POIS, in_scenery_radius  # noqa: E402
except ImportError:
    SCENERY_POIS, in_scenery_radius = {}, lambda *a: []

from .prompts import TIER_KEYWORDS


def poi_tier(name):
    """Return 1, 2, or 3 based on landmark importance."""
    n = name.lower()
    for tier in (1, 2):
        if any(kw in n for kw in TIER_KEYWORDS[tier]):
            return tier
    return 3


def sample_destinations_weighted(start_gps, pois, n, min_m, max_m, rng,
                                  tier_quota=(0.7, 0.25, 0.05)):
    """Tier-weighted destination sampling.

    Defaults: 70% of selected destinations are tier-1 (iconic), 25% tier-2,
    5% tier-3. Falls back to filling from any tier if a quota can't be met.
    """
    cands = []
    for name, p in pois.items():
        d = haversine_m(start_gps[0], start_gps[1], p["lat"], p["lon"])
        if min_m <= d <= max_m:
            cands.append((name, p, d))
    if not cands:
        return []

    by_tier = {1: [], 2: [], 3: []}
    for name, p, d in cands:
        by_tier[poi_tier(name)].append((name, p, d))

    quotas = [int(round(q * n)) for q in tier_quota]
    quotas[2] = max(0, n - quotas[0] - quotas[1])

    chosen = []
    for tier_idx, tier in enumerate((1, 2, 3)):
        pool = by_tier[tier][:]
        rng.shuffle(pool)
        chosen.extend(pool[: quotas[tier_idx]])

    if len(chosen) < n:
        remaining = [c for c in cands if c not in chosen]
        rng.shuffle(remaining)
        chosen.extend(remaining[: n - len(chosen)])

    return [(name, p["lat"], p["lon"], d) for name, p, d in chosen[:n]]


def reverse_lookup_location(gps, pois, graph=None,
                             max_poi_m=80, max_street_m=30):
    """Map a raw GPS to a human-readable USER_LOCATION string.

    Resolution order (most specific wins):
      1. nearest OSM POI within `max_poi_m`
      2. nearest scenery POI (street / river / lake / building / bridge)
         within ITS OWN radius — usually larger than 80m
      3. nearest named street from OSM walking graph within `max_street_m`
      4. literal "an unnamed street"
    """
    # Tier 1: OSM point POIs
    best = None
    for name, p in pois.items():
        d = haversine_m(gps[0], gps[1], p["lat"], p["lon"])
        if best is None or d < best[1]:
            best = (name, d)
    if best and best[1] <= max_poi_m:
        return best[0]

    # Tier 2: scenery POIs (use each entry's own radius)
    sc = in_scenery_radius(gps[0], gps[1])
    if sc:
        return sc[0][0]

    # Tier 3: nearest named street from OSM walking graph
    if graph is not None:
        try:
            import osmnx as ox
            u, v, _key = ox.distance.nearest_edges(graph, gps[1], gps[0])
            edge_data = min(graph[u][v].values(),
                            key=lambda d: d.get("length", float("inf")))
            name = edge_data.get("name")
            if name:
                if isinstance(name, list):
                    name = name[0]
                eu_lat, eu_lon = graph.nodes[u]["y"], graph.nodes[u]["x"]
                ev_lat, ev_lon = graph.nodes[v]["y"], graph.nodes[v]["x"]
                d_to_edge = min(
                    haversine_m(gps[0], gps[1], eu_lat, eu_lon),
                    haversine_m(gps[0], gps[1], ev_lat, ev_lon),
                )
                if d_to_edge <= max_street_m:
                    return f"{name}"
        except Exception:
            pass

    return "an unnamed street"

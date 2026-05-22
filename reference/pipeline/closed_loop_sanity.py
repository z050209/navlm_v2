"""Sanity check the closed-loop heading-action math on existing synth data.

For each sample we have:
  start_gps  (user_gps)
  heading_gt (user_heading)
  destination (name → GPS)
  first_action (planner output, computed using heading_gt)

We compute:
  1. Re-plan the route to recover the FIRST WAYPOINT (= second OSM node on
     the planned path), since we need its absolute bearing.
  2. h_walks = heading_gt + ACTION_DELTA[first_action]    (mod 360)
  3. θ      = bearing(start_gps, first_waypoint)
  4. δ      = angle_diff(h_walks, θ)

Because the planner DESIGNED first_action to point toward the first
waypoint (using heading_gt), δ should be small (<30°) for almost all
samples — this validates the forward math. Significant tails would mean
either the discretization (left=−90 etc.) or the planner cleanup is too
lossy.
"""

import json
import math
import sys
from collections import Counter
from pathlib import Path

import networkx as nx

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "toolbox"))
from way_planner import WayPlanner  # noqa: E402
from synth_utils import bearing_deg  # noqa: E402

ACTION_DELTA = {
    "continue ahead": 0.0,
    "turn left":      -90.0,
    "turn right":      90.0,
    "turn around":    180.0,
}


def angle_diff(a, b):
    """Smallest signed difference, |result| <= 180."""
    return ((a - b + 540) % 360) - 180


def first_waypoint(planner, start_gps, dest_gps, user_heading):
    """Recover the FIRST WAYPOINT on the planned path.

    way_planner returns cleaned-up steps but discards intermediate node
    coordinates. We re-run shortest_path here and take node[1] as the
    first waypoint (the one immediately after the start).
    """
    import osmnx as ox
    G = planner.G
    s_lat, s_lon = start_gps
    d_lat, d_lon = dest_gps
    sn = ox.distance.nearest_nodes(G, s_lon, s_lat)
    dn = ox.distance.nearest_nodes(G, d_lon, d_lat)
    try:
        route = nx.shortest_path(G, sn, dn, weight="length")
    except nx.NetworkXNoPath:
        return None
    if len(route) < 2:
        return None
    # node[1] = first OSM node after start
    nxt = route[1]
    return G.nodes[nxt]["y"], G.nodes[nxt]["x"]


def main():
    synth_path = Path("/pub/evaluation_group/ning/test/navlm/data/cities/zurich/synth_unified.jsonl")
    pois_path  = Path("/pub/evaluation_group/ning/test/navlm/data/cities/zurich/landmarks_zurich_osm.json")
    graph_path = Path("/pub/evaluation_group/ning/test/navlm/data/cities/zurich/osm_walking.pkl")

    with open(pois_path) as f:
        pois = json.load(f)
    # also load scenery POIs
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "toolbox"))
    try:
        from scenery_pois import SCENERY_POIS
        for n, info in SCENERY_POIS.items():
            if n not in pois:
                pois[n] = {"lat": info["lat"], "lon": info["lon"]}
    except ImportError:
        pass

    planner = WayPlanner(str(graph_path))

    samples = []
    for ln in open(synth_path):
        try:
            r = json.loads(ln)
            samples.append(r)
        except Exception:
            pass
    print(f"loaded {len(samples)} synth samples")

    # Take a 200-sample stratified subset
    import random
    rng = random.Random(0)
    rng.shuffle(samples)
    samples = samples[:200]

    results = []
    skipped = 0
    for s in samples:
        m = s["_meta"]
        start = m["user_gps"]
        heading = m["user_heading"]
        action  = m["first_action"]
        dest    = m["destination"]
        if heading is None or action not in ACTION_DELTA:
            skipped += 1; continue
        if dest not in pois:
            skipped += 1; continue
        d_lat, d_lon = pois[dest]["lat"], pois[dest]["lon"]

        wp = first_waypoint(planner, start, (d_lat, d_lon), heading)
        if wp is None:
            skipped += 1; continue

        theta = bearing_deg(start[0], start[1], wp[0], wp[1])
        h_walks = (heading + ACTION_DELTA[action]) % 360
        delta = abs(angle_diff(h_walks, theta))
        results.append({
            "frame":   m["start_frame"],
            "dest":    dest,
            "heading_gt": round(heading, 1),
            "action":     action,
            "h_walks":    round(h_walks, 1),
            "theta":      round(theta, 1),
            "delta":      round(delta, 1),
        })

    print(f"\nevaluated {len(results)} samples, skipped {skipped}")
    if not results:
        return

    # Distribution
    bins = [(0, 10), (10, 20), (20, 30), (30, 45), (45, 60),
            (60, 90), (90, 180)]
    bucket = Counter()
    for r in results:
        d = r["delta"]
        for lo, hi in bins:
            if lo <= d < hi:
                bucket[(lo, hi)] += 1
                break
    print(f"\nclosed-loop angular error δ distribution:")
    for lo, hi in bins:
        n = bucket[(lo, hi)]
        bar = "█" * int(40 * n / len(results))
        print(f"  [{lo:>3}–{hi:>3}°)  {n:>4d}  {bar}")

    n_pass  = sum(1 for r in results if r["delta"] < 30)
    n_warn  = sum(1 for r in results if 30 <= r["delta"] <= 60)
    n_fail  = sum(1 for r in results if r["delta"] > 60)
    print(f"\n  PASS  (<30°)   {n_pass}/{len(results)}  {100*n_pass/len(results):.1f}%")
    print(f"  WARN  (30-60°) {n_warn}/{len(results)}  {100*n_warn/len(results):.1f}%")
    print(f"  FAIL  (>60°)   {n_fail}/{len(results)}  {100*n_fail/len(results):.1f}%")

    # Show a few examples from each bucket
    print(f"\n=== sample PASS (small δ) ===")
    for r in sorted(results, key=lambda x: x["delta"])[:3]:
        print(f"  {r}")
    print(f"\n=== sample FAIL (large δ) ===")
    for r in sorted(results, key=lambda x: -x["delta"])[:5]:
        print(f"  {r}")


if __name__ == "__main__":
    main()

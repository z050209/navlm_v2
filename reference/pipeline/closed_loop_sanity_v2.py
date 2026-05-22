"""Self-consistent closed-loop sanity check.

Instead of trusting the stored synth `first_action`, we re-run the
planner NOW for each sample and use its internal first-segment bearing
as ground truth. Then check whether our forward math:

    h_walks = heading_gt + ACTION_DELTA[planner.first_action]
              vs
    first_seg_bearing  (what the planner actually used)

agrees within action's tolerance band:
    continue:    ±35°
    turn left:   ±55° around -90°  (range -35° to -135°)
    turn right:  ±55° around +90°  (range +35° to +135°)
    turn around: ±45° around 180°  (range >135° or <-135°)

If h_walks lands in the SAME band as first_seg_bearing relative to heading,
the math is consistent.
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
    return ((a - b + 540) % 360) - 180


def first_seg_bearing(planner, start_gps, dest_gps):
    """Match planner's internal logic exactly: nearest_node from start,
    then bearing from route[0] to route[1] (NOT from user_gps to route[1])."""
    import osmnx as ox
    G = planner.G
    sn = ox.distance.nearest_nodes(G, start_gps[1], start_gps[0])
    dn = ox.distance.nearest_nodes(G, dest_gps[1], dest_gps[0])
    try:
        route = nx.shortest_path(G, sn, dn, weight="length")
    except nx.NetworkXNoPath:
        return None
    if len(route) < 2:
        return None
    return bearing_deg(
        G.nodes[route[0]]["y"], G.nodes[route[0]]["x"],
        G.nodes[route[1]]["y"], G.nodes[route[1]]["x"],
    )


def main():
    synth_path = Path("/pub/evaluation_group/ning/test/navlm/data/cities/zurich/synth_unified.jsonl")
    pois_path  = Path("/pub/evaluation_group/ning/test/navlm/data/cities/zurich/landmarks_zurich_osm.json")
    graph_path = Path("/pub/evaluation_group/ning/test/navlm/data/cities/zurich/osm_walking.pkl")

    with open(pois_path) as f:
        pois = json.load(f)
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "toolbox"))
    try:
        from scenery_pois import SCENERY_POIS
        for n, info in SCENERY_POIS.items():
            if n not in pois:
                pois[n] = {"lat": info["lat"], "lon": info["lon"]}
    except ImportError:
        pass

    planner = WayPlanner(str(graph_path))

    samples = [json.loads(l) for l in open(synth_path) if l.strip()]
    import random
    random.Random(0).shuffle(samples)
    samples = samples[:200]
    print(f"sampling {len(samples)} for closed-loop check")

    # 1) RE-RUN planner with current heading data; record planner's
    #    first_action AND first_seg_bearing
    # 2) Compute h_walks = heading + ACTION_DELTA[first_action]
    # 3) δ = angle_diff(h_walks, first_seg_bearing)
    results = []
    for s in samples:
        m = s["_meta"]
        start = m["user_gps"]
        heading = m["user_heading"]
        dest = m["destination"]
        if heading is None or dest not in pois:
            continue
        d_lat, d_lon = pois[dest]["lat"], pois[dest]["lon"]

        try:
            plan = planner.plan(tuple(start), (d_lat, d_lon), user_heading=heading)
        except Exception:
            continue
        if plan is None or not plan["steps"]:
            continue

        first_action = plan["steps"][0]["action"]
        if first_action not in ACTION_DELTA:
            continue

        seg_bearing = first_seg_bearing(planner, tuple(start), (d_lat, d_lon))
        if seg_bearing is None:
            continue

        h_walks = (heading + ACTION_DELTA[first_action]) % 360
        delta = abs(angle_diff(h_walks, seg_bearing))

        # Also compute the "true" planner-internal turn angle (what the
        # planner used to choose the action verb)
        true_turn = angle_diff(seg_bearing, heading)

        results.append({
            "frame": m["start_frame"], "dest": dest,
            "heading": round(heading, 1),
            "true_turn": round(true_turn, 1),
            "action": first_action,
            "seg_bearing": round(seg_bearing, 1),
            "h_walks": round(h_walks, 1),
            "delta": round(delta, 1),
        })

    print(f"\nevaluated {len(results)} samples (re-planned with current graph)\n")

    # Distribution
    bins = [(0, 5), (5, 15), (15, 30), (30, 45), (45, 55), (55, 70)]
    bucket = Counter()
    for r in results:
        d = r["delta"]
        for lo, hi in bins:
            if lo <= d < hi:
                bucket[(lo, hi)] += 1
                break
        else:
            bucket[("70+", "")] += 1

    print(f"δ = |angle(h_walks, true_first_seg_bearing)| distribution:")
    for lo, hi in bins:
        n = bucket[(lo, hi)]
        bar = "█" * int(40 * n / max(len(results), 1))
        print(f"  [{lo:>3}–{hi:>3}°)  {n:>4d}  {bar}")
    n70 = bucket[("70+", "")]
    print(f"  [ 70+    °)  {n70:>4d}  " + "█"*int(40*n70/max(len(results),1)))

    n_pass  = sum(1 for r in results if r["delta"] < 30)
    n_warn  = sum(1 for r in results if 30 <= r["delta"] <= 55)
    n_fail  = sum(1 for r in results if r["delta"] > 55)
    print(f"\n  PASS  (<30°)   {n_pass}/{len(results)}  {100*n_pass/len(results):.1f}%")
    print(f"  WARN  (30-55°) {n_warn}/{len(results)}  {100*n_warn/len(results):.1f}%")
    print(f"  FAIL  (>55°)   {n_fail}/{len(results)}  {100*n_fail/len(results):.1f}%")

    # By action class
    print(f"\nδ by action class (mean and 90th pct):")
    by_act = {}
    for r in results:
        by_act.setdefault(r["action"], []).append(r["delta"])
    for a in ["continue ahead", "turn left", "turn right", "turn around"]:
        if a in by_act:
            v = sorted(by_act[a])
            n = len(v)
            mean = sum(v) / n
            p90 = v[int(0.9 * n)] if n > 0 else 0
            print(f"  {a:<15} n={n:>3d}  mean={mean:>5.1f}°  p90={p90:>5.1f}°")

    print(f"\n=== FAIL examples (δ>55°) ===")
    for r in sorted(results, key=lambda x: -x["delta"])[:5]:
        print(f"  {r}")


if __name__ == "__main__":
    main()

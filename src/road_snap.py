"""Stage 7 — OSM + HMM road-snapping (DEV_MANUAL §2.5).

Snaps the noisy per-frame GPS sequence onto the OSM walking graph with
HMM map-matching (Newson-Krumm Viterbi). Per GPS observation we take
candidate road points; the **emission** probability comes from the
GPS-to-candidate distance, the **transition** probability from how well
the on-road distance between consecutive candidates matches their
straight-line distance; Viterbi then picks the most likely road path.

Pure functions (emission_logp, transition_logp, viterbi) are
unit-tested; `snap()` needs the osmnx walking graph.
"""

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config  # noqa: E402


def emission_logp(gps_dist_m, sigma_m=20.0):
    """Log emission probability — Gaussian on GPS-to-candidate distance.
    Closer candidate -> higher (less negative) score."""
    return -0.5 * (gps_dist_m / sigma_m) ** 2


def transition_logp(great_circle_m, route_m, beta_m=30.0):
    """Log transition probability — penalises a big gap between the
    on-road distance and the straight-line distance (a detour/teleport)."""
    return -abs(route_m - great_circle_m) / beta_m


def viterbi(obs_states, emit_logp, trans_logp):
    """Generic Viterbi decoder.

    obs_states  : list (per observation) of candidate-state lists
    emit_logp   : f(t, state) -> log emission score
    trans_logp  : f(t, prev_state, state) -> log transition score
    Returns the most-likely state path (one state per observation).
    """
    if not obs_states:
        return []
    V = [{s: emit_logp(0, s) for s in obs_states[0]}]
    back = [{}]
    for t in range(1, len(obs_states)):
        V.append({})
        back.append({})
        for s in obs_states[t]:
            best_prev, best = None, float("-inf")
            for ps in obs_states[t - 1]:
                sc = V[t - 1][ps] + trans_logp(t, ps, s)
                if sc > best:
                    best, best_prev = sc, ps
            V[t][s] = best + emit_logp(t, s)
            back[t][s] = best_prev
    path = [max(V[-1], key=V[-1].get)]
    for t in range(len(obs_states) - 1, 0, -1):
        path.append(back[t][path[-1]])
    return list(reversed(path))


def _haversine_m(la1, lo1, la2, lo2):
    R = 6_371_000.0
    p1, p2 = math.radians(la1), math.radians(la2)
    dphi = math.radians(la2 - la1)
    dlam = math.radians(lo2 - lo1)
    a = (math.sin(dphi / 2) ** 2
         + math.cos(p1) * math.cos(p2) * math.sin(dlam / 2) ** 2)
    return 2 * R * math.asin(math.sqrt(a))


def snap(gps_seq, graph_path=None, radius_m=40.0):
    """HMM-snap a [(lat, lon), ...] sequence onto the OSM walking graph.

    Returns a snapped [(lat, lon), ...]. Needs osmnx + a pickled graph.
    """
    import pickle
    import networkx as nx
    import osmnx as ox

    graph_path = graph_path or (config.CITY_DIR / "osm_walking.pkl")
    with open(graph_path, "rb") as f:
        G = pickle.load(f)

    # candidate graph nodes within radius_m of each observation
    obs_states = []
    for lat, lon in gps_seq:
        node = ox.distance.nearest_nodes(G, lon, lat)
        cands = [node] + [n for n in G.neighbors(node)]
        obs_states.append(cands)

    def emit(t, node):
        olat, olon = gps_seq[t]
        d = _haversine_m(olat, olon, G.nodes[node]["y"], G.nodes[node]["x"])
        return emission_logp(d)

    def trans(t, prev, cur):
        gc = _haversine_m(G.nodes[prev]["y"], G.nodes[prev]["x"],
                          G.nodes[cur]["y"], G.nodes[cur]["x"])
        try:
            route = nx.shortest_path_length(G, prev, cur, weight="length")
        except nx.NetworkXNoPath:
            route = gc * 10           # heavy penalty for unreachable
        return transition_logp(gc, route)

    path = viterbi(obs_states, emit, trans)
    return [(G.nodes[n]["y"], G.nodes[n]["x"]) for n in path]


def main():
    print("[road_snap] HMM map-matching — needs recovered GPS "
          "(src.gps_recovery) and the OSM walking graph.")


if __name__ == "__main__":
    main()

"""HMM-based map matching of GPS trajectories to OSM walking graph.

Implements Newson & Krumm (2009) "Hidden Markov Map Matching Through Noise
and Sparseness". For each frame in time order:

  state space: candidate OSM edges within `--max-radius-m` of the GPS
  emission   : exp(-perpendicular_distance / sigma)
  transition : exp(-|great_circle_dist - osm_route_dist| / beta)
  decode     : Viterbi → most likely sequence of edges

Output per frame:
  matched_edge_id (u, v, key)
  snapped_lat, snapped_lon  (perpendicular foot on the edge)
  match_confidence          (Viterbi log-prob normalised)

Usage
-----
    python toolbox/map_match.py \\
        --input data/cities/zurich/frame_gps.jsonl \\
        --graph data/cities/zurich/osm_walking.pkl \\
        --out   data/cities/zurich/frame_gps_matched.jsonl
"""

import argparse
import json
import math
import pickle
import sys
import time
from pathlib import Path

import networkx as nx


# ---------- geometry helpers ----------

def haversine_m(lat1, lon1, lat2, lon2):
    R = 6371000.0
    p1 = math.radians(lat1); p2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1); dlam = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlam / 2) ** 2
    return 2 * R * math.asin(math.sqrt(a))


def perpendicular_dist_to_edge(p_lat, p_lon, a_lat, a_lon, b_lat, b_lon):
    """Approximate perpendicular distance from point P to segment AB
    in metres. Project to local-plane (small enough city, fine).
    Returns (perp_dist_m, foot_lat, foot_lon, t) where t in [0,1] along AB."""
    # local plane in metres
    LAT_M = 111000.0
    LON_M = 78000.0   # at ~47°N
    px = (p_lon - a_lon) * LON_M
    py = (p_lat - a_lat) * LAT_M
    bx = (b_lon - a_lon) * LON_M
    by = (b_lat - a_lat) * LAT_M
    seg_len2 = bx * bx + by * by
    if seg_len2 < 1e-9:
        return haversine_m(p_lat, p_lon, a_lat, a_lon), a_lat, a_lon, 0.0
    t = (px * bx + py * by) / seg_len2
    t_clamp = max(0.0, min(1.0, t))
    foot_x = t_clamp * bx
    foot_y = t_clamp * by
    foot_lat = a_lat + foot_y / LAT_M
    foot_lon = a_lon + foot_x / LON_M
    perp = math.hypot(px - foot_x, py - foot_y)
    return perp, foot_lat, foot_lon, t_clamp


# ---------- candidate edges ----------

def build_edge_index(graph):
    """Pre-compute (u, v, k, lat1, lon1, lat2, lon2) for every edge so we
    can do fast bbox + distance scans."""
    edges = []
    for u, v, k, _ in graph.edges(keys=True, data=True):
        try:
            lat1, lon1 = graph.nodes[u]["y"], graph.nodes[u]["x"]
            lat2, lon2 = graph.nodes[v]["y"], graph.nodes[v]["x"]
        except KeyError:
            continue
        edges.append((u, v, k, lat1, lon1, lat2, lon2))
    return edges


def candidates_within(edges, p_lat, p_lon, max_radius_m, top_k):
    """Return list of (key, edge_idx, perp_dist_m, foot_lat, foot_lon, t)
    for the closest top_k edges within max_radius_m."""
    LAT_M = 111000.0
    LON_M = 78000.0
    bbox_deg_lat = max_radius_m / LAT_M
    bbox_deg_lon = max_radius_m / LON_M
    cands = []
    for ei, (u, v, k, la1, lo1, la2, lo2) in enumerate(edges):
        # Quick bbox reject
        if (max(la1, la2) < p_lat - bbox_deg_lat or
            min(la1, la2) > p_lat + bbox_deg_lat or
            max(lo1, lo2) < p_lon - bbox_deg_lon or
            min(lo1, lo2) > p_lon + bbox_deg_lon):
            continue
        d, fla, flo, t = perpendicular_dist_to_edge(
            p_lat, p_lon, la1, lo1, la2, lo2)
        if d > max_radius_m:
            continue
        cands.append(((u, v, k), ei, d, fla, flo, t))
    cands.sort(key=lambda x: x[2])
    return cands[:top_k]


# ---------- Viterbi ----------

def viterbi(observations, edges, graph, sigma=20.0, beta=10.0,
             max_radius_m=80.0, top_k=10):
    """Run HMM map matching on a list of (frame_id, lat, lon) observations.

    Returns list of dicts: {frame_id, edge, snap_lat, snap_lon, confidence}.
    `confidence` ∈ [0,1] is local emission strength (close = 1, far = 0).
    """
    if not observations:
        return []

    # Pre-compute candidate edges per observation
    cands_per_t = []
    for fid, lat, lon in observations:
        cands = candidates_within(edges, lat, lon, max_radius_m, top_k)
        cands_per_t.append(cands)
    print(f"  candidates: avg={sum(len(c) for c in cands_per_t)/max(len(cands_per_t),1):.1f} "
          f"max={max((len(c) for c in cands_per_t), default=0)} "
          f"empty={sum(1 for c in cands_per_t if not c)}")

    # Forward pass
    log_prob = []   # list of dict {state_idx → log_prob}
    backptr  = []   # list of dict {state_idx → previous_state_idx}

    # Initial step: emission only
    init = {}
    for j, c in enumerate(cands_per_t[0]):
        d = c[2]
        init[j] = -(d * d) / (2 * sigma * sigma)  # log of gaussian (un-norm)
    log_prob.append(init)
    backptr.append({})

    # Forward
    for t in range(1, len(observations)):
        prev = log_prob[-1]
        if not prev or not cands_per_t[t]:
            log_prob.append({})
            backptr.append({})
            continue

        prev_obs = observations[t - 1]
        curr_obs = observations[t]
        gc_dist = haversine_m(prev_obs[1], prev_obs[2], curr_obs[1], curr_obs[2])

        cur_lp = {}
        cur_bp = {}
        for j, c_curr in enumerate(cands_per_t[t]):
            # Emission
            em = -(c_curr[2] ** 2) / (2 * sigma * sigma)

            # Find best predecessor
            best, best_lp = None, float("-inf")
            curr_foot = (c_curr[3], c_curr[4])      # snap point on curr edge
            curr_u, curr_v, _ = c_curr[0]
            for i, c_prev in enumerate(cands_per_t[t - 1]):
                if i not in prev:
                    continue
                # OSM routed distance between snap points (use endpoint approx)
                prev_foot = (c_prev[3], c_prev[4])
                prev_u, prev_v, _ = c_prev[0]
                # Approximate OSM distance via shortest_path between
                # the two edges' nearer endpoints + foot offsets.
                osm_d = approx_osm_dist(graph, c_prev, c_curr)
                if osm_d is None:
                    continue
                trans = -abs(gc_dist - osm_d) / beta
                lp = prev[i] + trans
                if lp > best_lp:
                    best_lp = lp
                    best = i
            if best is not None:
                cur_lp[j] = best_lp + em
                cur_bp[j] = best

        if not cur_lp:
            # Recovery: re-init from emission only.  Mark predecessor as
            # "best emission of t-1" so the backtrace can still walk through.
            best_prev_state = None
            best_prev_lp = float("-inf")
            for i, lp in prev.items():
                if lp > best_prev_lp:
                    best_prev_lp = lp
                    best_prev_state = i
            for j, c in enumerate(cands_per_t[t]):
                cur_lp[j] = -(c[2] ** 2) / (2 * sigma * sigma)
                cur_bp[j] = best_prev_state

        log_prob.append(cur_lp)
        backptr.append(cur_bp)

        if (t + 1) % 500 == 0:
            print(f"  viterbi t={t+1}/{len(observations)}")

    # Backtrace
    if not log_prob[-1]:
        # No final state, fall back to per-frame argmin emission
        out = []
        for (fid, lat, lon), cands in zip(observations, cands_per_t):
            if cands:
                c = cands[0]
                out.append({"frame_id": fid, "edge": list(c[0]),
                             "snap_lat": c[3], "snap_lon": c[4],
                             "perp_m": c[2], "confidence": math.exp(-(c[2]**2)/(2*sigma*sigma)),
                             "matched": False})
            else:
                out.append({"frame_id": fid, "edge": None,
                             "snap_lat": lat, "snap_lon": lon,
                             "perp_m": None, "confidence": 0.0,
                             "matched": False})
        return out

    final_state = max(log_prob[-1], key=log_prob[-1].get)
    path = [final_state]
    for t in range(len(observations) - 1, 0, -1):
        cur = path[-1]
        prev = backptr[t].get(cur) if cur is not None else None
        # If no backptr (None or current state missing), fall back to
        # best emission state at t-1 — keeps the chain alive instead
        # of leaving a None gap that propagates.
        if prev is None and log_prob[t - 1]:
            prev = max(log_prob[t - 1], key=log_prob[t - 1].get)
        path.append(prev)
    path.reverse()

    out = []
    for t, ((fid, lat, lon), cands) in enumerate(zip(observations, cands_per_t)):
        s = path[t]
        if s is not None and s < len(cands):
            c = cands[s]
            out.append({
                "frame_id": fid,
                "edge": list(c[0]),
                "snap_lat": c[3],
                "snap_lon": c[4],
                "perp_m": c[2],
                "confidence": math.exp(-(c[2] ** 2) / (2 * sigma * sigma)),
                "matched": True,
            })
        else:
            # No state available at t — return raw GPS, low confidence
            out.append({
                "frame_id": fid,
                "edge": None,
                "snap_lat": lat,
                "snap_lon": lon,
                "perp_m": None,
                "confidence": 0.0,
                "matched": False,
            })
    return out


# ---------- approximate OSM distance between two candidate edges ----------

_DIST_CACHE = {}


_DBG = {"calls": 0, "same_edge": 0, "no_path": 0, "too_far": 0, "ok": 0}


def approx_osm_dist(graph, c_prev, c_curr, max_m=1000.0):
    """Approximate OSM walking distance between snap-points on two
    candidate edges. Cached.
    Returns None if endpoints disconnected or path > max_m.
    """
    _DBG["calls"] += 1
    prev_u, prev_v, _ = c_prev[0]
    curr_u, curr_v, _ = c_curr[0]

    # Cheap case: same edge
    if (prev_u == curr_u and prev_v == curr_v) or (prev_u == curr_v and prev_v == curr_u):
        _DBG["same_edge"] += 1
        return haversine_m(c_prev[3], c_prev[4], c_curr[3], c_curr[4])

    # Find shortest path between the *nearer* endpoints of the two edges
    candidates = [(prev_u, curr_u), (prev_u, curr_v), (prev_v, curr_u), (prev_v, curr_v)]
    best = float("inf")
    for a, b in candidates:
        if a == b:
            d = 0.0
        else:
            key = (a, b) if a < b else (b, a)
            if key in _DIST_CACHE:
                d = _DIST_CACHE[key]
            else:
                try:
                    d = nx.shortest_path_length(graph, a, b, weight="length")
                except (nx.NetworkXNoPath, nx.NodeNotFound):
                    d = float("inf")
                _DIST_CACHE[key] = d
        if d < best:
            best = d
    if math.isinf(best):
        _DBG["no_path"] += 1
        return None
    if best > max_m:
        _DBG["too_far"] += 1
        return None
    _DBG["ok"] += 1
    return best


# ---------- main ----------

def load_observations(input_path):
    """Load (frame_id, lat, lon) sorted by frame index."""
    rows = []
    with open(input_path) as f:
        for ln in f:
            ln = ln.strip()
            if not ln:
                continue
            r = json.loads(ln)
            # Use the GPS we have, regardless of confidence — HMM cleans noise
            gps = r.get("gps_v2") or r.get("gps")
            if not gps:
                continue
            rows.append((r["frame_id"], gps[0], gps[1]))
    # sort by trailing number in frame_id
    import re
    def idx(fid):
        m = re.search(r"(\d+)", fid)
        return int(m.group(1)) if m else -1
    rows.sort(key=lambda r: idx(r[0]))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True,
                    help="frame_gps*.jsonl (uses gps_v2 or gps field)")
    ap.add_argument("--graph", default="data/cities/zurich/osm_walking.pkl")
    ap.add_argument("--out", required=True)
    ap.add_argument("--sigma", type=float, default=20.0,
                    help="emission sigma in metres (GPS noise)")
    ap.add_argument("--beta", type=float, default=10.0,
                    help="transition beta (lower = stricter)")
    ap.add_argument("--max-radius-m", type=float, default=80.0)
    ap.add_argument("--top-k", type=int, default=10)
    ap.add_argument("--max-frames", type=int, default=None)
    args = ap.parse_args()

    print(f"[mm] loading graph {args.graph}")
    with open(args.graph, "rb") as f:
        G = pickle.load(f)
    # Walking graph should be bidirectional. osmnx loads directed by default
    # which makes shortest_path fail across one-way segments.
    if G.is_directed():
        G = G.to_undirected()
        print(f"[mm] converted to undirected for walking")
    print(f"[mm] graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

    print(f"[mm] indexing edges...")
    edges = build_edge_index(G)
    print(f"[mm] {len(edges)} edges indexed")

    print(f"[mm] loading observations from {args.input}")
    obs = load_observations(args.input)
    if args.max_frames:
        obs = obs[: args.max_frames]
    print(f"[mm] {len(obs)} observations to match")

    t0 = time.time()
    matched = viterbi(obs, edges, G,
                       sigma=args.sigma, beta=args.beta,
                       max_radius_m=args.max_radius_m,
                       top_k=args.top_k)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n_matched = sum(1 for r in matched if r["matched"])
    with open(out_path, "w") as f:
        for r in matched:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    elapsed = time.time() - t0
    print(f"[mm] done. matched={n_matched}/{len(matched)} "
          f"({100*n_matched/max(len(matched),1):.1f}%) "
          f"({elapsed:.0f}s)  → {out_path}")
    print(f"[mm] approx_osm_dist stats: {_DBG}")


if __name__ == "__main__":
    main()

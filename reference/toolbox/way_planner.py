"""Independent way-planner tool.

Wraps `osmnx` + `networkx` shortest-path routing with cleanup logic so the
output is human-readable. Designed to be called as a TOOL by the VLM:

  Input :  start GPS  +  destination (landmark name OR GPS)
  Output:  {dest_gps, total_distance_m, estimated_minutes, mode, steps[]}
           where each step has:  action ∈ {continue, turn left, turn right,
                                           cross, arrive},
                                 street, distance_m, distance_phrase

The VLM does NOT do route planning. It only takes this output + the current
camera frame and writes scene-anchored directions for the user.

CLI usage
---------
  python toolbox/way_planner.py \
      --start 47.37563,8.54058 \
      --dest Hauptbahnhof
"""

import argparse
import json
import pickle
import sys
from pathlib import Path

import networkx as nx
import osmnx as ox

sys.path.insert(0, str(Path(__file__).resolve().parent))
from synth_utils import bearing_deg, haversine_m  # noqa
from zurich_landmarks_gps import ZURICH_LANDMARKS  # noqa
try:
    from scenery_pois import SCENERY_POIS  # noqa
except ImportError:
    SCENERY_POIS = {}


class WayPlanner:
    """Walking-route planner backed by a pickled osmnx graph."""

    def __init__(self, graph_path):
        with open(graph_path, "rb") as f:
            self.G = pickle.load(f)

    # ---------- public API ----------

    def plan(self, start_gps, dest, user_heading=None,
             min_segment_m=15.0, sharp_turn_deg=35.0):
        """Plan a walking route.

        start_gps:    (lat, lon)
        dest:         landmark name (str) or (lat, lon)
        user_heading: optional compass deg the user is currently facing.
                      If given, the FIRST step's action is rewritten to be
                      relative to the user's facing direction (so "continue
                      ahead" / "turn around" / etc. is correct from the user's
                      perspective, not just from the route's perspective).
        Returns a dict or None if no path.
        """
        s_lat, s_lon = start_gps
        if isinstance(dest, str):
            # 1. Hand-written tourist landmarks
            if dest in ZURICH_LANDMARKS:
                d_lat, d_lon, _ = ZURICH_LANDMARKS[dest]
            # 2. Scenery POIs (rivers, streets, bridges, big buildings)
            elif dest in SCENERY_POIS:
                d_lat = SCENERY_POIS[dest]["lat"]
                d_lon = SCENERY_POIS[dest]["lon"]
            else:
                raise KeyError(f"Unknown landmark: {dest!r}")
            dest_name = dest
        else:
            d_lat, d_lon = dest
            dest_name = None

        sn = ox.distance.nearest_nodes(self.G, s_lon, s_lat)
        dn = ox.distance.nearest_nodes(self.G, d_lon, d_lat)
        try:
            route = nx.shortest_path(self.G, sn, dn, weight="length")
        except nx.NetworkXNoPath:
            return None
        if len(route) < 2:
            return None

        total_m = sum(self._edge_len(u, v) for u, v in zip(route[:-1], route[1:]))
        steps = self._build_steps(route, min_segment_m, sharp_turn_deg)

        # First segment absolute bearing — independent of any user_heading
        # override. Always returned so v2 prompts can show it to the model.
        first_seg_bearing = bearing_deg(
            self.G.nodes[route[0]]["y"], self.G.nodes[route[0]]["x"],
            self.G.nodes[route[1]]["y"], self.G.nodes[route[1]]["x"],
        )

        # Override first step's action relative to user_heading (if given)
        if user_heading is not None and steps:
            user_to_route = ((first_seg_bearing - user_heading + 540) % 360) - 180
            steps[0]["action"] = self._action_for(user_to_route)
            steps[0]["turn_deg"] = round(user_to_route, 1)

        return {
            "start_gps": [round(s_lat, 6), round(s_lon, 6)],
            "dest_gps": [round(d_lat, 6), round(d_lon, 6)],
            "dest_name": dest_name,
            "mode": "walking",
            "total_distance_m": round(total_m, 1),
            "estimated_minutes": round(total_m / 84.0, 1),  # 1.4 m/s
            "raw_node_count": len(route),
            "user_heading": user_heading,
            "first_seg_bearing": round(first_seg_bearing, 1),
            "steps": steps,
        }

    def to_human(self, plan):
        """Convert a plan dict into a list of natural-language bullet lines."""
        if plan is None:
            return ["No route found."]
        lines = []
        for s in plan["steps"]:
            phrase = self._distance_phrase(s["distance_m"])
            street = s["street"]
            named = street and street != "unnamed path"
            action = s["action"]
            if action == "arrive":
                if named:
                    lines.append(f"Arrive at {plan['dest_name'] or 'your destination'} "
                                 f"after walking {phrase} along {street}.")
                else:
                    lines.append(f"Arrive at {plan['dest_name'] or 'your destination'} "
                                 f"after walking {phrase}.")
            elif action == "continue ahead":
                lines.append(f"Continue ahead {phrase}"
                             + (f" along {street}." if named else "."))
            elif action == "turn around":
                tail = f" onto {street}" if named else ""
                lines.append(f"Turn around{tail} and walk {phrase}.")
            else:  # turn left / turn right
                tail = f" onto {street}" if named else ""
                lines.append(f"{action.capitalize()}{tail} and walk {phrase}.")
        return lines

    # ---------- internals ----------

    def _edge(self, u, v):
        if not self.G.has_edge(u, v):
            return None
        return min(self.G[u][v].values(), key=lambda d: d.get("length", float("inf")))

    def _edge_len(self, u, v):
        e = self._edge(u, v)
        return e.get("length", 0.0) if e else 0.0

    def _edge_name(self, u, v):
        e = self._edge(u, v)
        if not e:
            return None
        n = e.get("name")
        if n is None:
            return None
        return n if isinstance(n, str) else (n[0] if n else None)

    def _build_steps(self, route, min_segment_m, sharp_turn_deg):
        """Collapse a node sequence into clean human steps.

        Rules:
          1. Track current street name. While name stays the same AND turns
             stay below sharp_turn_deg, keep accumulating distance.
          2. When name changes OR a sharp turn occurs, emit a step.
          3. Drop emitted steps shorter than min_segment_m by merging the
             distance into the next step.
          4. Final step is always 'arrive'.
        """
        nodes = route
        steps = []

        cur_name = self._edge_name(nodes[0], nodes[1])
        cur_len = 0.0
        # action of the segment we are currently building (set by previous turn)
        cur_action = "continue"

        for i in range(1, len(nodes)):
            cur_len += self._edge_len(nodes[i - 1], nodes[i])

            if i == len(nodes) - 1:
                break  # final segment handled below

            next_name = self._edge_name(nodes[i], nodes[i + 1])
            b1 = bearing_deg(self.G.nodes[nodes[i - 1]]["y"], self.G.nodes[nodes[i - 1]]["x"],
                             self.G.nodes[nodes[i]]["y"],     self.G.nodes[nodes[i]]["x"])
            b2 = bearing_deg(self.G.nodes[nodes[i]]["y"],     self.G.nodes[nodes[i]]["x"],
                             self.G.nodes[nodes[i + 1]]["y"], self.G.nodes[nodes[i + 1]]["x"])
            turn = ((b2 - b1 + 540) % 360) - 180

            name_change = (next_name != cur_name) and (next_name is not None)
            sharp = abs(turn) > sharp_turn_deg

            if name_change or sharp:
                steps.append({
                    "action": cur_action,
                    "street": cur_name or "unnamed path",
                    "distance_m": round(cur_len, 1),
                    "turn_deg": round(turn, 1),
                })
                # next segment's action is determined by *this* turn
                cur_action = self._action_for(turn)
                cur_name = next_name if name_change else cur_name
                cur_len = 0.0

        # final 'arrive' step
        steps.append({
            "action": "arrive",
            "street": cur_name or "unnamed path",
            "distance_m": round(cur_len, 1),
            "turn_deg": 0.0,
        })

        # Post-process: merge tiny steps into their neighbours
        return self._merge_small(steps, min_segment_m)

    @staticmethod
    def _merge_small(steps, min_segment_m):
        if not steps:
            return steps
        out = []
        for s in steps:
            if out and s["distance_m"] < min_segment_m and s["action"] != "arrive":
                # merge this micro-step's distance into the previous step
                out[-1]["distance_m"] = round(out[-1]["distance_m"] + s["distance_m"], 1)
                continue
            out.append(s)
        # also collapse repeated 'continue' on same street
        merged = []
        for s in out:
            if (merged and s["action"] == "continue"
                    and merged[-1]["street"] == s["street"]
                    and merged[-1]["action"] in ("continue", "turn left", "turn right")):
                merged[-1]["distance_m"] = round(merged[-1]["distance_m"] + s["distance_m"], 1)
                continue
            merged.append(s)
        return merged

    @staticmethod
    def _action_for(turn_deg):
        """Translate a turn angle to a perspective-relative action verb.

        turn_deg ∈ (-180, 180]:  positive = clockwise (right), negative = ccw (left).
        """
        a = turn_deg
        if abs(a) <= 35:
            return "continue ahead"
        if abs(a) > 135:
            return "turn around"
        if a < 0:
            return "turn left"
        return "turn right"

    @staticmethod
    def _distance_phrase(m):
        """Map metres to vague phrase — never expose raw distances to the VLM."""
        if m < 30:
            return "just a few steps"
        if m < 100:
            return "about a block"
        if m < 250:
            return "a couple of blocks"
        if m < 500:
            return "a few blocks"
        return "several blocks"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--graph", default="data/cities/zurich/osm_walking.pkl")
    ap.add_argument("--start", required=True, help="lat,lon")
    ap.add_argument("--dest",  required=True, help="lat,lon OR a landmark name")
    args = ap.parse_args()

    s_lat, s_lon = map(float, args.start.split(","))
    if "," in args.dest:
        dest = tuple(map(float, args.dest.split(",")))
    else:
        dest = args.dest

    planner = WayPlanner(args.graph)
    plan = planner.plan((s_lat, s_lon), dest)
    if plan is None:
        print("NO ROUTE")
        return

    print("=== STRUCTURED PLAN (machine-readable) ===")
    print(json.dumps(plan, indent=2, ensure_ascii=False))
    print()
    print("=== HUMAN-READABLE STEPS ===")
    for line in planner.to_human(plan):
        print(f"  • {line}")


if __name__ == "__main__":
    main()

"""Unified training-data synthesizer with chain-of-thought reasoning.

This file is the orchestrator. The reusable building blocks live in the
`synth/` subpackage:

  synth/prompts.py    — system prompt, user template, question variations
  synth/backends.py   — VLM backend dispatch (gemma / openai / anthropic)
  synth/verifier.py   — output parser + format checks + visual verifier
  synth/sampling.py   — POI tier weighting + destination + reverse-lookup

Run:
    python toolbox/synth_unified.py \\
        --backend gemma \\
        --vllm-url http://localhost:8003/v1 \\
        --model    google/gemma-4-31b-it
"""

import argparse
import json
import random
import sys
import time
from pathlib import Path

# Make sibling modules importable when invoked as a script
_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

from synth_utils import frame_idx_from_id, load_jsonl  # noqa: E402
from way_planner import WayPlanner  # noqa: E402

from synth.prompts import (  # noqa: E402
    SYSTEM_PROMPT, USER_TEMPLATE, QUESTION_TEMPLATES,
    bearing_to_compass, format_nearby_poi_list, format_remaining_route,
)
from synth.backends import call_teacher  # noqa: E402
from synth.verifier import parse_assistant, verify, visual_verify_poi  # noqa: E402
from synth.sampling import (  # noqa: E402
    poi_tier,
    sample_destinations_weighted,
    reverse_lookup_location,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--starts", default="data/cities/zurich/frame_starts_trusted.jsonl",
                    help="output of build_trusted_starts.py")
    ap.add_argument("--frames-dir", default="data/cities/zurich/frames/zurich")
    ap.add_argument("--pois", default="data/cities/zurich/landmarks_zurich_osm.json")
    ap.add_argument("--graph", default="data/cities/zurich/osm_walking.pkl")
    ap.add_argument("--out", default="data/cities/zurich/synth_unified.jsonl")
    ap.add_argument("--backend", choices=["gemma", "openai", "anthropic"], default="gemma")
    ap.add_argument("--vllm-url", default="http://localhost:8003/v1")
    ap.add_argument("--model", default="google/gemma-4-31b-it")
    ap.add_argument("--limit-frames", type=int, default=None)
    ap.add_argument("--limit-emit", type=int, default=None)
    ap.add_argument("--n-dest-per-frame", type=int, default=5)
    ap.add_argument("--min-dest-m", type=float, default=150)
    ap.add_argument("--max-dest-m", type=float, default=1500)
    ap.add_argument("--max-route-steps", type=int, default=99,
                    help="drop pairs whose cleaned route exceeds this. We keep "
                         "long routes — the answer just describes the first "
                         "leg and asks the user to send another photo at a "
                         "visible checkpoint (multi-turn conversational mode).")
    ap.add_argument("--skip-visual-verify", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    # Load trusted starts (already filtered for high-confidence GPS+heading)
    frames = load_jsonl(args.starts)
    frames.sort(key=lambda r: frame_idx_from_id(r["frame_id"]))
    if args.limit_frames:
        frames = frames[: args.limit_frames]

    with open(args.pois) as f:
        pois = json.load(f)
    # Merge in scenery POIs (rivers, streets, bridges, big buildings) — these
    # become valid destinations and reverse-lookup answers too.
    try:
        from scenery_pois import SCENERY_POIS
        for name, info in SCENERY_POIS.items():
            if name not in pois:
                pois[name] = {
                    "lat": info["lat"], "lon": info["lon"],
                    "kind_label": info["kind_label"],
                    "kinds": [info["kind"]],
                    "aliases": info.get("aliases", [name]),
                    "_scenery_radius_m": info["radius_m"],
                }
        print(f"[synth] +{len(SCENERY_POIS)} scenery POIs merged")
    except ImportError:
        pass
    planner = WayPlanner(args.graph)
    print(f"[synth] trusted_starts={len(frames)}  pois={len(pois)}  "
          f"visual_verify={'OFF' if args.skip_visual_verify else 'ON'}")

    # Resume support
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done_keys = set()
    if out_path.exists():
        for r in load_jsonl(args.out):
            done_keys.add((r["_meta"]["start_frame"], r["_meta"]["destination"]))
        print(f"[synth] resume: {len(done_keys)} pairs done")

    rng = random.Random(args.seed)
    n_emit = n_skip = n_filter = 0
    t0 = time.time()
    backend_kw = {"vllm_url": args.vllm_url, "model": args.model}

    with open(out_path, "a") as fout:
        for fi, f in enumerate(frames):
            if args.limit_emit and n_emit >= args.limit_emit:
                break
            fid = f["frame_id"]
            user_gps = f["gps"]
            user_heading = f.get("heading")
            heading_conf = f.get("heading_confidence", "unknown")
            gps_source = f.get("gps_source", "unknown")

            # v2: USER_LOCATION is no longer in the prompt — model derives
            # heading from image + nearby-POI list. Skip the (slow)
            # reverse_lookup_location call and the per-frame visual verify.
            frame_visual_meta = None

            dests = sample_destinations_weighted(user_gps, pois,
                                                  args.n_dest_per_frame,
                                                  args.min_dest_m, args.max_dest_m, rng)
            if not dests:
                continue

            for dest_name, d_lat, d_lon, dist_m in dests:
                if args.limit_emit and n_emit >= args.limit_emit:
                    break
                if (fid, dest_name) in done_keys:
                    continue

                # Plan
                try:
                    plan = planner.plan(user_gps, (d_lat, d_lon),
                                        user_heading=user_heading)
                except Exception as e:
                    print(f"  plan-err {fid}/{dest_name}: {e}", file=sys.stderr)
                    n_skip += 1; continue
                if plan is None or not plan["steps"]:
                    n_skip += 1; continue

                # Drop pairs with too many turns — answer can't compress them.
                if len(plan["steps"]) > args.max_route_steps:
                    n_skip += 1; continue

                first_action = plan["steps"][0]["action"]
                first_seg_bearing = plan.get("first_seg_bearing")
                first_seg_compass = bearing_to_compass(first_seg_bearing) \
                                    if first_seg_bearing is not None else "?"
                first_seg_distance = plan["steps"][0]["distance_m"]
                first_seg_street = plan["steps"][0]["street"]
                steps_human = planner.to_human(plan)
                # n_turns = number of route segments excluding the final 'arrive'
                n_turns = max(0, len(plan["steps"]) - 1)
                question = rng.choice(QUESTION_TEMPLATES).format(dest=dest_name)

                # Build NEARBY_POI_LIST. Make sure destination is in the list,
                # even if outside the 350m radius — the model needs to look up
                # destination GPS by name.
                pois_for_prompt = dict(pois)
                if dest_name in pois:
                    pois_for_prompt.setdefault(dest_name, pois[dest_name])
                nearby_block = format_nearby_poi_list(
                    pois_for_prompt, user_gps, max_n=8, max_radius_m=350)
                if dest_name not in nearby_block:
                    # destination not in the formatted list (too far) — append it
                    p = pois[dest_name]
                    kind = p.get("kind_label") or ""
                    kind_str = f" ({kind})" if kind else ""
                    nearby_block += (f"\n  - {dest_name}{kind_str}: "
                                     f"({p['lat']:.5f}, {p['lon']:.5f})")

                remaining_block = format_remaining_route(
                    steps_human, plan["steps"], n_turns)

                user_msg = USER_TEMPLATE.format(
                    USER_LAT=user_gps[0], USER_LON=user_gps[1],
                    USER_HEADING=user_heading or 0,
                    NEARBY_POI_LIST=nearby_block,
                    FIRST_SEG_BEARING=first_seg_bearing or 0,
                    FIRST_SEG_DISTANCE=first_seg_distance,
                    FIRST_SEG_STREET=first_seg_street,
                    TOTAL_DIST_M=plan["total_distance_m"],
                    ESTIMATED_MINUTES=int(plan["estimated_minutes"]),
                    N_TURNS=n_turns,
                    REMAINING_ROUTE_BLOCK=remaining_block,
                    USER_QUESTION=question,
                )

                img_path = (Path(f["image"]) if f.get("image")
                            else Path(args.frames_dir) / f"{fid}.jpg")
                if not img_path.exists():
                    n_skip += 1; continue

                try:
                    raw = call_teacher(args.backend, img_path,
                                       SYSTEM_PROMPT, user_msg, **backend_kw)
                except Exception as e:
                    print(f"  vlm-err {fid}/{dest_name}: {type(e).__name__}: "
                          f"{str(e)[:80]}", file=sys.stderr)
                    n_skip += 1; continue

                thinking, answer = parse_assistant(raw)
                fails = verify(thinking, answer)
                if fails:
                    n_filter += 1
                    if n_filter <= 3:
                        print(f"  filt(format) {fid}/{dest_name}: {fails}",
                              file=sys.stderr)
                    continue

                row = {
                    "image": str(img_path.resolve()),
                    "messages": [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": user_msg},
                        {"role": "assistant",
                         "content": f"<thinking>\n{thinking}\n</thinking>\n\n<answer>\n{answer}\n</answer>"},
                    ],
                    "_meta": {
                        "start_frame": fid,
                        "destination": dest_name,
                        "destination_tier": poi_tier(dest_name),
                        "user_gps": user_gps,
                        "user_heading": user_heading,
                        "heading_confidence": heading_conf,
                        "gps_source": gps_source,
                        "first_action": first_action,
                        "first_seg_bearing": first_seg_bearing,
                        "distance_m": round(dist_m, 1),
                        "estimated_minutes": plan["estimated_minutes"],
                        "n_route_steps": len(plan["steps"]),
                        "n_turns": n_turns,
                        "question_template": question,
                        "visual_verifier": frame_visual_meta,
                    },
                }
                fout.write(json.dumps(row, ensure_ascii=False) + "\n")
                fout.flush()
                n_emit += 1

                if n_emit % 5 == 0:
                    rate = n_emit / max(time.time() - t0, 1e-3)
                    print(f"  [{n_emit}] rate={rate:.2f}/s  skip={n_skip}  "
                          f"filt={n_filter}")

    print(f"[synth] done. emitted={n_emit}  skipped={n_skip}  "
          f"filtered={n_filter}  ({time.time()-t0:.0f}s)  → {out_path}")


if __name__ == "__main__":
    main()

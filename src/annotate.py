"""Phase B — instruction-tuning annotation with Gemini 2.5 Pro.

For each trusted frame (Phase A output) we:

  1.  draw N=3 destination POIs from the OSM table, distance-banded
      (80 % ≤500 m, 10 % 500–1000 m, 10 % 1000–1500 m — Q6, §2.7);
  2.  plan the OSM walking route to each destination (`routing.plan_route`),
      get the first-segment absolute bearing `B`;
  3.  build a prompt containing the **frame image, GPS, heading, route
      polyline summary, nearby POIs, and destination POI**;
  4.  call Gemini 2.5 Pro, which returns `<thinking>` (6 CoT steps,
      INCLUDING an INFERRED_HEADING line) + `<answer>` (2–4 TTS-friendly
      sentences, relative verbs anchored to a visible object);
  5.  parse the verb out of `<answer>` and gate with the closed-loop
      verifier (`δ = |heading + ACTION_DELTA[verb] − B| < 30°`);
  6.  write the kept sample to `annotations.jsonl` (one record per
      (frame, destination) pair).

The teacher's reply is kept **with the full CoT including heading**.
Downstream we derive three training views from the same file:

  given      — drop nothing; user message keeps heading visible
  implicit   — strip the heading line from user message + drop heading
               step from <thinking>
  explicit   — strip the heading line from user message but KEEP the
               INFERRED_HEADING step in <thinking>

That derivation lives in `src/derive_variants.py` (next step); this
module just produces the **single rich annotation** all variants are
built from.

  python -m src.annotate --limit 5                  # smoke (5 frames)
  python -m src.annotate --prompt-variant compact   # try a different prompt
  python -m src.annotate                            # full batch

Pure helpers (`sample_destinations`, `verify`, `parse_answer`,
`build_user_msg`) are unit-tested.
"""

import argparse
import json
import math
import random
import re
import sys
from collections import defaultdict
from pathlib import Path

from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config                                       # noqa: E402
from src.routing import (                           # noqa: E402
    ACTION_DELTA, action_for, angle_diff, bearing_deg,
    closed_loop_delta, distance_phrase, plan_route)

# (min_m, max_m, share) — DEV_MANUAL §2.7
DIST_BANDS = [(0, 500, 0.80), (500, 1000, 0.10), (1000, 1500, 0.10)]

# ────────────────────────────────────────────────────────────────────
# System-prompt VARIANTS the teacher Gemini 2.5 Pro is run with.
# Pick on the CLI with `--prompt-variant {strict,compact,reasoner,
# scene}`. They differ in (a) how rigid the CoT schema is, and (b)
# how loudly the system instruction tells the model to commit to a
# heading. Each variant uses the SAME user message format, so the
# kept annotations are directly comparable.
# ────────────────────────────────────────────────────────────────────
SYS_PROMPTS = {
    # ----- strict (default) -- explicit 6-step CoT schema with a
    # named INFERRED_HEADING line. Closest to the v1 navlm_ss prompt.
    "strict": (
        "You are a Zurich-local walking-tour guide. You give SPOKEN "
        "directions to a tourist who has sent you a phone photo of "
        "what they currently see, plus their GPS and the destination "
        "they want to walk to. They have no compass.\n"
        "Reply with EXACTLY this format and nothing else:\n"
        "<thinking>\n"
        "STEP 1 SCENE: what is visible in the photo (one sentence).\n"
        "STEP 2 LANDMARKS: which of the nearby POIs you can identify.\n"
        "STEP 3 INFERRED_HEADING: a number 0-359 (degrees, 0=N, 90=E), "
        "reasoning from the landmarks above.\n"
        "STEP 4 ROUTE: the planned bearing to the destination and "
        "what that means relative to the inferred heading.\n"
        "STEP 5 ACTION: one of {continue ahead, turn left, turn right, "
        "turn around}.\n"
        "STEP 6 ANCHOR: a CONCRETE object visible in the photo to "
        "anchor the action to (e.g. 'the tram tracks').\n"
        "</thinking>\n"
        "<answer>\n"
        "2-4 short sentences, TTS-friendly. Use the action verb from "
        "STEP 5, the anchor from STEP 6, the distance phrase, and the "
        "destination name. NO compass words, NO numbers, NO GPS, NO "
        "street names not in the route. End with a one-sentence "
        "check-in (\"when you reach <X>, send me another photo\") only "
        "when the route is long (>3 turns).\n"
        "</answer>"
    ),
    # ----- compact -- short schema, no labelled steps. Saves output
    # tokens (cheaper per call) but loses the explicit heading step.
    "compact": (
        "You are a Zurich-local walking-tour guide giving SPOKEN "
        "directions from a phone photo + GPS + destination, with no "
        "compass.\n"
        "Reply with:\n"
        "<thinking> Identify the place. Decide which way the camera "
        "is facing (0=N, 90=E). Compare to the planned bearing and "
        "pick one of {continue ahead, turn left, turn right, turn "
        "around}. Pick a visible anchor object.</thinking>\n"
        "<answer> 2-4 TTS sentences, no compass words / numbers / GPS, "
        "anchored to a visible object. </answer>"
    ),
    # ----- reasoner -- pushes the model to reason longer about the
    # heading (helpful for the explicit-CoT condition).
    "reasoner": (
        "You are a Zurich-local walking-tour guide. The tourist has "
        "no compass; their phone photo is your ONLY clue to which way "
        "they are facing. Reason carefully about this before giving "
        "directions.\n"
        "Reply with EXACTLY this format:\n"
        "<thinking>\n"
        "SCENE: what is in the photo.\n"
        "VISIBLE_LANDMARKS: which nearby POIs you can identify, and "
        "where they sit in the frame (left/centre/right, near/far).\n"
        "HEADING_REASONING: walk through the geometry — if landmark X "
        "is at GPS (lat, lon) and appears on the LEFT of the frame, "
        "the camera must be facing roughly ___ degrees. Cross-check "
        "with a second landmark.\n"
        "INFERRED_HEADING: a single number 0-359 (degrees, 0=N).\n"
        "ROUTE: planned bearing to destination, relative direction.\n"
        "ACTION: one of {continue ahead, turn left, turn right, turn "
        "around}.\n"
        "ANCHOR: a concrete visible object.\n"
        "</thinking>\n"
        "<answer> 2-4 TTS-friendly sentences anchored to the anchor "
        "object. No compass words/numbers/GPS. </answer>"
    ),
    # ----- scene -- biases the CoT toward grounding the answer in
    # what is *concretely* visible (counter-hallucination).
    "scene": (
        "You are a Zurich-local walking-tour guide giving SPOKEN "
        "directions from a phone photo + GPS to a tourist with no "
        "compass. EVERY claim you make must trace back to something "
        "visible in the photo.\n"
        "<thinking>\n"
        "VISIBLE: enumerate concrete objects/signs/landmarks in the "
        "frame.\n"
        "PLACE: which of the nearby POIs match.\n"
        "INFERRED_HEADING: number 0-359, reasoned from the visible "
        "POIs and their map positions.\n"
        "ROUTE: planned bearing, relative direction.\n"
        "ACTION: continue ahead | turn left | turn right | turn around.\n"
        "ANCHOR: a visible object the tourist will see RIGHT NOW.\n"
        "</thinking>\n"
        "<answer> 2-4 TTS sentences, no compass words/numbers/GPS, "
        "anchored to the anchor object. </answer>"
    ),
}


# ── pure helpers ────────────────────────────────────────────────────
def sample_destinations(candidates, n=config.DEST_PER_FRAME, seed=0):
    """Pick `n` destinations by distance band — see DIST_BANDS.

    candidates: list of (name, distance_m). Returns a list of
    (name, distance_m). Each slot draws a band by its share, then a
    candidate inside that band; empty bands fall back to the nearest
    unused candidate so we still return `n` when enough candidates
    exist."""
    rng = random.Random(seed)
    by_band = [[c for c in candidates if lo <= c[1] < hi]
               for lo, hi, _ in DIST_BANDS]
    weights = [share for _, _, share in DIST_BANDS]
    pool = sorted(candidates, key=lambda c: c[1])
    chosen, used = [], set()
    for _ in range(n):
        band = rng.choices(range(len(DIST_BANDS)), weights=weights)[0]
        opts = [c for c in by_band[band] if c[0] not in used]
        if not opts:
            opts = [c for c in pool if c[0] not in used]
        if not opts:
            break
        pick = rng.choice(opts)
        chosen.append(pick)
        used.add(pick[0])
    return chosen


def verify(heading, action, route_bearing, max_delta=30.0):
    """Closed-loop verifier — True if the action points the right way.
    DEV_MANUAL §2.7 Q5: the 30° tolerance."""
    return closed_loop_delta(heading, action, route_bearing) < max_delta


def parse_answer(raw):
    """Pull `<thinking>...</thinking>` and `<answer>...</answer>` out of
    the teacher reply. Returns (thinking, answer, action_verb) — verb
    is the first match of any of the four action verbs in `<answer>`,
    or None. Pure — unit-tested."""
    think = re.search(r"<thinking>(.*?)</thinking>", raw, re.S | re.I)
    answ = re.search(r"<answer>(.*?)</answer>", raw, re.S | re.I)
    thinking = think.group(1).strip() if think else ""
    answer = answ.group(1).strip() if answ else ""
    verb = None
    for v in ACTION_DELTA.keys():
        if re.search(r"\b" + re.escape(v) + r"\b", answer, re.I):
            verb = v
            break
    return thinking, answer, verb


def build_user_msg(frame, route, nearby_pois, dest_name, dest_dist_m):
    """Format the user message — kept identical across SYS_PROMPTS so
    the CoT is the only thing that varies between variants. Pure."""
    lat, lon = frame["gps"]
    near_lines = ["- " + p["name"] for p in nearby_pois[:8]]
    msg = (
        f"My GPS: {lat:.5f}, {lon:.5f}\n"
        f"My camera heading: {frame['heading']:.0f}° (0=N, 90=E)\n"
        f"Destination: {dest_name} "
        f"({distance_phrase(dest_dist_m)}, "
        f"first-segment bearing {route['first_seg_bearing']:.0f}°)\n"
        f"Walking-route distance: {route['distance_m']:.0f} m\n"
    )
    if near_lines:
        msg += "Nearby POIs:\n" + "\n".join(near_lines) + "\n"
    msg += ("Tell me what to do, in 2-4 spoken sentences, anchored to "
            "something I can actually see in the photo.")
    return msg


def _haversine_m(la1, lo1, la2, lo2):
    R = 6_371_000.0
    p1, p2 = math.radians(la1), math.radians(la2)
    dphi = math.radians(la2 - la1)
    dlam = math.radians(lo2 - lo1)
    a = (math.sin(dphi / 2) ** 2
         + math.cos(p1) * math.cos(p2) * math.sin(dlam / 2) ** 2)
    return 2 * R * math.asin(math.sqrt(a))


# ── data loading ─────────────────────────────────────────────────────
def _load_trusted_frames():
    """phaseA_trusted.jsonl if present, else accepted rows of
    gps_recovery_all.jsonl as a degraded fallback (so the smoke test
    can be run before HMM+heading_qc exist)."""
    trusted = config.CITY_DIR / "phaseA_trusted.jsonl"
    if trusted.exists():
        print(f"[annotate] using {trusted.name}", flush=True)
        return [json.loads(l) for l in trusted.open(encoding="utf-8")
                if l.strip()]
    fallback = config.CITY_DIR / "gps_recovery_all.jsonl"
    print(f"[annotate] {trusted.name} absent — falling back to "
          f"{fallback.name} (accepted rows only)", flush=True)
    out = []
    for line in fallback.open(encoding="utf-8"):
        if not line.strip():
            continue
        r = json.loads(line)
        if not r.get("accepted"):
            continue
        out.append({
            "video": r["video"], "frame_id": r["frame_id"],
            "gps": r["gps"], "heading": r.get("heading", 0.0),
            "heading_gap": r.get("heading_gap", 0.0),
            "tier": r.get("tier"),
        })
    return out


def _load_point_pois():
    """POIs that have a single (lat, lon) — destinations the route
    planner can target. Drops ways/polygons (a route to "Bahnhofstrasse"
    needs a chosen endpoint, not a polyline)."""
    pois = json.loads((config.CITY_DIR / "pois.json").read_text(
        encoding="utf-8"))
    return [p for p in pois if p.get("lat") and p.get("lon")]


def _nearby_pois(frame_gps, point_pois, radius_m=300.0, k=10):
    """Up to k POIs within radius_m of the frame GPS, by haversine."""
    out = []
    for p in point_pois:
        d = _haversine_m(frame_gps[0], frame_gps[1], p["lat"], p["lon"])
        if d <= radius_m:
            out.append((d, p))
    out.sort(key=lambda x: x[0])
    return [p for _, p in out[:k]]


def _candidates_for(frame_gps, point_pois, max_m=1500.0):
    """All point POIs within `max_m` of the frame, as
    [(name, distance_m, lat, lon, kind_label)] sorted by distance."""
    out = []
    for p in point_pois:
        d = _haversine_m(frame_gps[0], frame_gps[1], p["lat"], p["lon"])
        if d <= max_m:
            out.append({"name": p["name"], "dist_m": d,
                        "lat": p["lat"], "lon": p["lon"],
                        "kind_label": p.get("kind_label", "")})
    out.sort(key=lambda x: x["dist_m"])
    return out


# ── main ────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("--limit", type=int, default=5,
                    help="annotate only N frames (5-sample smoke first)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--prompt-variant",
                    choices=list(SYS_PROMPTS.keys()), default="strict",
                    help="which system prompt to use (see SYS_PROMPTS)")
    ap.add_argument("--output", default=None,
                    help="annotations.jsonl path (default in CITY_DIR)")
    ap.add_argument("--dest-per-frame", type=int,
                    default=config.DEST_PER_FRAME,
                    help=f"destinations per frame (default "
                         f"{config.DEST_PER_FRAME})")
    ap.add_argument("--max-delta", type=float, default=30.0,
                    help="closed-loop verifier tolerance (deg)")
    ap.add_argument("--per-video-cap", type=int, default=0,
                    help="cap kept frames per video (0=no cap)")
    args = ap.parse_args()

    out_path = (Path(args.output) if args.output
                else config.CITY_DIR /
                f"annotations_{args.prompt_variant}.jsonl")

    # ─── load Phase A + POIs ──────────────────────────────────────
    frames = _load_trusted_frames()
    point_pois = _load_point_pois()
    print(f"[annotate] frames available: {len(frames):,}  ·  "
          f"point POIs: {len(point_pois)}", flush=True)

    rng = random.Random(args.seed)
    rng.shuffle(frames)

    if args.per_video_cap:
        capped, seen = [], defaultdict(int)
        for f in frames:
            if seen[f["video"]] < args.per_video_cap:
                capped.append(f)
                seen[f["video"]] += 1
        frames = capped

    if args.limit:
        frames = frames[:args.limit]
    print(f"[annotate] processing {len(frames)} frames  ·  "
          f"prompt='{args.prompt_variant}'  ·  "
          f"{args.dest_per_frame} dest/frame  ·  "
          f"verifier δ<{args.max_delta}°", flush=True)

    sys_prompt = SYS_PROMPTS[args.prompt_variant]
    from src.gemini_api import call_gemini

    # ─── resume support ───────────────────────────────────────────
    done = set()
    if out_path.exists():
        for line in out_path.open(encoding="utf-8"):
            if not line.strip():
                continue
            d = json.loads(line)
            done.add((d["video"], d["frame_id"], d["dest_name"]))
        print(f"[annotate] already done: {len(done)} (resume mode)",
              flush=True)

    # ─── per frame ────────────────────────────────────────────────
    n_kept = n_skipped_no_route = n_failed_verify = n_errored = 0
    cost_total = 0.0
    with out_path.open("a", encoding="utf-8") as fout:
        pbar = tqdm(frames, desc="[annotate]", unit="frame",
                    dynamic_ncols=True)
        for f_idx, frame in enumerate(pbar):
            cands = _candidates_for(frame["gps"], point_pois)
            picks = sample_destinations(
                [(c["name"], c["dist_m"]) for c in cands],
                n=args.dest_per_frame, seed=args.seed + f_idx)
            by_name = {c["name"]: c for c in cands}

            img_path = (config.FRAMES_DIR / frame["video"] /
                        f"{frame['frame_id']}.jpg")

            for dest_name, dest_dist_m in picks:
                if (frame["video"], frame["frame_id"], dest_name) in done:
                    continue
                d = by_name[dest_name]
                # plan route — needs the osmnx walking graph; skip if absent
                try:
                    route = plan_route(frame["gps"], (d["lat"], d["lon"]))
                except FileNotFoundError:
                    pbar.write("[annotate] osm_walking.pkl missing — "
                               "haversine bearing fallback")
                    route = None
                if route is None:
                    # bearing fallback: straight-line bearing
                    b = bearing_deg(frame["gps"][0], frame["gps"][1],
                                    d["lat"], d["lon"])
                    route = {"distance_m": dest_dist_m,
                             "first_seg_bearing": b,
                             "n_nodes": 0,
                             "route_latlon": [frame["gps"],
                                              [d["lat"], d["lon"]]]}

                nearby = _nearby_pois(frame["gps"], point_pois)
                user_msg = build_user_msg(frame, route, nearby,
                                          dest_name, dest_dist_m)

                try:
                    raw = call_gemini(
                        img_path, sys_prompt, user_msg,
                        model=config.GEMINI_ANNOTATE,
                        max_tokens=2048,
                        label=f"annotate_{args.prompt_variant}_"
                              f"{frame['video']}_{frame['frame_id']}")
                except Exception as e:
                    n_errored += 1
                    pbar.write(f"  ERROR {frame['frame_id']} -> "
                               f"{dest_name}: {type(e).__name__}: {e}")
                    continue

                thinking, answer, verb = parse_answer(raw)
                if verb is None:
                    n_failed_verify += 1
                    delta = None
                    accepted = False
                else:
                    delta = closed_loop_delta(
                        frame["heading"], verb, route["first_seg_bearing"])
                    accepted = delta < args.max_delta
                    if not accepted:
                        n_failed_verify += 1

                rec = {
                    "video": frame["video"],
                    "frame_id": frame["frame_id"],
                    "gps": frame["gps"],
                    "heading": frame["heading"],
                    "heading_gap": frame.get("heading_gap"),
                    "dest_name": dest_name,
                    "dest_gps": [d["lat"], d["lon"]],
                    "dest_dist_m": dest_dist_m,
                    "route_bearing": route["first_seg_bearing"],
                    "route_distance_m": route["distance_m"],
                    "route_latlon": route["route_latlon"][:50],
                    "nearby_pois": [p["name"] for p in nearby],
                    "prompt_variant": args.prompt_variant,
                    "raw_response": raw,
                    "thinking": thinking,
                    "answer": answer,
                    "action": verb,
                    "verifier_delta": delta,
                    "accepted": accepted,
                }
                fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
                fout.flush()
                if accepted:
                    n_kept += 1
                cost_total += 0.014   # measured ~$/frame, rough
                pbar.set_postfix_str(
                    f"kept={n_kept} fail={n_failed_verify} "
                    f"~${cost_total:.2f}")
        pbar.close()

    print(flush=True)
    print("=== annotation summary ===", flush=True)
    print(f"kept (verifier-pass):  {n_kept}", flush=True)
    print(f"failed verifier:       {n_failed_verify}", flush=True)
    print(f"errored API calls:     {n_errored}", flush=True)
    print(f"output:                {out_path}", flush=True)
    print(f"rough total cost:      ~${cost_total:.2f}", flush=True)


if __name__ == "__main__":
    main()

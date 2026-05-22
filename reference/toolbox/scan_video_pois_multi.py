"""Multi-POI per-frame VLM scan.

Earlier `scan_video_pois.py` picks ONE top POI per frame. This version asks
Gemma to list ALL visible POIs from the candidate list (can be 0 or many).
Lets us answer "in which frames does Hauptbahnhof appear?" properly.

Output JSONL per frame:
  {"video": ..., "frame_id": ..., "visible_pois": ["Hauptbahnhof", "Bahnhofstrasse"]}

~1.5 s per frame on Gemma. 1358 frames → ~30 min.
"""

import argparse
import base64
import json
import re
import sys
import time
from pathlib import Path

import requests

# Same candidate list as the single-pick scan + scenery additions
CANDIDATE_POIS = [
    # Iconic landmarks (also in OSM table)
    "Hauptbahnhof", "Lindenhof", "Paradeplatz", "Münsterhof",
    "Fraumünster", "Grossmünster", "St. Peter",
    "Bellevueplatz", "Sechseläutenplatz", "Bürkliplatz",
    "Quaibrücke", "Münsterbrücke", "Rathausbrücke",
    "Rathaus", "Stadthaus",
    "Opernhaus", "Kunsthaus", "Landesmuseum", "Polyterrasse",
    "Globus", "Jelmoli",
    # Scenery (streets, river, lake)
    "Bahnhofstrasse", "Niederdorfstrasse", "Limmatquai", "Rennweg",
    "Limmat river", "Lake Zurich",
]


def img_to_data_url(path):
    return f"data:image/jpeg;base64,{base64.b64encode(path.read_bytes()).decode()}"


def build_prompt():
    listing = "\n".join(f"  - {n}" for n in CANDIDATE_POIS)
    return (
        "Look at this photograph taken in Zurich, Switzerland.\n"
        "From the candidate list below, identify ALL landmarks/places that "
        "are clearly visible in the photo (can be 0, 1, or several).\n\n"
        f"Candidates:\n{listing}\n\n"
        "Reply with one line per visible landmark, in the format:\n"
        "  POI: <exact name from list>\n"
        "If none of them is visible, reply only:\n"
        "  POI: none\n"
        "No explanations, no extra text."
    )


def parse_response(text, candidates):
    found = []
    for line in text.splitlines():
        m = re.match(r"\s*POI\s*:\s*([^\n]+)", line)
        if not m:
            continue
        raw = m.group(1).strip().strip(".,;:'\" ")
        if raw.lower() == "none":
            return []
        # Match against candidate list
        for c in candidates:
            if c.lower() == raw.lower():
                if c not in found:
                    found.append(c)
                break
        else:
            # Fuzzy substring match
            rl = raw.lower()
            for c in candidates:
                if c.lower() in rl or rl in c.lower():
                    if c not in found:
                        found.append(c)
                    break
    return found


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames-root", default="data/cities/zurich/frames")
    ap.add_argument("--out", default="data/cities/zurich/_video_poi_multi.jsonl")
    ap.add_argument("--vllm-url", default="http://localhost:8003/v1")
    ap.add_argument("--model", default="google/gemma-4-31b-it")
    ap.add_argument("--every-n", type=int, default=20)
    args = ap.parse_args()

    root = Path(args.frames_root)
    video_dirs = []
    if (root / "zurich").is_dir():
        video_dirs.append(("Original", root / "zurich"))
    for d in sorted(root.glob("extra_*")):
        if d.name.endswith("_dense") or not any(d.glob("*.jpg")):
            continue
        label = d.name.replace("extra_", "")[:55]
        video_dirs.append((label, d))
    print(f"[multi-scan] {len(video_dirs)} videos")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if out_path.exists():
        for ln in open(out_path):
            try:
                r = json.loads(ln)
                done.add((r["video"], r["frame_id"]))
            except Exception:
                pass
        print(f"[multi-scan] resume: {len(done)} done")

    prompt = build_prompt()
    t0 = time.time()
    n_done = 0
    with open(out_path, "a") as fout:
        for vid_label, d in video_dirs:
            frames = sorted(d.glob("*.jpg"))[::args.every_n]
            print(f"\n  [{vid_label}] {len(frames)} frames")
            for f in frames:
                fid = f.stem
                if (vid_label, fid) in done:
                    continue
                try:
                    r = requests.post(
                        f"{args.vllm_url}/chat/completions",
                        json={"model": args.model,
                              "messages": [{"role": "user", "content": [
                                  {"type": "text", "text": prompt},
                                  {"type": "image_url",
                                   "image_url": {"url": img_to_data_url(f)}},
                              ]}],
                              "max_tokens": 200, "temperature": 0.0},
                        timeout=60)
                    r.raise_for_status()
                    out = r.json()["choices"][0]["message"]["content"].strip()
                except Exception as e:
                    print(f"  err {fid}: {type(e).__name__}", file=sys.stderr)
                    continue
                pois_seen = parse_response(out, CANDIDATE_POIS)
                fout.write(json.dumps({
                    "video": vid_label, "frame_id": fid,
                    "visible_pois": pois_seen,
                }, ensure_ascii=False) + "\n")
                fout.flush()
                n_done += 1
                if n_done % 50 == 0:
                    rate = n_done / max(time.time() - t0, 1e-3)
                    print(f"    [{n_done}] rate={rate:.2f}/s")

    print(f"\n[multi-scan] {n_done} new frames ({time.time()-t0:.0f}s) → {out_path}")


if __name__ == "__main__":
    main()

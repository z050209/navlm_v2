#!/usr/bin/env python3
"""A/B test: does a Google Street View reference index recover GPS better
than the Mapillary index?

Ground truth: the trusted-starts frames already carry Phase-A verified GPS.
We embed a sample of them, match against each reference index, and measure
the haversine error of the recovered GPS vs the known coordinate.

All embeddings use DINOv2-base + avg pooling — the SAME method that built
the existing Mapillary embeddings.npz, so the comparison is fair.

Run from navlm_ss/ after torch+transformers are installed:
    .venv/Scripts/python.exe validate_streetview_match.py
"""

import json
import math
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModel

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
MODEL = "facebook/dinov2-base"
BATCH = 32
N_QUERY = 250          # trusted frames to sample as queries
TOPK = 5
CORE_BBOX = (8.539, 47.368, 8.548, 47.376)

SV_IMG_DIR = Path("data/cities/streetview/zurich/images")
SV_META = Path("data/cities/streetview/zurich/meta.jsonl")
SV_EMB = Path("data/cities/streetview/zurich/embeddings.npz")
MLY_EMB = Path("data/cities/mapillary/zurich/embeddings.npz")
TRUSTED = Path("data/cities/zurich/frame_starts_trusted_all.jsonl")
FRAMES = Path("data/cities/zurich/frames")
HTML_OUT = Path("../preview/match_grid_sv.html")

# trusted-frame `video` field -> frames/ subdir
VIDEO_DIR = {
    "zurich_main": "zurich",
    "bahnhofstrasse": "extra_Switzerland_Zurich_Bahnhofstrasse_Walking_tour_Cit",
    "most_famous": "extra_Walking_Tour_of_Switzerland_s_Most_Famous_City_Zur",
    "hidden_streets": "extra_Zurich_in_Summer_Hidden_Streets_River_Views_Swiss_",
    "saturday_morning": "extra_Zurich_looks_STUNNING_on_Saturday_Morning_Switzerl",
    "looks_perfect": "extra_Zurich_Switzerland_A_City_That_Looks_Too_Perfect_t",
    "old_town_limmat": "extra_Zurich_Switzerland_Old_Town_Limmat_River_Walking_T",
    "most_elegant": "extra_ZURICH_Switzerland_The_Most_Elegant_City_in_Europe",
}


def haversine_m(la1, lo1, la2, lo2):
    R = 6371000.0
    p1, p2 = math.radians(la1), math.radians(la2)
    dphi = math.radians(la2 - la1)
    dlam = math.radians(lo2 - lo1)
    a = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlam / 2) ** 2
    return 2 * R * math.asin(math.sqrt(a))


def load_model():
    print(f"[model] DINOv2-base on {DEVICE}")
    proc = AutoImageProcessor.from_pretrained(MODEL)
    model = AutoModel.from_pretrained(MODEL, torch_dtype=torch.float32).to(DEVICE).eval()
    return proc, model


def embed_paths(paths, proc, model):
    """Return (N,768) L2-normalised avg-pooled DINOv2 embeddings."""
    out = []
    for i in range(0, len(paths), BATCH):
        imgs = [Image.open(p).convert("RGB") for p in paths[i:i + BATCH]]
        inp = proc(images=imgs, return_tensors="pt").to(DEVICE)
        with torch.inference_mode():
            h = model(**inp).last_hidden_state
        e = torch.nn.functional.normalize(h[:, 1:].mean(dim=1), dim=-1)
        out.append(e.cpu().numpy().astype(np.float32))
        if (i // BATCH) % 10 == 0:
            print(f"  embedded {min(i + BATCH, len(paths))}/{len(paths)}")
    return np.vstack(out)


def estimate_gps(qe, ref_embs, ref_lat, ref_lon, topk):
    """For each query embedding return (est_lat, est_lon, top_idxs)."""
    sims = qe @ ref_embs.T
    idxs = np.argsort(-sims, axis=1)[:, :topk]
    est = []
    for r in range(qe.shape[0]):
        jj = idxs[r]
        est.append((float(np.median(ref_lat[jj])), float(np.median(ref_lon[jj])), jj))
    return est


def main():
    proc, model = load_model()

    # ---- reference 1: Street View ----
    sv_meta = {json.loads(l)["id"]: json.loads(l) for l in SV_META.open(encoding="utf-8")}
    sv_files = sorted(SV_IMG_DIR.glob("*.jpg"))
    print(f"[sv] embedding {len(sv_files)} Street View images")
    sv_embs = embed_paths(sv_files, proc, model)
    sv_ids = [f.stem for f in sv_files]
    sv_lat = np.array([sv_meta[i]["lat"] for i in sv_ids])
    sv_lon = np.array([sv_meta[i]["lon"] for i in sv_ids])
    np.savez_compressed(SV_EMB, ids=np.array(sv_ids), embs=sv_embs,
                        meta=np.array([json.dumps(sv_meta[i]) for i in sv_ids]),
                        method=np.array(["avg"]))
    print(f"[sv] saved {sv_embs.shape} -> {SV_EMB}")

    # ---- reference 2: Mapillary (existing) ----
    mly = np.load(MLY_EMB, allow_pickle=True)
    mly_embs = mly["embs"]
    mly_meta = [json.loads(m) for m in mly["meta"]]
    mly_ids = [str(x) for x in mly["ids"]]
    mly_lat = np.array([m["lat"] for m in mly_meta], dtype=float)
    mly_lon = np.array([m["lon"] for m in mly_meta], dtype=float)
    print(f"[mly] loaded {mly_embs.shape} reference embeddings")

    # ---- query: trusted frames in the core bbox ----
    w, s, e, n = CORE_BBOX
    cand = []
    for ln in TRUSTED.open(encoding="utf-8"):
        d = json.loads(ln)
        g = d.get("gps")
        if not g:
            continue
        lat, lon = (g if isinstance(g, list) else (g["lat"], g["lon"]))
        if w <= lon <= e and s <= lat <= n and d["video"] in VIDEO_DIR:
            fp = FRAMES / VIDEO_DIR[d["video"]] / f"{d['frame_id']}.jpg"
            if fp.exists():
                cand.append({"key": f"{d['video']}__{d['frame_id']}",
                             "path": fp, "lat": lat, "lon": lon})
    random.seed(0)
    random.shuffle(cand)
    cand = cand[:N_QUERY]
    print(f"[query] embedding {len(cand)} trusted frames (core bbox)")
    q_embs = embed_paths([c["path"] for c in cand], proc, model)

    # ---- match both ways, measure GPS error ----
    sv_est = estimate_gps(q_embs, sv_embs, sv_lat, sv_lon, TOPK)
    mly_est = estimate_gps(q_embs, mly_embs, mly_lat, mly_lon, TOPK)

    sv_err, mly_err = [], []
    rows = []
    for c, (svla, svlo, svj), (mla, mlo, mlj) in zip(cand, sv_est, mly_est):
        es = haversine_m(c["lat"], c["lon"], svla, svlo)
        em = haversine_m(c["lat"], c["lon"], mla, mlo)
        sv_err.append(es)
        mly_err.append(em)
        rows.append((c, svj, es, mlj, em))

    def stats(errs):
        a = np.array(errs)
        return (np.median(a), np.mean(a),
                100 * np.mean(a < 50), 100 * np.mean(a < 100))

    sm, sa, s50, s100 = stats(sv_err)
    mm, ma, m50, m100 = stats(mly_err)
    print("\n" + "=" * 62)
    print(f"  GPS-recovery error vs Phase-A ground truth ({len(cand)} frames)")
    print("=" * 62)
    print(f"  {'index':<14}{'median':>10}{'mean':>10}{'<50m':>9}{'<100m':>9}")
    print(f"  {'Mapillary':<14}{mm:>9.0f}m{ma:>9.0f}m{m50:>8.0f}%{m100:>8.0f}%")
    print(f"  {'StreetView':<14}{sm:>9.0f}m{sa:>9.0f}m{s50:>8.0f}%{s100:>8.0f}%")
    print("=" * 62)
    better = "StreetView" if sm < mm else "Mapillary"
    print(f"  -> {better} recovers GPS better (lower median error)\n")

    # ---- HTML grid: query | top-3 SV | top-3 Mapillary ----
    rows.sort(key=lambda r: r[2])  # best SV matches first
    show = rows[:20] + rows[-20:]
    def img(p, cap):
        return (f'<figure><img src="{p}" width="200">'
                f'<figcaption>{cap}</figcaption></figure>')
    html = ["<html><body><h2>Match grid: query frame vs Street View vs Mapillary</h2>"]
    for c, svj, es, mlj, em in show:
        html.append('<div style="display:flex;border-bottom:1px solid #ccc;padding:6px">')
        html.append('<div style="border-right:3px solid #333;padding-right:6px">')
        html.append(img(c["path"].as_posix(), f'QUERY {c["key"]}'))
        html.append('</div><div><b>StreetView</b> err={:.0f}m<br>'.format(es))
        for j in svj[:3]:
            html.append(img((SV_IMG_DIR / f"{sv_ids[j]}.jpg").as_posix(), sv_ids[j][:14]))
        html.append('</div><div><b>Mapillary</b> err={:.0f}m<br>'.format(em))
        for j in mlj[:3]:
            mp = f"data/cities/mapillary/zurich/images/{mly_ids[j]}.jpg"
            html.append(img(mp, mly_ids[j][:14]))
        html.append('</div></div>')
    html.append("</body></html>")
    HTML_OUT.write_text("\n".join(html), encoding="utf-8")
    print(f"  match grid -> {HTML_OUT.resolve()}")


if __name__ == "__main__":
    main()

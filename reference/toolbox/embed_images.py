"""Compute visual embeddings for a directory of images, saving to an .npz.

Three embedding methods are supported (ordered by VPR-accuracy in general):

  --method cls      → DINOv2 CLS token (simplest, our v1 default)
  --method avg      → DINOv2 patch-token MEAN pooling (AnyLoc-Lite recipe,
                      consistently beats CLS for place recognition)
  --method mixvpr   → MixVPR pretrained on GSV-Cities (dedicated VPR model)

Usage
-----
    # Reference: Mapillary
    python toolbox/embed_images.py \\
        --images data/mapillary/zurich/images \\
        --meta data/mapillary/zurich/meta.jsonl \\
        --out data/mapillary/zurich/embeddings.npz \\
        --method avg

    # Query: our video frames (must use the SAME method as reference)
    python toolbox/embed_images.py \\
        --images data/cities/zurich/frames/zurich \\
        --out data/cities/zurich/frames/zurich_embeddings.npz \\
        --method avg

Output
------
    .npz with fields:
        ids:   (N,) image ids
        embs:  (N, D) float32 L2-normalised embeddings
        meta:  (N,) optional JSON strings (lat/lon if --meta given)
        method: single-string attribute noting which method produced embs
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModel


def load_dinov2(name="facebook/dinov2-base", device="cuda:0"):
    proc = AutoImageProcessor.from_pretrained(name)
    model = AutoModel.from_pretrained(name, torch_dtype=torch.float32).to(device).eval()
    return proc, model


def embed_dinov2(imgs, proc, model, device, pool: str = "cls"):
    """pool: 'cls' (first token) or 'avg' (mean of patch tokens excluding CLS)."""
    inputs = proc(images=imgs, return_tensors="pt").to(device)
    with torch.inference_mode():
        out = model(**inputs)
    h = out.last_hidden_state  # (B, 1+P, D)
    if pool == "cls":
        e = h[:, 0]
    elif pool == "avg":
        e = h[:, 1:].mean(dim=1)
    else:
        raise ValueError(pool)
    e = torch.nn.functional.normalize(e, dim=-1)
    return e.cpu().numpy().astype(np.float32)


def load_mixvpr(device="cuda:0"):
    """Load a MixVPR checkpoint. Uses torch.hub → upstream repo."""
    # MixVPR exposes a helper via torch.hub; if unavailable, instruct user.
    try:
        model = torch.hub.load("amaralibey/MixVPR", "mixvpr",
                               backbone="resnet50", agg_config="dinov2_trivia",
                               pretrained=True, trust_repo=True)
    except Exception as e:
        raise SystemExit(
            f"MixVPR load failed ({e}). Install manually: "
            "git clone https://github.com/amaralibey/MixVPR and place on PYTHONPATH, "
            "then re-run this script. Falling back: use --method avg instead."
        )
    return model.to(device).eval()


def embed_mixvpr(imgs, model, device):
    import torchvision.transforms as T
    tfm = T.Compose([
        T.Resize((320, 320)),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    x = torch.stack([tfm(i) for i in imgs]).to(device)
    with torch.inference_mode():
        e = model(x)
    e = torch.nn.functional.normalize(e, dim=-1)
    return e.cpu().numpy().astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", required=True)
    ap.add_argument("--meta", default=None, help="optional JSONL w/ GPS metadata (for Mapillary)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--method", default="avg", choices=["cls", "avg", "mixvpr"],
                    help="embedding method. avg (default) is AnyLoc-Lite; "
                         "cls is the original v1 behaviour; mixvpr uses a VPR-specialised backbone.")
    ap.add_argument("--model", default="facebook/dinov2-base",
                    help="HF model id (only used for cls/avg)")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    img_dir = Path(args.images)
    files = sorted([p for p in img_dir.glob("*.jpg")]
                   + [p for p in img_dir.glob("*.jpeg")]
                   + [p for p in img_dir.glob("*.png")])
    print(f"[embed] {len(files)} images in {img_dir}   method={args.method}")

    id2gps = {}
    if args.meta:
        with open(args.meta) as f:
            for ln in f:
                r = json.loads(ln)
                id2gps[r["id"]] = (r["lat"], r["lon"])

    proc = dino = mvpr = None
    if args.method in ("cls", "avg"):
        print(f"[embed] loading DINOv2 ({args.model}) ...")
        proc, dino = load_dinov2(args.model, args.device)
    else:  # mixvpr
        print("[embed] loading MixVPR ...")
        mvpr = load_mixvpr(args.device)

    def embed_batch_dispatch(imgs):
        if args.method == "mixvpr":
            return embed_mixvpr(imgs, mvpr, args.device)
        return embed_dinov2(imgs, proc, dino, args.device, pool=args.method)

    ids, embs_list, metas = [], [], []
    t0 = time.time()
    batch, batch_files = [], []

    def flush():
        if not batch:
            return
        embs_list.append(embed_batch_dispatch(batch))
        for bf in batch_files:
            fid = bf.stem
            ids.append(fid)
            if id2gps:
                lat, lon = id2gps.get(fid, (None, None))
                metas.append(json.dumps({"id": fid, "lat": lat, "lon": lon}))
            else:
                metas.append(json.dumps({"id": fid, "file": str(bf)}))
        batch.clear()
        batch_files.clear()

    for i, f in enumerate(files):
        try:
            img = Image.open(f).convert("RGB")
        except Exception as e:
            print(f"  skip {f}: {e}")
            continue
        batch.append(img)
        batch_files.append(f)
        if len(batch) >= args.batch_size:
            flush()
            if len(ids) % (args.batch_size * 10) == 0:
                rate = len(ids) / (time.time() - t0)
                eta = (len(files) - len(ids)) / rate / 60
                print(f"  [{len(ids)}/{len(files)}] rate={rate:.1f}/s ETA={eta:.1f}min")
    flush()

    embs = np.vstack(embs_list)
    ids_arr = np.array(ids)
    metas_arr = np.array(metas)
    method_arr = np.array([args.method])
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out, ids=ids_arr, embs=embs, meta=metas_arr,
                        method=method_arr)
    print(f"[embed] saved {embs.shape} (method={args.method}) → {args.out}  "
          f"({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()

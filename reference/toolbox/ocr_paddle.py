"""Fast OCR using PaddleOCR (drop-in replacement for ocr_augment.py).

PaddleOCR is a dedicated OCR model — much smaller and faster than Gemma 31B
for text extraction (~50-200ms per frame on CPU, GPU optional).

Output JSONL is identical to ocr_augment.py so landmark_match.py works
unchanged:
    {"frame_id": "frame_00123", "texts": ["Bahnhofstrasse", "TUDOR", ...]}

Usage
-----
    python toolbox/ocr_paddle.py \\
        --frames data/cities/zurich/frames/zurich \\
        --out    data/cities/zurich/frame_ocr.jsonl

Resume support: skip frames already in the output file.
"""

import argparse
import json
import sys
import time
import warnings
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

warnings.filterwarnings("ignore")  # noisy paddle deprecation warnings


def _norm_text(t):
    """Strip whitespace + tidy. Drop very short / non-alpha noise."""
    t = (t or "").strip().strip(".,;:")
    if len(t) < 2:
        return None
    if not any(c.isalpha() for c in t):
        return None
    return t


def _extract_lines(result):
    """PaddleOCR result format varies by version. Be defensive."""
    out = []
    if not result:
        return out
    # PaddleOCR 3.x returns a list of dicts with 'rec_texts'
    for r in result:
        if isinstance(r, dict) and "rec_texts" in r:
            for t in (r.get("rec_texts") or []):
                out.append(t)
            continue
        # Older format: list[list[box, (text, conf)]]
        if isinstance(r, list):
            for item in r:
                if isinstance(item, list) and len(item) >= 2:
                    rec = item[1]
                    if isinstance(rec, (list, tuple)) and rec:
                        out.append(str(rec[0]))
                    elif isinstance(rec, str):
                        out.append(rec)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--lang", default="en",
                    help="paddleocr language (en covers Latin scripts)")
    ap.add_argument("--use-gpu", action="store_true",
                    help="run on GPU (auto-falls-back to CPU if no GPU)")
    ap.add_argument("--workers", type=int, default=4,
                    help="threads for I/O parallelism (model itself runs single-threaded)")
    ap.add_argument("--max-frames", type=int, default=None)
    args = ap.parse_args()

    from paddleocr import PaddleOCR

    print(f"[ocr-paddle] initializing PaddleOCR (lang={args.lang}, "
          f"gpu={args.use_gpu})...")
    # PaddleOCR 3.x removed use_gpu flag; runtime auto-detects via env / paddle device
    if args.use_gpu:
        import os
        os.environ.setdefault("FLAGS_use_cuda", "1")
    try:
        ocr = PaddleOCR(lang=args.lang, use_textline_orientation=False,
                         show_log=False)
    except TypeError:
        ocr = PaddleOCR(lang=args.lang)

    frames = sorted(Path(args.frames).glob("*.jpg"))
    if args.max_frames:
        frames = frames[: args.max_frames]
    print(f"[ocr-paddle] {len(frames)} frames")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if out_path.exists():
        for ln in open(out_path):
            try:
                done.add(json.loads(ln)["frame_id"])
            except Exception:
                pass
        print(f"[ocr-paddle] resume; {len(done)} already processed")

    todo = [f for f in frames if f.stem not in done]
    n_total = len(todo)
    print(f"[ocr-paddle] {n_total} to process")

    t0 = time.time()
    n_done = 0
    n_with_text = 0

    with open(out_path, "a") as fout:
        for img in todo:
            fid = img.stem
            try:
                # PaddleOCR 3.x: predict() ; older: ocr()
                if hasattr(ocr, "predict"):
                    raw = ocr.predict(str(img))
                else:
                    raw = ocr.ocr(str(img))
                lines = _extract_lines(raw)
            except Exception as e:
                print(f"  err {fid}: {type(e).__name__}: {str(e)[:80]}",
                      file=sys.stderr)
                lines = []

            texts = sorted({t for t in
                            (_norm_text(x) for x in lines) if t})
            fout.write(json.dumps({"frame_id": fid, "texts": list(texts)},
                                  ensure_ascii=False) + "\n")
            fout.flush()
            n_done += 1
            if texts:
                n_with_text += 1
            if n_done % 200 == 0:
                rate = n_done / max(time.time() - t0, 1e-3)
                eta = (n_total - n_done) / max(rate, 1e-3) / 60
                print(f"  [{n_done}/{n_total}] rate={rate:.1f}/s  "
                      f"with_text={n_with_text}  ETA={eta:.0f}min")

    print(f"[ocr-paddle] done. processed={n_done}  with_text={n_with_text}  "
          f"({time.time()-t0:.0f}s)  → {out_path}")


if __name__ == "__main__":
    main()

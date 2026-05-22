"""Ask Gemma to locate a named POI in an image and draw a bbox.

Uses Gemma's spatial-grounding ability via prompt:
    'Output the bounding box of <POI> in this photo as four numbers
     in pixel space: x1 y1 x2 y2.'

If the model can't find it (returns no bbox or 'not visible'), we just
return the original image unchanged.

Cached by (image_path, poi_name) so toggling on/off is instant after
first hit.

~3-5 s per uncached call (1 Gemma vLLM call).
"""

import base64
import io
import re
import threading
from pathlib import Path

import requests
from PIL import Image, ImageDraw, ImageFont


_BBOX_CACHE = {}
_LOCK = threading.Lock()


_PROMPT_TMPL = (
    "In this photograph, locate the building or place named {poi!r}.\n"
    "If it's clearly visible, output ONE line:\n"
    "  BBOX: x1 y1 x2 y2\n"
    "where x,y are pixel coordinates, (0,0) at top-left, image size given.\n"
    "If {poi!r} is not visible in the image, output:\n"
    "  NOT_VISIBLE\n"
    "Reply with EXACTLY one of those two formats, nothing else.\n\n"
    "(Image is {W}x{H} pixels.)"
)


def _img_data_url(path: Path):
    data = path.read_bytes()
    b64 = base64.b64encode(data).decode()
    ext = path.suffix.lower().lstrip(".")
    if ext == "jpg":
        ext = "jpeg"
    return f"data:image/{ext};base64,{b64}"


def _parse_bbox(text):
    m = re.search(r"BBOX\s*:\s*(\d+)\s+(\d+)\s+(\d+)\s+(\d+)", text)
    if not m:
        return None
    return tuple(int(x) for x in m.groups())


def find_bbox(image_path, poi_name, vllm_url, model, timeout=45):
    """Returns (x1, y1, x2, y2) in pixel space, or None."""
    key = (str(Path(image_path).resolve()), poi_name)
    with _LOCK:
        if key in _BBOX_CACHE:
            return _BBOX_CACHE[key]

    img = Image.open(image_path)
    W, H = img.width, img.height
    prompt = _PROMPT_TMPL.format(poi=poi_name, W=W, H=H)

    try:
        r = requests.post(
            f"{vllm_url}/chat/completions",
            json={
                "model": model,
                "messages": [{"role": "user", "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url",
                     "image_url": {"url": _img_data_url(Path(image_path))}},
                ]}],
                "max_tokens": 30,
                "temperature": 0.0,
            },
            timeout=timeout,
        )
        r.raise_for_status()
        out = r.json()["choices"][0]["message"]["content"].strip()
    except Exception:
        with _LOCK:
            _BBOX_CACHE[key] = None
        return None

    bbox = _parse_bbox(out)
    # Sanity check
    if bbox:
        x1, y1, x2, y2 = bbox
        if not (0 <= x1 < x2 <= W and 0 <= y1 < y2 <= H):
            bbox = None

    with _LOCK:
        _BBOX_CACHE[key] = bbox
    return bbox


def _load_font(size):
    for p in ["/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
              "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf"]:
        if Path(p).exists():
            try:
                return ImageFont.truetype(p, size)
            except Exception:
                continue
    return ImageFont.load_default()


def draw_bbox(image_path, poi_name, vllm_url, model, max_width=1024):
    """Draw the bbox on the image and return JPEG bytes."""
    bbox = find_bbox(image_path, poi_name, vllm_url, model)
    img = Image.open(image_path).convert("RGB")
    orig_w, orig_h = img.width, img.height
    if img.width > max_width:
        new_h = int(orig_h * max_width / orig_w)
        img = img.resize((max_width, new_h), Image.LANCZOS)

    if bbox is not None:
        # Scale bbox to resized image
        x1, y1, x2, y2 = bbox
        scale = img.width / orig_w
        x1, y1, x2, y2 = [int(v * scale) for v in (x1, y1, x2, y2)]
        d = ImageDraw.Draw(img)
        # Outer + inner stroke for visibility
        for w, color in [(8, (0, 0, 0)), (4, (50, 220, 110))]:
            d.rectangle([x1, y1, x2, y2], outline=color, width=w)
        # Label tag at top-left of box
        font = _load_font(28)
        label = poi_name
        tb = d.textbbox((0, 0), label, font=font)
        tw, th = tb[2] - tb[0], tb[3] - tb[1]
        d.rectangle([x1, y1 - th - 12, x1 + tw + 16, y1],
                    fill=(50, 220, 110))
        d.text((x1 + 8, y1 - th - 6), label, fill=(0, 0, 0), font=font)
    else:
        # Failed to locate — leave image untouched but draw a small banner
        d = ImageDraw.Draw(img)
        font = _load_font(24)
        msg = f"{poi_name}: not visible"
        tb = d.textbbox((10, 10), msg, font=font)
        d.rectangle([8, 8, tb[2] + 8, tb[3] + 8], fill=(60, 60, 60))
        d.text((14, 12), msg, fill=(255, 220, 100), font=font)

    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85, optimize=True)
    return buf.getvalue()

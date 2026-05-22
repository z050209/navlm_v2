"""Draw a direction arrow on a frame.

Given:
  - The frame image
  - first_action  (continue ahead / turn left / turn right / turn around)
  - user_heading  (compass deg, optional)
  - dest_bearing  (compass deg from user → destination, optional)

Pure PIL — no model call, no GPU. ~10 ms per image.

Used by:
  - synth_viewer.py /image_arrow/<idx> endpoint
  - eval visualizations
"""

import io
import math
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


# Map the 5 first_action verbs to a screen-space angle in degrees,
# where 0 = up (12 o'clock), 90 = right, 180 = down, 270 = left.
ACTION_TO_ANGLE = {
    "continue ahead":            0,
    "continue":                  0,
    "turn left":                -65,
    "turn right":                65,
    "turn around":              175,
    "go back the way you came": 175,
    "go back":                  175,
    "arrive":                    0,
}


def _load_font(size=42):
    for path in [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
        "/usr/share/fonts/TTF/DejaVuSans-Bold.ttf",
    ]:
        if Path(path).exists():
            try:
                return ImageFont.truetype(path, size)
            except Exception:
                continue
    return ImageFont.load_default()


def screen_angle_for(first_action, user_heading=None, dest_bearing=None):
    """Choose the angle to draw.

    Priority:
      1. If user_heading + dest_bearing both given → exact relative angle
      2. else → look up canonical angle by first_action verb
    """
    if user_heading is not None and dest_bearing is not None:
        # Camera "ahead" is the user's heading (compass). Destination is at
        # dest_bearing (compass). Screen-space angle = bearing - heading,
        # normalised to (-180, 180].
        rel = ((dest_bearing - user_heading + 540) % 360) - 180
        return rel
    return ACTION_TO_ANGLE.get((first_action or "").strip().lower(), 0)


def draw_arrow(image_path, first_action, user_heading=None, dest_bearing=None,
                label=None, max_width=1024):
    """Return JPEG bytes of the image with a direction arrow drawn on it."""
    img = Image.open(image_path).convert("RGB")
    if img.width > max_width:
        new_h = int(img.height * max_width / img.width)
        img = img.resize((max_width, new_h), Image.LANCZOS)

    W, H = img.width, img.height
    angle = screen_angle_for(first_action, user_heading, dest_bearing)

    # Translucent overlay so we can compose the arrow on top
    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(overlay)

    # Arrow geometry
    cx = W // 2
    cy = int(H * 0.78)                    # anchor near bottom-centre
    length = int(min(W, H) * 0.22)
    head_size = int(length * 0.42)

    rad = math.radians(angle)
    # in PIL coords, +y is down; arrow points "up" by default → rotate
    end_x = cx + int(length * math.sin(rad))
    end_y = cy - int(length * math.cos(rad))

    # Big translucent shaft
    d.line([(cx, cy), (end_x, end_y)], fill=(255, 80, 80, 230),
            width=18)
    # Arrow head: triangle at end_x, end_y
    head_angle = rad
    left_x = end_x + int(head_size * math.sin(head_angle - 2.5))
    left_y = end_y - int(head_size * math.cos(head_angle - 2.5))
    right_x = end_x + int(head_size * math.sin(head_angle + 2.5))
    right_y = end_y - int(head_size * math.cos(head_angle + 2.5))
    d.polygon([(end_x, end_y), (left_x, left_y), (right_x, right_y)],
              fill=(255, 80, 80, 240))

    # Text label below arrow base
    label = label or first_action or ""
    if label:
        font = _load_font(38)
        # Background pill for readability
        bbox = d.textbbox((0, 0), label, font=font)
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
        pad = 12
        bx0 = cx - tw // 2 - pad
        by0 = cy + 14
        bx1 = cx + tw // 2 + pad
        by1 = cy + 14 + th + pad
        d.rounded_rectangle([bx0, by0, bx1, by1], radius=10,
                             fill=(0, 0, 0, 180))
        d.text((cx - tw // 2, by0 + pad // 2), label,
                fill=(255, 255, 255, 255), font=font)

    img = Image.alpha_composite(img.convert("RGBA"), overlay).convert("RGB")
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85, optimize=True)
    return buf.getvalue()

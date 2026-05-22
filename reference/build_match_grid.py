"""Build an HTML grid comparing trusted frames with their GPS-nearest Mapillary matches.

For each trusted frame, finds the top-3 closest Mapillary images by GPS distance
and creates a side-by-side comparison grid.

Usage:
    python build_match_grid.py
    # Then open match_grid.html in a browser
"""

import json
import math
import os
from pathlib import Path

# --- Paths (adjust as needed) ---
TRUSTED = "C:/Users/Admin/Desktop/navlm_ss/navlm_ss/data/cities/zurich/frame_starts_trusted_all.jsonl"
MLY_META = "C:/Users/Admin/Desktop/navlm_ss/navlm_ss/data/cities/mapillary/zurich/meta.jsonl"
FRAMES_BASE = "C:/Users/Admin/Desktop/navlm_ss/navlm_ss/data/cities/zurich/frames"
MLY_IMAGES = "C:/Users/Admin/Desktop/navlm_ss/navlm_ss/data/cities/mapillary/zurich/images"
OUT_HTML = "G:/My Drive/cs231n/project/cs231n/cs231n/navlm_ss/match_grid.html"

# How many trusted frames to include (set None for all)
MAX_FRAMES = 200  # first 200 for quick review
TOP_K = 3  # top-K nearest Mapillary matches

def haversine_m(lat1, lon1, lat2, lon2):
    R = 6371000
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat/2)**2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon/2)**2
    return R * 2 * math.asin(math.sqrt(a))

def angle_diff(a, b):
    d = (a - b) % 360
    return d if d <= 180 else 360 - d

# Load Mapillary
print("Loading Mapillary metadata...")
mly = []
with open(MLY_META) as f:
    for line in f:
        r = json.loads(line)
        mly.append(r)
print(f"  {len(mly)} Mapillary images")

# Load trusted frames
print("Loading trusted frames...")
trusted = []
with open(TRUSTED) as f:
    for line in f:
        trusted.append(json.loads(line))
print(f"  {len(trusted)} trusted frames")

if MAX_FRAMES:
    # Sample evenly across videos
    from collections import defaultdict
    by_video = defaultdict(list)
    for t in trusted:
        by_video[t['video']].append(t)
    sampled = []
    per_video = MAX_FRAMES // len(by_video)
    for vid, frames in sorted(by_video.items()):
        step = max(1, len(frames) // per_video)
        sampled.extend(frames[::step][:per_video])
    trusted = sampled[:MAX_FRAMES]
    print(f"  Sampled {len(trusted)} for grid")

# For each trusted frame, find top-K Mapillary by GPS
print("Finding GPS-nearest matches...")
rows = []
for i, tf in enumerate(trusted):
    lat, lon = tf['gps']
    heading = tf['heading']

    # Compute distance to all Mapillary
    dists = []
    for m in mly:
        d = haversine_m(lat, lon, m['lat'], m['lon'])
        hdiff = angle_diff(heading, m['compass_angle'])
        dists.append((d, hdiff, m))

    dists.sort(key=lambda x: x[0])
    topk = dists[:TOP_K]

    # Resolve frame image path
    img_path = tf['image']
    parts = img_path.split('/frames/')[-1]
    video_dir, fname = parts.split('/')
    local_frame = os.path.join(FRAMES_BASE, video_dir, fname)

    rows.append({
        'frame': tf,
        'local_frame': local_frame,
        'video_dir': video_dir,
        'fname': fname,
        'matches': [(d, hdiff, m) for d, hdiff, m in topk],
    })

    if (i+1) % 50 == 0:
        print(f"  {i+1}/{len(trusted)}...")

# Build HTML
print("Building HTML...")
html_parts = ["""<!DOCTYPE html>
<html><head><meta charset="utf-8">
<title>NavLM: Trusted Frame vs Mapillary Match Grid</title>
<style>
body { font-family: Arial, sans-serif; background: #f5f5f5; margin: 20px; }
h1 { color: #1a1a1a; }
.stats { background: #e9edf4; padding: 10px 15px; border-radius: 6px; margin-bottom: 20px; }
.row { display: flex; gap: 10px; margin-bottom: 15px; background: white;
       padding: 10px; border-radius: 8px; box-shadow: 0 1px 3px rgba(0,0,0,0.1); }
.cell { text-align: center; }
.cell img { width: 280px; height: auto; border-radius: 4px; }
.cell.frame img { border: 3px solid #1f3a68; }
.cell.match img { border: 2px solid #ccc; }
.label { font-size: 11px; color: #555; margin-top: 4px; }
.good { color: #2d7d2d; font-weight: bold; }
.ok { color: #b58900; }
.bad { color: #cc3333; }
.video-header { background: #1f3a68; color: white; padding: 8px 15px;
                border-radius: 6px; margin: 20px 0 10px 0; }
</style>
</head><body>
<h1>Trusted Frame vs Mapillary GPS-Match Grid</h1>
<div class="stats">
<b>Total trusted frames:</b> 2,177 | <b>Showing:</b> """ + str(len(rows)) + """ |
<b>Mapillary index:</b> 5,000 images | <b>Matches:</b> Top-3 by GPS distance<br>
<b>Color coding:</b> <span class="good">&lt;20m = good</span> |
<span class="ok">20-50m = ok</span> | <span class="bad">&gt;50m = poor</span>
</div>
"""]

current_video = None
for row in rows:
    vid = row['frame']['video']
    if vid != current_video:
        current_video = vid
        html_parts.append(f'<div class="video-header">Video: {vid}</div>')

    tf = row['frame']
    lat, lon = tf['gps']
    heading = tf['heading']
    sim = tf['evidence'].get('top1_sim', 0)

    html_parts.append('<div class="row">')

    # Trusted frame (use file:// for local viewing)
    frame_path = row['local_frame'].replace('\\', '/')
    html_parts.append(f'''<div class="cell frame">
        <img src="file:///{frame_path}" loading="lazy">
        <div class="label"><b>Trusted Frame</b><br>
        {row["fname"]} ({vid})<br>
        GPS: {lat:.5f}, {lon:.5f}<br>
        Heading: {heading:.1f}° | Sim: {sim:.3f}</div>
    </div>''')

    # Top-K matches
    for dist, hdiff, m in row['matches']:
        mly_img = os.path.join(MLY_IMAGES, f"{m['id']}.jpg").replace('\\', '/')
        dist_class = 'good' if dist < 20 else ('ok' if dist < 50 else 'bad')
        hdiff_class = 'good' if hdiff < 30 else ('ok' if hdiff < 60 else 'bad')

        html_parts.append(f'''<div class="cell match">
            <img src="file:///{mly_img}" loading="lazy">
            <div class="label">Mapillary {m["id"][:8]}...<br>
            Dist: <span class="{dist_class}">{dist:.0f}m</span> |
            Heading diff: <span class="{hdiff_class}">{hdiff:.0f}°</span><br>
            Compass: {m["compass_angle"]:.0f}°</div>
        </div>''')

    html_parts.append('</div>')

html_parts.append('</body></html>')

with open(OUT_HTML, 'w', encoding='utf-8') as f:
    f.write('\n'.join(html_parts))

print(f"Done! Open: {OUT_HTML}")
print(f"  {len(rows)} rows written")

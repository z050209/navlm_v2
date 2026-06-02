# DEV_MANUAL_v2 — Attempt 2 reference

Single-document reference for the Attempt 2 pipeline. Covers the
re-design from the destination-naming bug in Attempt 1 through to the
network-routed annotation ground truth currently being computed.

Every script lives under `src/a2_*.py`. Every data output lives under
`data/cities/zurich/a2/*.jsonl`. Every visualisation lives under
`viz/a2_*.html`. The constant `ATTRACTIONS_21` (the 21-attraction
vocabulary) is defined in `src/a2_attraction_slots.py` and imported by
every other a2_ script.

---

## 1. Why Attempt 2 — the destination-naming bug

Attempt 1 used the **top-30 OSM place names from the cohort** as the
destination pool. The vast majority were **streets** (Bahnhofstrasse,
Storchengasse, Limmatquai, Münstergasse, …) because
`gps_recovery`'s `nearest_poi_m` step picks the *polygon-distance*
nearest OSM feature — and the walker is always standing on a street
polyline (distance = 0 by definition). So the model was trained on
`dest_name = "Storchengasse"` — not how a tourist asks for directions.

The bug surfaces at inference: the model has only ever seen street
tokens as destinations, so `"navigate to Grossmünster"` is
out-of-distribution.

### The reframing — source vs destination

Walking-tour frames are taken **on the way to a destination**. The
walker's GPS is just *where they happen to be on the path* — usually
on a street. That street-name location tag is **not wrong** — it
correctly identifies where the walker is right now.

The error in attempt 1 was using that same street name as the
**destination**. Source ≠ destination:

| | What it is | Where the data comes from |
|---|---|---|
| **Source location** | Where the walker IS right now | DINOv2 GPS → nearest OSM POI; usually a street; fine as-is |
| **Destination** | Where the walker is heading | One of a small fixed list of **famous attractions** the walker would actually ask for |

Attempt 2 keeps the source-location pipeline unchanged and **changes
only the destination vocabulary** to the 21 curated attractions
below.

---

## 2. The 21 famous attractions

Sourced from three authoritative tourism resources, cross-referenced
with VLM coverage, geographically filtered to the central
walking-tour zone. See `src/a2_attraction_slots.py` for the
machine-readable list.

| # | English | 中文 | Kind | Public sources |
|---:|---|---|---|---|
| 1 | Grossmünster | 大教堂 | church | zuerich.com · PlanetWare · myswitzerland |
| 2 | Fraumünster | 圣母大教堂 | church | zuerich.com · PlanetWare · myswitzerland |
| 3 | St. Peter | 圣彼得教堂 | church | search snippets · PlanetWare |
| 4 | Wasserkirche | 水教堂 | church | myswitzerland |
| 5 | Lindenhof | 林登霍夫山丘 | hill | zuerich.com · PlanetWare · myswitzerland |
| 6 | Niederdorfstrasse | 下村街 | street | zuerich.com · PlanetWare · myswitzerland |
| 7 | Bahnhofstrasse | 班霍夫大街 | street | zuerich.com · PlanetWare |
| 8 | Lake Zurich (Zürichsee) | 苏黎世湖 | water | zuerich.com · PlanetWare |
| 9 | Limmat River | 利马特河 | water | PlanetWare · myswitzerland |
| 10 | Landesmuseum | 瑞士国家博物馆 | museum | zuerich.com · PlanetWare |
| 11 | Kunsthaus | 苏黎世美术馆 | museum | zuerich.com · PlanetWare |
| 12 | Opernhaus | 苏黎世歌剧院 | culture | zuerich.com |
| 13 | Bürkliplatz (Ganymede / lake viewpoint) | 比尔克利广场 | square | zuerich.com |
| 14 | Helmhaus | 赫尔姆豪斯 | civic | myswitzerland |
| 15 | Hauptbahnhof | 苏黎世中央车站 | station | search snippets |
| 16 | Münsterhof | 明斯特霍夫广场 | square | navigation anchor |
| 17 | Paradeplatz | 阅兵广场 | square | navigation anchor |
| 18 | Rathaus | 市政厅 | civic | navigation anchor |
| 19 | Münsterbrücke | 大教堂桥 | bridge | navigation anchor |
| 20 | Limmatquai | 利马特河滨道 | street | navigation anchor |
| 21 | Sechseläutenplatz | 六鸣节广场 | square | navigation anchor |

Sources:
- [Zürich Tourism — Top 10](https://www.zuerich.com/en/sightseeing-activities/places-to-visit/top-10-places-to-visit)
- [PlanetWare — Top 12](https://www.planetware.com/2037820/zurich-switzerland-most-popular-tourist-attractions-worth-visiting/)
- [Switzerland Tourism — Old Town](https://www.myswitzerland.com/en-us/experiences/zurichs-old-town/)

Each attraction has an alias table (`ALIASES` in `src/a2_attraction_slots.py`)
covering variants the VLM produces:
- `St. Peter` ← `Kirche St. Peter`, `St. Peter Church`, `St. Peter's Church`, `St. Peterkirche`, `St. Peterhofstatt`
- `Hauptbahnhof` ← `Zürich Hauptbahnhof`, `Zurich Main Station`, `Main Station`, `Zurich HB`
- `Lake Zurich` ← `Zürichsee`
- `Limmat river` ← `Limmat`
- `Landesmuseum` ← `Swiss National Museum`
- `Opernhaus` ← `Opernhaus Zürich`, `Zurich Opera House`
- `Kunsthaus` ← `Kunsthaus Zürich`
- `Stadthaus` (not in 21) ← `Stadthausquai`

OSM also misses two of our 21 in `pois.json` (Paradeplatz, Rathaus —
OSM has them but `src/pois.py` POINT_TAGS filter dropped them). They
are restored via a supplementary file:

**`data/cities/zurich/a2/extra_pois.json`** — 2 manually-curated OSM
entries, GPS verified against Nominatim, auto-merged at read time by
STEP 1.

---

## 3. Pipeline overview

```
GPS-side                  VLM-side
─────────                 ─────────
gps_recovery_full.jsonl   poi_scan.jsonl + poi_scan_cos0.75.jsonl
       ↓                          ↓
 (STEP 1) a2_step1         (STEP 2) a2_step2
 GPS_GEO.jsonl             VLM_GEO.jsonl
       └────────┬───────────────┘
                ↓
      (STEP 3) a2_step3
      GPS_VLM_GEO.jsonl
                ↓
        matched cohort: 1,219
                ↓
      (STEP 4) a2_viz_matched
      viz/a2_vlmagreed.html   (visual QC)
                ↓
      a2_target_frames
      target_attraction_frames.jsonl   (per-attraction destination pool)
                ↓
                ↓ + heading_v2 (a2_heading_v2)
                ↓ + road_snapped_a2 (road_snap, re-run)
                ↓ + destination_targets (a2_destination_targets)
                ↓
      (current) a2_route
      routes.jsonl   (network-routed bearing + GT verbs)
                ↓
                (re-annotate with Gemini Pro 2.5 — pending)
                (re-train L-given LoRA — pending)
```

---

## 4. STEP 1 — `GPS_GEO.jsonl` (GPS-side candidates)

**Script**: `src/a2_step1_gps_geo.py`
**Output**: `data/cities/zurich/a2/GPS_GEO.jsonl` — **15,053 rows**

For every DINOv2-accepted frame in `gps_recovery_full.jsonl` (both
tiers), derive three lists from GPS + OSM only (no VLM):

| List | Source | Typical size per frame |
|---|---|---:|
| `attractions_within_R` | 21-list, hand-curated GPS within R metres of frame's `g_dino` | 0–7 |
| `landmarks_within_R` | OSM POIs with landmark-class `osm_kind` (tourism/historic/place_of_worship/etc.) within R | 1–10 |
| `pois_within_R` | every OSM POI of any kind (incl. streets) within R | 5–30 |

Default radius **R = 100 m**. The OSM table is `pois.json` merged with
`a2/extra_pois.json` (adds Paradeplatz, Rathaus).

The merged STEP 1 also appends three fields from `heading_v2.jsonl`
(§7): `heading_v2`, `heading_v2_decision`, `heading_v2_gap`.

### Run
```bash
python -m src.a2_step1_gps_geo --radius 100
```

### Coverage at radius=100m
```
frames with ≥1 attraction (21-list) nearby:  8,186  (54 %)
frames with ≥1 OSM landmark nearby:         14,621  (97 %)
frames with ≥1 OSM POI of any kind:         14,994  (99.6 %)
```

---

## 5. STEP 2 — `VLM_GEO.jsonl` (VLM-side candidates)

**Script**: `src/a2_step2_vlm_geo.py`
**Output**: `data/cities/zurich/a2/VLM_GEO.jsonl` — **4,891 rows**

For every VLM-scanned frame (merged + deduped from
`poi_scan.jsonl` 872 + `poi_scan_cos0.75.jsonl` 4,101 = 4,891 unique
frames), derive three lists from VLM output only (no GPS):

| List | Source |
|---|---|
| `attractions_from_vlm` | 21-list names found in `visible[]` or `guess` (after fold + alias) |
| `landmarks_from_vlm` | OSM landmark-class names found in `visible[]` or `guess` |
| `pois_from_vlm` | any OSM POI name found in `visible[]` or `guess` |

Compound VLM strings (`"Bahnhofstrasse am Paradeplatz"`,
`"Limmat | Limmatquai"`) are split on `/`, `,`, `|`, ` am `, ` at `,
` near ` before lookup. Each entry tracks `source =
visible / guess / both`.

### Run
```bash
python -m src.a2_step2_vlm_geo
```

### Coverage
```
4,891 VLM-scanned frames
  with ≥1 attraction (21-list) named by VLM:  2,366  (48 %)
  with ≥1 OSM landmark named by VLM:          1,437  (29 %)
  with ≥1 OSM POI (any kind) named by VLM:    4,690  (96 %)
```

---

## 6. STEP 3 — `GPS_VLM_GEO.jsonl` (per-frame coincidence match)

**Script**: `src/a2_step3_gps_vlm_geo.py`
**Output**: `data/cities/zurich/a2/GPS_VLM_GEO.jsonl` — **4,158 rows at cos≥0.75**

Per-frame match between GPS_GEO and VLM_GEO lists. A frame matches
iff **at least one name from the GPS-side union coincides with at
least one name from the VLM-side union**. Three coincidence types
(give space for affiliation, no neighborhood-distance check):

| Type | Rule | Example |
|---|---|---|
| `exact` | `fold(a) == fold(b)` | "Grossmünster" ↔ "grossmunster" |
| `substring` | `fold(a)` ⊂ `fold(b)` or vice-versa | "Grossmünster" ⊂ "Grossmünsterplatz"; "Limmat" ⊂ "Limmatquai"; "Stadthaus" ⊂ "Stadthausquai" |
| `word_share` | both names share a meaningful word (≥4 chars, not common) | "Hotel Storchen" / "Storchengasse" share "storchen" |

Each row carries `best_level`:
- **attraction** — at least one match has a 21-list canonical name on either side
- **landmark** — match on an OSM landmark POI name
- **poi** — only on an OSM POI name (any kind)

### Cohort-shaping flags
```
--cos-min 0.75                — DINOv2 cosine threshold
--drop-ambiguous-heading      — also drop heading_v2='ambiguous' frames
```

### Run
```bash
python -m src.a2_step3_gps_vlm_geo --cos-min 0.75
```

### Headline at cos≥0.75
```
union of frames (either side):       4,158
matchable (in both filtered sets):   2,527
MATCHED:                             1,219  (29 % of union, 48 % of matchable)
   by best_level:
      attraction:                      774  (63 % of matched)
      landmark:                        133
      poi:                             312
unmatched (in both, no coincidence):  1,308  (suspect lookalikes)
```

### Match-type breakdown
```
exact         1,378
substring       487   ← affiliation matches (Limmat ⊂ Limmatquai, etc.)
word_share       45
```

---

## 7. heading_v2 — gap-tiered heading

**Script**: `src/a2_heading_v2.py`
**Output**: `data/cities/zurich/a2/heading_v2.jsonl` — **15,053 rows**

Replaces `gps_recovery`'s all-4-cosine-weighted heading with a
gap-tiered rule based on DINOv2 cosine difference between top-1 and
top-2 compass crops at the same pano:

```
gap = sims[0] − sims[1]               # absolute, not relative

gap > 0.20         →  decision = "top1"
                       heading = top-1 crop's compass_angle
                       (one direction clearly dominant)

0.00 < gap ≤ 0.20  →  decision = "top1+top2"
                       heading = cosine-weighted circular mean of
                                 top-1 and top-2 angles only
                       (two adjacent directions both strong)
```

The user opted not to drop the gap ≤ 0.05 frames — they go into the
`top1+top2` bucket. The `--lo 0.05` flag still produces the
3-tier rule with an `ambiguous` bucket if needed.

### Run
```bash
python -m src.a2_heading_v2 --hi 0.20 --lo 0.0
```

### Distribution
```
top1        (gap > 0.20)    4,088 frames  (27.2 %)
top1+top2   (gap ≤ 0.20)   10,965 frames  (72.8 %)
ambiguous                       0
```

v1 vs v2 heading agreement:
```
< 5° apart    : 7,009 (47 %)
5-15°         : 2,316 (15 %)
15-45°        : 3,338 (22 %)
≥ 45° apart   : 2,390 (16 %)
```

The new fields get merged back into `GPS_GEO.jsonl` (re-run STEP 1
after generating `heading_v2.jsonl`).

---

## 8. HMM road-snap (re-run for the matched cohort)

**Script**: `src/road_snap.py` (existing from Attempt 1)
**Output**: `data/cities/zurich/a2/road_snapped_a2.jsonl` — **2,470 rows**

Original `road_snapped.jsonl` from Attempt 1 was filtered to
`--tier 1 --top-pois 30` and covers only 1,072 of 1,219 matched-cohort
frames. We re-run with `--tier 1 --top-pois 0` (no POI filter), so all
2,470 VLM-confirmed accepted frames get an HMM-snapped GPS +
segment_bearing.

### Run
```bash
python -m src.road_snap --tier 1 --top-pois 0 \
       --output data/cities/zurich/a2/road_snapped_a2.jsonl
# (file ends up under data/cities/zurich/road_snapped_a2.jsonl due
#  to road_snap.py path handling; manually mv into a2/)
```

### Coverage of matched cohort
```
matched cohort:                            1,219
matched ∩ road_snapped_a2:                 1,215  (99.7 %)
matched missing:                               4
```

### Per-row schema
- `gps_snapped` — on-the-road GPS
- `segment_id` — OSM walking-graph edge
- `segment_bearing` — compass direction of the edge
- `segment_length_m` — edge length
- `snap_offset_m` — distance from raw GPS to road

The HMM output is used for:
1. **Cleaner source-GPS** when computing the routing (we use `gps_snapped` as the walker's position)
2. **Heading QC** (existing pipeline drops frames whose heading disagrees with `segment_bearing`)
3. **Future**: per-frame route visualisation

---

## 9. Per-attraction frame counts — `target_attraction_frames.jsonl`

**Script**: `src/a2_target_frames.py`
**Output**: `data/cities/zurich/a2/target_attraction_frames.jsonl` — **21 rows**

For each attraction, collect the matched-cohort frames that represent
it (any `frame_attractions(r)` membership). One frame can represent
multiple attractions in dense old-town clusters.

| # | Attraction | 中文 | Kind | Frames | Panos | Attr-L | Land-L | POI-L |
|---:|---|---|---|---:|---:|---:|---:|---:|
| 1 | Grossmünster | 大教堂 | church | 118 | 20 | 100 | 4 | 14 |
| 2 | Fraumünster | 圣母大教堂 | church | 282 | 20 | 200 | 13 | 69 |
| 3 | St. Peter | 圣彼得教堂 | church | 347 | 22 | 179 | 63 | 105 |
| 4 | Wasserkirche | 水教堂 | church | 52 | 8 | 49 | 0 | 3 |
| 5 | Lindenhof | 林登霍夫山丘 | hill | 67 | 9 | 29 | 2 | 36 |
| 6 | Niederdorfstrasse | 下村街 | street | 108 | 12 | 48 | 29 | 31 |
| 7 | Bahnhofstrasse | 班霍夫大街 | street | 212 | 10 | 198 | 1 | 13 |
| 8 | Lake Zurich | 苏黎世湖 | water | 38 | 10 | 35 | 1 | 2 |
| 9 | Limmat river | 利马特河 | water | 312 | 15 | 221 | 60 | 31 |
| 10 | Landesmuseum | 瑞士国家博物馆 | museum | 11 | 3 | 7 | 0 | 4 |
| 11 | Kunsthaus | 苏黎世美术馆 | museum | **1** | 1 | 0 | 0 | 1 |
| 12 | Opernhaus | 苏黎世歌剧院 | culture | 16 | 2 | 16 | 0 | 0 |
| 13 | Bürkliplatz | 比尔克利广场 | square | **1** | 1 | 1 | 0 | 0 |
| 14 | Helmhaus | 赫尔姆豪斯 | civic | 17 | 7 | 14 | 0 | 3 |
| 15 | Hauptbahnhof | 苏黎世中央车站 | station | 106 | 8 | 105 | 1 | 0 |
| 16 | Münsterhof | 明斯特霍夫广场 | square | 299 | 9 | 152 | 61 | 86 |
| 17 | Paradeplatz | 阅兵广场 | square | **1** | 1 | 1 | 0 | 0 |
| 18 | Rathaus | 市政厅 | civic | 238 | 10 | 157 | 55 | 26 |
| 19 | Münsterbrücke | 大教堂桥 | bridge | 187 | 12 | 181 | 5 | 1 |
| 20 | Limmatquai | 利马特河滨道 | street | 158 | 17 | 109 | 12 | 37 |
| 21 | Sechseläutenplatz | 六鸣节广场 | square | 16 | 2 | 16 | 0 | 0 |
| | unique matched | | | **1,219** | **89** | | | |

**18 attractions have ≥10 matched frames** (viable target set).
**3 attractions have only 1 frame** (Kunsthaus, Bürkliplatz, Paradeplatz)
— too thin; pending decision: drop, augment with SV crops, or accept.

---

## 10. `destination_targets.jsonl` — HYBRID routing targets

**Script**: `src/a2_destination_targets.py`
**Output**: `data/cities/zurich/a2/destination_targets.jsonl` — **21 rows**

For each attraction, decide ONE canonical routing target:

| target_type | applies to | Rule |
|---|---|---|
| **point** | 16 attractions (kind ∉ {water, street}) | Snap canonical GPS to nearest walking-graph node (single point) |
| **multi** | 5 long features (kind ∈ {water, street}: Lake Zurich, Limmat river, Bahnhofstrasse, Niederdorfstrasse, Limmatquai) | List of walking-graph nodes — one per matched-cohort pano tagged with this attraction, PLUS the canonical-centroid node as a fallback; "arrived" = reach ANY node in the list |

The multi-target candidate pool is built as:

```
multi_targets =
    { nearest_walking_node(pano.gps_snapped or pano.g_dino)
      for every pano in GPS_VLM_GEO where matched==True and the pano
      was tagged with this attraction }
  ∪
    { nearest_walking_node(canonical_lat_lon) }     ← +1 fallback row,
                                                      stored with
                                                      video="<canonical>"
```

The fallback exists so that if the matched cohort missed an obvious
section (e.g. every Lake-Zurich pano happens to be on the west shore),
routing can still land on the curated centroid.

### Per-row schema

```jsonl
// POINT target
{"attraction": "Grossmünster", "kind": "church",
 "target_type": "point",
 "canonical_gps": [47.37018, 8.54425],
 "snapped_node_id": 12345678,
 "snapped_gps":     [47.37025, 8.54420],
 "snap_offset_m":   29.2,
 "is_routable":     true}

// MULTI target
{"attraction": "Lake Zurich", "kind": "water",
 "target_type": "multi",
 "n_targets": 39,
 "multi_targets": [
    {"video": "<video>", "frame_id": "<fid>",
     "node_id": 12345678, "gps": [...], "snap_offset_m": 8.4},
    ...
    {"video": "<canonical>", "frame_id": "<canonical>",
     "node_id": 87654321, "gps": [<curated lat,lon>],
     "snap_offset_m": 12.1}    ← centroid fallback row
 ]}
```

### Run
```bash
python -m src.a2_destination_targets
```

### Snap-offset distribution

POINT targets (all 16 within 60 m of nearest walking node — clean):
```
Paradeplatz             0.5 m   ✓
Bürkliplatz             4.4 m
Opernhaus               4.3 m
Rathaus                 9.0 m
Fraumünster            10.3 m
St. Peter              13.6 m
Münsterbrücke          13.2 m
Münsterhof             14.1 m
Lindenhof              20.9 m
Helmhaus               28.1 m
Hauptbahnhof           32.4 m
Sechseläutenplatz      29.7 m
Wasserkirche           27.3 m
Landesmuseum           19.2 m
Grossmünster           29.2 m
Kunsthaus              52.4 m   ← worst (no walking node closer)
```

MULTI targets (one per matched-cohort pano tagged with the attraction):
```
Limmat river       313 targets
Bahnhofstrasse     213 targets
Limmatquai         159 targets
Niederdorfstrasse  109 targets
Lake Zurich         39 targets
```

The OSM walking graph (`osm_walking.pkl`) is **UTM-projected (EPSG:32632)**,
so all nearest-node queries project lat/lon through
`pyproj.Transformer` before calling `osmnx.distance.nearest_nodes`.
Node coordinates are converted back to lat/lon for bearing computation.

---

## 11. `routes.jsonl` — band-sampled (frame, destination) pairs with GT verbs

**Script**: `src/a2_route.py`
**Output**: `data/cities/zurich/a2/routes.jsonl` — **~3,657 rows**
**Status**: currently being computed

### Sampling rule per matched frame (1,219 frames)

For each of N=3 destination slots:

```
1. Roll a band:  80% near    (50 - 500 m straight-line from frame)
                 10% medium   (500 - 1000 m)
                 10% far      (1000 - 1500 m)
2. PURE RANDOM among the 21 attractions in that band.
3. RE-ROLL the band if the sampled destination duplicates an earlier
   slot for THIS frame (up to 5 retries; drop slot if still dup).
4. Fall back to any attraction in 50-1500m if the band has zero
   candidates (rare; happens at frames at the edge of the cohort).
```

Random seed: `42` (reproducible).

### Routing per (frame, destination)

```
src_node = nearest walking node to frame.gps_snapped
target   = destination_targets row (point or multi)

if point:
   path = nx.shortest_path(G, src_node, target.snapped_node_id, weight='length')
if multi:
   # run Dijkstra to EVERY candidate node in the list, keep the
   # path with the smallest total edge-length:
   for t in target.multi_targets:
       p_t = nx.shortest_path(G, src_node, t.node_id, weight='length')
       len_t = sum(edge.length for edge in p_t)
   path = argmin_t len_t

route_bearing_network = bearing( lat-lon(path[0]) → lat-lon(path[1]) )
route_distance_m      = sum of all edge lengths in path
first_segment_length_m = length of path[0]→path[1]
```

**Multi-target "nearest" is network walking distance, not straight-line.**
The router does N Dijkstra calls (N = `n_targets`, up to 313 for the
Limmat river) and keeps the path with the smallest total `length`-summed
edge weight. This matters in Zurich because of the river + limited
bridges: a Limmat-river candidate 80 m straight-line away on the far
bank can be 400 m of walking via the nearest bridge, so the router will
correctly prefer a same-bank candidate 300 m upstream over it, and the
GT verb will reflect "continue ahead along this bank" rather than a
spurious "turn right toward the bridge".

The **distance-band sampling** in `routes.jsonl` (the 80 % near /
10 % mid / 10 % far cohort) is the only place that uses straight-line
distance — `_frame_to_dest_distance()` returns the *minimum haversine*
to any node in `multi_targets`, used purely to assign the (frame, dest)
pair into a band. Once the pair is selected, the route + GT verb come
from the network-shortest version above.

### GT verb — combining heading and route direction

The function `gt_verb_from_route` in `src/a2_route.py` takes the two
independent inputs and produces a deterministic verb. The full code:

```python
ACTION_DELTA = {"continue ahead": 0.0,
                "turn left":   -90.0,
                "turn right":  +90.0,
                "turn around": 180.0}

def gt_verb_from_route(G, path, heading, to_latlon):
    """Pick the verb whose new_heading is closest to first_edge_bearing."""
    if not path or len(path) < 2:
        return "continue ahead", 0.0, None, {}

    # ── PART A — extract the ROUTE direction ─────────────────────────
    # path[0] is the walker's current node; path[1] is the next
    # intersection. nx.shortest_path() returns nodes in TRAVERSAL ORDER
    # so bearing(path[0] → path[1]) is the forward-walking direction.
    # The graph is UTM-projected, so we convert each node's (x,y) back
    # to (lat,lon) before computing the compass bearing.
    n0, n1 = path[0], path[1]
    g0 = _node_latlon(G, n0, to_latlon)
    g1 = _node_latlon(G, n1, to_latlon)
    edge_bearing = _bearing(g0, g1)       # the OSM route's direction

    # ── PART B — combine with the camera HEADING ────────────────────
    # For every verb, simulate the heading AFTER that verb is applied,
    # then measure the gap from the route direction. Lowest gap wins.
    errors = {}
    for verb, delta in ACTION_DELTA.items():
        new_h = (heading + delta) % 360
        errors[verb] = round(abs(_angle_diff(new_h, edge_bearing)), 1)
    best_verb = min(errors, key=errors.get)

    return best_verb, edge_bearing, errors[best_verb], errors
```

Two independent inputs combined into one decision:

```
HEADING (camera direction, from heading_v2)
   "the walker's camera is pointed at 95° (east)"
              │
              ▼
        (heading + ACTION_DELTA[verb]) → new_h
                                           │
                                           │
                                           ▼
                                  error[verb] = |angle_diff(new_h, edge_bearing)|
                                           ▲
                                           │
                                  bearing(path[0] → path[1]) → edge_bearing
                                           ▲
                                           │
   path[0]→path[1] = first walkable segment of OSM shortest path

ROUTE DIRECTION (network first-edge, from nx.shortest_path)
              │
              ▼
            argmin → correct verb
```

### Worked example — line by line

`heading = 95°` (east) and the first edge of the OSM shortest path
goes from `(47.371, 8.543)` west to `(47.371, 8.541)`.

```python
# PART A — route direction
g0 = (47.371, 8.543)
g1 = (47.371, 8.541)
edge_bearing = _bearing(g0, g1)
   # dlon = -0.002, lat1 ≈ lat2
   # x ≈ sin(-0.002)·cos(lat2) ≈ -3.5e-5
   # y ≈ 0
   # atan2(x, y) ≈ -90° → modulo 360 = 270°
edge_bearing = 270.0       # the route goes west from here

# PART B — for each verb, compute the gap to the route direction
errors = {
    "continue ahead"  :  abs(angle_diff((95 +   0) % 360, 270)) = 175,
    "turn left"       :  abs(angle_diff((95 - 90) % 360, 270)) =  95,
    "turn right"      :  abs(angle_diff((95 + 90) % 360, 270)) =  85,
    "turn around"     :  abs(angle_diff((95 +180) % 360, 270)) =   5,   ★
}
best_verb = "turn around"
```

The walker is facing east but the route goes west → the only verb
that rotates the camera toward the route is `turn around` (gap = 5°).
The other verbs leave the walker pointed at 85-175° off-axis.

### Where the outputs go in `routes.jsonl`

```
gt_verb               = best_verb                # the chosen verb
route_bearing_network = edge_bearing             # the route direction (degrees)
verb_error_deg        = errors[best_verb]        # gap of the chosen verb
verb_errors           = errors                   # all 4 for debugging
```

### Per-row schema

```jsonl
{
  "video":           "bahnhofstrasse",
  "frame_id":        "frame_01380",
  // CURRENT
  "current_gps_raw":     [47.36988, 8.54213],   // DINOv2 raw
  "current_gps_snapped": [47.37001, 8.54218],   // HMM
  "current_node_id":     11111,
  "current_snap_offset_m": 4.2,
  "heading":             95.0,                  // from heading_v2

  // TARGET
  "destination":         "Grossmünster",
  "destination_zh":      "大教堂",
  "destination_kind":    "church",
  "destination_target_type": "point",
  "target_gps":          [47.37025, 8.54420],   // snapped, NOT canonical
  "target_node_id":      12345678,

  // OSM ROUTE
  "route_node_ids":      [11111, 22222, 33333, 12345678],
  "n_segments":          3,
  "route_bearing_network":     112.0,           // first-edge bearing
  "route_bearing_great_circle": 95.0,           // straight-line (for comparison)
  "bearing_diff_deg":           17.0,
  "route_distance_m":          187.4,
  "first_segment_length_m":    62.0,
  "frame_dest_distance_m_raw": 287.4,
  "sampling_band":             "near",

  // GROUND-TRUTH VERB
  "gt_verb":             "continue ahead",
  "verb_error_deg":      17.0,
  "verb_errors": {
    "continue ahead":  17.0,
    "turn left":      107.0,
    "turn right":      73.0,
    "turn around":   -163.0
  }
}
```

### Run
```bash
python -m src.a2_route
```

### Expected counts
```
matched frames:                   1,219
destinations per frame:               3
total (frame, destination) pairs: 3,657
   target distribution: ~80/10/10 across bands
```

---

## 12. Visualisations

### `viz/a2_vlmagreed.html` — STEP 4 visual QC
**Script**: `src/a2_viz_matched.py`
Random sample of 30 matched frames. Each row: QUERY frame + all 4
compass crops at top-1 pano (red on best) + heading v1 / heading v2 /
decision badge + GPS-side and VLM-side lists + match coincidences.
For verifying matched frames are not DINOv2 lookalike false positives.

### `viz/a2_mapped_GPS_spot.html` — 89 matched panos on a map
**Script**: `src/a2_viz_map.py` (uses folium)
Each pano = a circle (size ∝ matched-frame count, colour by # of
attractions visible at it: blue = 1, green = 2-3, orange = 4-7,
red = 8+). Overlay: the 21 canonical attraction GPS as red star markers.
Reveals the dense old-town cluster around Münsterhof / Münsterbrücke
and the singleton panos for thin attractions.

### `viz/a2_thin_attractions.html` — the 3+4 thin cohort
**Script**: `src/a2_viz_thin.py`
ALL matched frames grouped by attraction for the 7 thin cases:
- **≤1 frame**: Kunsthaus, Bürkliplatz, Paradeplatz
- **11-30**: Helmhaus, Sechseläutenplatz, Opernhaus, Landesmuseum

For visually assessing per-thin-attraction whether to drop, accept,
or augment with SV crops.

---

## 13. Evaluation semantics — how to judge a model's verb

### The boundary problem

The 4-verb scheme has 60° "dual-correct zones" at every boundary
angle (e.g., at exactly 60° to the right, both `continue ahead` and
`turn right` come within 30° of route_bearing).

### Rules considered

| Rule | What it asks | Trade-off |
|---|---|---|
| **A. Strict shortest** | model's verb == first verb of unique shortest path | Penalises slightly-longer-but-valid alternatives |
| **B. Top-K shortest paths** | model's verb ∈ first-verbs of K shortest paths | Multi-correct; K is arbitrary |
| **C. Within-ε of shortest** | first-verb of any path with length ≤ (1+ε) × shortest | Same as B but parameterised by length |
| **D. Makes progress** | after applying the verb, walker's network distance to destination decreases | Naturally accepts all valid first steps; expensive (Dijkstra per evaluated verb) |
| **E. Compass tolerance (Attempt 1)** | new_heading within 30° of great-circle bearing | Simple but ignores walkable paths (could say "through walls") |

### Recommended for Attempt 2 eval
- **Primary**: Rule D (`progress_correct`) — does the walker actually
  end up closer to the destination after the verb? Naturally accepts
  multiple correct routes.
- **Secondary** for diagnostic: report `strict_correct` (Rule A) and
  `soft_correct` (Rule E) alongside.

### For TRAINING annotation
- Use **Rule A (deterministic shortest path)** — clean labels, no
  teacher hallucination. `gt_verb` in `routes.jsonl` is computed via
  the algorithm in §11.

---

## 14. File manifest

All paths relative to project root.

### Scripts (`src/a2_*.py`)

| Script | Purpose |
|---|---|
| `a2_attraction_slots.py` | Defines `ATTRACTIONS_21` constants + alias table (single source of truth) |
| `a2_step1_gps_geo.py` | STEP 1 — GPS-side candidates per frame |
| `a2_step2_vlm_geo.py` | STEP 2 — VLM-side candidates per frame |
| `a2_step3_gps_vlm_geo.py` | STEP 3 — per-frame coincidence match (exact/substring/word_share) |
| `a2_viz_matched.py` | STEP 4 — random sample QC grid (a2_vlmagreed.html) |
| `a2_target_frames.py` | Per-attraction matched-frame list |
| `a2_heading_v2.py` | Gap-tiered heading decision (top1 vs top1+top2) |
| `a2_destination_targets.py` | Hybrid routing targets (16 point + 5 multi) |
| `a2_route.py` | Network-routed bearing + GT verb per (frame, dest) pair |
| `a2_viz_map.py` | Folium map of 89 matched panos |
| `a2_viz_thin.py` | All frames for the 7 thin attractions |
| `a2_match_strict.py` | (legacy) strict exact-match filter — superseded by STEP 3 |
| `a2_proximity_tag.py` | (legacy) proximity tag — superseded by STEP 1 |
| `a2_join_3way.py` | (legacy) 3-way join — superseded by STEP 3 |
| `a2_vlm_coverage.py` | Diagnostic — per-attraction VLM mention count |
| `a2_raw_vlm_strings.py` | Diagnostic — raw VLM `guess`/`visible[]` dump |
| `a2_sanity_check.py` | DINO↔VLM mapping sanity check (not yet run) |
| `a2_sv_pano_attractions.py` | Per-SV-crop attraction tag (4,431 crops) |

### Data outputs (`data/cities/zurich/a2/*.jsonl`)

| File | Rows | Produced by | Purpose |
|---|---:|---|---|
| `GPS_GEO.jsonl` | 15,053 | a2_step1_gps_geo | GPS+OSM candidates per frame |
| `VLM_GEO.jsonl` | 4,891 | a2_step2_vlm_geo | VLM candidates per frame |
| `GPS_VLM_GEO.jsonl` | 4,158 | a2_step3_gps_vlm_geo | Match results (filtered cos≥0.75) |
| `target_attraction_frames.jsonl` | 21 | a2_target_frames | Per-attraction frame lists |
| `attraction_slots.jsonl` | 21 | (older path) | Per-attraction slot map |
| `heading_v2.jsonl` | 15,053 | a2_heading_v2 | Gap-tiered headings |
| `road_snapped_a2.jsonl` | 2,470 | road_snap.py | HMM-snapped GPS for tier-1 |
| `destination_targets.jsonl` | 21 | a2_destination_targets | Hybrid routing targets |
| `routes.jsonl` | ~3,657 | a2_route | (frame, dest) + route + GT verb |
| `sv_attractions.jsonl` | 4,431 | a2_sv_pano_attractions | Per-SV-crop tag |
| `match_strict.jsonl` | 752 | a2_match_strict | (legacy) strict filter output |
| `proximity_tag.jsonl` | 2,470 | a2_proximity_tag | (legacy) per-frame proximity |
| `join_3way.{jsonl,tsv}` | 1,858 | a2_join_3way | (legacy) 3-way audit |
| `extra_pois.json` | 2 | hand-curated | Paradeplatz + Rathaus (OSM-extraction-missed) |

### Visualisations (`viz/a2_*.html`)

| File | Produced by | Purpose |
|---|---|---|
| `a2_vlmagreed.html` | a2_viz_matched | STEP 4 — 30 random matched frames QC |
| `a2_mapped_GPS_spot.html` | a2_viz_map | 89 matched panos on a map |
| `a2_thin_attractions.html` | a2_viz_thin | Thin-attraction cohort QC |

### Upstream files (read-only — do not modify)

| File | Origin |
|---|---|
| `gps_recovery_full.jsonl` | `src/gps_recovery.py` (Attempt 1) |
| `poi_scan.jsonl` | `src/poi_scan.py` (Attempt 1) |
| `poi_scan_cos0.75.jsonl` | `_vlm_test.py` (Attempt 1 expansion) |
| `pois.json` | `src/pois.py` (Attempt 1 OSM extraction) |
| `osm_walking.pkl` | `src/build_walking_graph.py` (Attempt 1; UTM-projected) |
| `dinov2/sv_v1.npz`, `dinov2/frames_n1_l0.npz` | `src/dinov2_match.py` (Attempt 1) |
| `streetview/zurich/meta.jsonl` | `src/streetview.py` (Attempt 1) — 4,431 SV crops |

---

## 15. Reproduction sequence

To regenerate everything from upstream Attempt 1 data:

```bash
# 1. Curated constants — edit if attraction list changes
$EDITOR src/a2_attraction_slots.py

# 2. Heading rule (no inputs, all from gps_recovery_full + DINOv2 cache)
python -m src.a2_heading_v2 --hi 0.20 --lo 0.0

# 3. GPS side (depends on heading_v2 for the merged fields)
python -m src.a2_step1_gps_geo --radius 100

# 4. VLM side (independent of GPS side)
python -m src.a2_step2_vlm_geo

# 5. Coincidence match at cos≥0.75
python -m src.a2_step3_gps_vlm_geo --cos-min 0.75

# 6. Per-attraction frame list (for destination cohort)
python -m src.a2_target_frames

# 7. HMM road-snap on the wider tier-1 cohort (no POI filter)
python -m src.road_snap --tier 1 --top-pois 0 \
       --output data/cities/zurich/a2/road_snapped_a2.jsonl
mv data/cities/zurich/road_snapped_a2.jsonl data/cities/zurich/a2/

# 8. Destination targets (hybrid: 16 point + 5 multi)
python -m src.a2_destination_targets

# 9. Routes + GT verbs (band-sampled 80/10/10)
python -m src.a2_route

# 10. Visualisations
python -m src.a2_viz_matched --n 30
python -m src.a2_viz_map
python -m src.a2_viz_thin
```

---

## 16. Next steps (pending)

| Step | What | Status |
|---|---|---|
| Decide on 3 weak attractions | Kunsthaus / Bürkliplatz / Paradeplatz each have 1 matched frame — drop, augment with SV crops, or accept | Pending |
| Per-attraction radii for long features | Bahnhofstrasse / Limmatquai / Lake Zurich / Limmat / Niederdorfstrasse need larger R or multi-anchor to recover the 59 unmatched-but-passing panos | Deferred to attempt 3 |
| Train/test split | **Per-variant independent random 80/10/10** (each variant shuffled with `seed=42` over its own `format_pass` rows). Cross-variant comparison in §20 is rate-vs-rate; within-variant `zs-X` vs `trained-X` is paired automatically. See §21. | **Resolved** |
| Re-annotation prompt v2 | New `src/a2_annotate.py` consuming `routes.jsonl` as input; teacher VLM reasons about route + emits verb; drop checkpoint step (unverifiable in Attempt 1) | Pending |
| Retrain L-given | Same training code (`train_modal.py`), new annotation file | Pending |
| Eval metrics implementation | `progress_correct` + `strict_correct` + `soft_correct` reporting alongside | Pending |

Cost estimate (unchanged from §6.5 of the original `DEV_MANUAL.md`):
- Re-annotate ~3,657 (frame, dest) pairs with Gemini Pro 2.5: ~$30
- Retrain L-given LoRA on Modal A100: ~$10 GPU
- Total: ~$40, ~4 h wall-time

---

## 18. The 6-condition experiment matrix

Six conditions, all evaluated against the SAME student model
**Qwen 2.5 VL 7B**. The difference between zero-shot and trained is
whether a LoRA adapter is applied:

- **Zero-shot**: base Qwen 2.5 VL 7B (no fine-tuning) prompted cold
- **Trained**: base Qwen + a LoRA adapter SFT-trained on a specific
  variant of the annotation data

The teacher model **Gemini Pro 2.5** is used ONLY to generate the
training-data annotations (one call per (frame, destination) pair).
We never evaluate Gemini Pro 2.5 itself — it is the labeler, not the
system under test.

```
                          ZERO-SHOT                       TRAINED
                          (base Qwen 2.5 VL 7B,          (Qwen 2.5 VL 7B + LoRA,
                           no fine-tuning,                 SFT on the corresponding
                           cold-prompted)                  variant of annotations)
                          ──────────────────────────      ──────────────────────────
heading-given             zs-heading-given                trained-heading-given
  (heading # in input)
heading-derived           zs-heading-derived              trained-heading-derived
  (no heading in input;
   CoT derives heading)
heading-implicit          zs-heading-implicit             trained-heading-implicit
  (no heading in input;
   visual-only CoT)
```

All 6 conditions are evaluated on **Modal A100** via `eval_modal.py`
(the inference harness from Attempt 1). Zero-shot uses the base model
weights as-is; trained loads the corresponding LoRA adapter on top.

### Per-condition input + CoT style

| # | Condition | Heading in user prompt? | `<thinking>` style | Training data |
|---:|---|:-:|---|---|
| 1 | `zs-heading-given` | YES | freely mentions heading | (zero-shot) |
| 2 | `zs-heading-derived` | NO | prompt instructs "first state estimated heading" | (zero-shot) |
| 3 | `zs-heading-implicit` | NO | prompt instructs "reason visually, no numeric heading" | (zero-shot) |
| 4 | `trained-heading-given` | YES | mentions heading numerically | base annotations as-is |
| 5 | `trained-heading-derived` | NO | begins with "I estimate I'm facing X°…" | base CoT prepended with heading-derivation sentence |
| 6 | `trained-heading-implicit` | NO | visual reasoning, no numeric heading | base CoT with numeric heading replaced by visual descriptors |

### Hypotheses

| Comparison | Tests |
|---|---|
| zs-* vs trained-* (same mode) | Does fine-tuning help? |
| trained-heading-given vs trained-heading-derived | Does deriving heading recover most accuracy? |
| trained-heading-given vs trained-heading-implicit | Cost of dropping heading entirely |
| trained-heading-derived vs trained-heading-implicit | Is explicit derivation better than visual-only? |

**Main project result**: if `trained-heading-derived` or
`trained-heading-implicit` PASS is within ~5-10 % of
`trained-heading-given`, the compass-free thesis holds.

---

## 19. Annotation prompt v2

### System prompt (same for all 6 conditions)

```
You are a Zurich walking-tour guide speaking directly to a tourist
who is looking at the photo right now. Help them take the next step.

Your reply has two parts (2-3 sentences total):

<thinking>
1-2 short sentences for your reasoning — where the route's first
segment is relative to the walker's heading, and which verb rotates
the camera onto that direction.
</thinking>

<answer>
1 sentence speaking DIRECTLY to the walker, then the action verb.
- Use "you" and point to specific things they can see:
    "Can you see X?"  "Look at the X."  "Notice the X ahead."
- Reference only landmarks from the "Visible landmarks" list. The
  walker has no map; only what they can SEE in the photo helps them.
- End with the action verb in its own short sentence.
</answer>

The action verb must be EXACTLY one of:
  continue ahead    turn left    turn right    turn around

GOOD <answer> examples:
  "Can you see Münsterbrücke directly ahead? Turn around."
  "Look at the cathedral towers on your left. Turn left."
  "Notice Bahnhofstrasse with shop signs stretching ahead. Continue ahead."
  "There is no clear landmark in front of you. Turn right."

AVOID:
  - Naming places NOT in the Visible landmarks list.
  - Mentioning the destination by name unless it's also visible now.
  - Compass directions ("head north") — say "ahead", "to your right".
```

### User-prompt template A — `heading-given` (conditions 1, 4)

```
[IMAGE]
You are at this location, facing 95° (east-by-north-east).

Destination: Grossmünster (大教堂), about 287 m walking distance.

OSM walking route:
  First segment heads 270° (west) along Limmatquai for 62 m,
  then 2 more turns over a total of 187 m.

Visible landmarks at this spot:
  Limmatquai, Münsterbrücke

Decide the next action verb.
```

### User-prompt template B — `heading-derived` (conditions 2, 5)

```
[IMAGE]
Destination: Grossmünster (大教堂), about 287 m walking distance.

OSM walking route:
  First segment heads 270° (west) along Limmatquai for 62 m,
  then 2 more turns over a total of 187 m.

Visible landmarks at this spot:
  Limmatquai, Münsterbrücke

The walker's heading is NOT provided. In <thinking>, FIRST infer the
heading from the photo by stating "I estimate I'm facing X° (direction)",
THEN reason about the route and verb.
```

### User-prompt template C — `heading-implicit` (conditions 3, 6)

```
[IMAGE]
Destination: Grossmünster (大教堂), about 287 m walking distance.

OSM walking route:
  First segment heads 270° (west) along Limmatquai for 62 m,
  then 2 more turns over a total of 187 m.

Visible landmarks at this spot:
  Limmatquai, Münsterbrücke

The walker's heading is NOT provided. Reason from visual cues in the
photo about where the destination is and which verb is needed. Do NOT
state a numeric heading.
```

### Base annotation generation

Only ONE Gemini Pro 2.5 call per (frame, destination) pair, using
template A (heading-given). The 3 trained variants are derived locally
by text transforms; the 3 zero-shot conditions are evaluated by calling
Gemini cold with the appropriate template at eval time.

### Expected response format

```
<thinking>
1-2 short sentences (style varies by condition).
</thinking>
<answer>
Can you see [Visible landmark] [position]? [VERB].
</answer>
```

Example (template A, `trained-heading-given`):
```
<thinking>
I am facing 95° (east); the route's first segment heads 270° (west),
so the destination lies 180° behind me.
</thinking>
<answer>
Notice Limmatquai with the tram tracks stretching ahead — you came
from that direction. Turn around.
</answer>
```

---

## 20. Evaluation metrics (v2)

Only **4 metrics** are scored — all computed by `src/a2_score.py` from
the per-condition `per_sample.jsonl` produced by `src/a2_eval_modal.py`.
The earlier `anchor_vocab_pass` and `interactive_style_pass` diagnostics
have been removed.

### The 4 metrics (per response)

```
format_pass           : has <thinking>…</thinking><answer>…</answer>
                        structure AND first verb-token in <answer> is one
                        of the 4 verbs
direction_pass        : the first verb in <answer> == GT verb (from
                        routes.jsonl)
PASS                  : format_pass AND direction_pass         ← headline
heading_inference_acc : only for `*-heading-derived` conditions —
                        if <thinking> contains "facing X°" (regex), check
                        |X − true_heading| < 22.5° (circular). Reported
                        as n/a for `given` and `implicit` conditions.
```

### Per-condition reporting

```
                              n     fmt    dir    PASS   h_inf  h_n
zs-heading-given              ?      ?      ?      ?      n/a    0
zs-heading-derived            ?      ?      ?      ?       ?     ?
zs-heading-implicit           ?      ?      ?      ?      n/a    0
trained-heading-given         ?      ?      ?      ?      n/a    0
trained-heading-derived       ?      ?      ?      ?       ?     ?
trained-heading-implicit      ?      ?      ?      ?      n/a    0
```

`h_n` = number of derived-condition rows where a "facing X°" string was
parseable from the model's `<thinking>` (denominator of
`heading_inference_acc`). A low `h_n` on a derived condition means the
model is not emitting the expected CoT pattern, so the accuracy number
should be interpreted with caution.

### Parsing rule (only the first verb counts)

```python
VERBS = ("continue ahead", "turn left", "turn right", "turn around")

def parse_response(text):
    # 1. find <thinking>…</thinking> and <answer>…</answer> blocks
    #    (closing </answer> may be missing — Pro 2.5 often omits it; the
    #     parser treats end-of-text as the close in that case)
    # 2. first verb = earliest verb match in answer_text (case-insensitive,
    #    longest-match-first to prefer "continue ahead" over "ahead")
    # 3. format_pass = both tags opened AND first_verb is not None
    # 4. derived_heading = float in regex r"facing\s+(\d{1,3}(?:\.\d+)?)\s*°"
    #    applied to the <thinking> text (None if no match)
    ...

def score_row(row):
    parsed = parse_response(row["model_response"])
    direction_pass = parsed["first_verb"] == row["gt_verb"]
    PASS = parsed["format_pass"] and direction_pass
    is_derived = row["condition"].endswith("-heading-derived")
    heading_inference_pass = None
    if is_derived and parsed["derived_heading"] is not None:
        heading_inference_pass = (
            circular_diff(parsed["derived_heading"], row["heading"]) < 22.5)
    ...
```

### Cost & wall-time estimate

```
Teacher annotation (Gemini Pro 2.5,            ~$30      ~3 h
  one call per (frame, dest) pair via
  src/a2_annotate.py — generates the base
  dataset used by all 3 trained conditions):

Derive 3 training files (text transforms       $0        instant
  via src/a2_derive_variants.py):

Train 3 LoRAs (Qwen 2.5 VL 7B + LoRA on        ~$15      ~1 h
  Modal A100, one adapter per training
  variant):

Eval 6 conditions on Modal A100                ~$10      ~1.5 h
  (Qwen base for the 3 zero-shot; Qwen +
   the corresponding LoRA for the 3 trained):
─────────────────────────────────────────────────────────
Total                                          ~$55      ~5.5 h
```

Gemini Pro 2.5 is only invoked during the teacher-annotation step.
All evaluation (including zero-shot) runs on Qwen 2.5 VL 7B via
Modal — same harness as Attempt 1.

---

## 21. Per-variant SFT conversion (`src/a2_to_sft.py`)

**Script**: `src/a2_to_sft.py`
**Inputs**: `data/cities/zurich/a2/annotations_a2_{variant}.jsonl`
            (one file per variant — given / derived / implicit)
**Outputs** (3 files per run, one variant at a time):

```
data/sft/a2_{variant}_train.jsonl     a2_{variant}_val.jsonl     a2_{variant}_test.jsonl
```

### Per-variant independent random 80/10/10

Each variant is processed in its own invocation and gets its own random
shuffle (`seed=42`). The 3 variants' train/val/test splits are NOT
aligned — they contain different `(video, frame_id, destination)`
instances. This is by design.

### Why we are NOT aligning splits across variants

Earlier we considered intersecting keys across the 3 variants to build
a single shared cohort, on the theory that "same test instances" would
let us compute per-row deltas across conditions. That framing is wrong,
for two reasons:

1. **Cross-variant comparisons are NOT per-row valid even with shared
   keys.** Each variant gives the same image a different input
   (heading shown for given, hidden + 4-step derive for derived,
   hidden + 3-step visual for implicit). So a delta like
   `trained-derived[i].PASS − trained-given[i].PASS` on the same
   instance #i is *not* isolating any single effect — it conflates
   input-prompt difference with output-behavior difference. The only
   thing meaningfully comparable across variants is **rates**
   (PASS rate, format rate, heading_inference rate).

2. **The intersection cost is severe at the teacher's true pass rate.**
   The teacher's per-variant PASS rate is roughly ~70 %. Intersecting
   `--only-pass` across all 3 variants would drop retention to
   ~0.7³ ≈ 0.34 — chopping the training pool by ~2× for an alignment
   property that, per point 1, isn't useful.

### Where per-row comparisons DO survive (for free)

Within a single variant, the trained-vs-zero-shot comparison is paired
by construction: `zs-given` and `trained-given` both load
`a2_given_test.jsonl`, so they see identical inputs on identical
instances. The 3 within-variant comparisons:

```
zs-given      ↔ trained-given        → paired (same test file)
zs-derived    ↔ trained-derived      → paired
zs-implicit   ↔ trained-implicit     → paired
```

These are the comparisons that answer the headline scientific question
"does the LoRA adapter help for this prompt style?"

The cross-variant comparison (e.g. `trained-derived` vs
`trained-given`) is rate-vs-rate only.

### Filters

| Flag | Effect | Trade-off |
|---|---|---|
| (default) | Keep only rows with `format_pass==True` | Largest cohort. Includes rows where the teacher's verb was wrong (direction_pass==False) — the student still learns the format and CoT style from those. |
| `--only-pass` | Additionally require `direction_pass==True` | Cleaner SFT labels; cohort shrinks to ~0.7× at the current teacher pass rate. Use if direction-accuracy is more important than data volume. |

### Run

```bash
python -m src.a2_to_sft --variant given
python -m src.a2_to_sft --variant derived
python -m src.a2_to_sft --variant implicit

# optional — cleaner SFT labels at ~0.7× cohort size:
python -m src.a2_to_sft --variant given --only-pass

# optional — video-holdout split instead of random:
python -m src.a2_to_sft --variant given --holdout-video <video_id>
```

Then upload to Modal once all 3 are produced:

```bash
modal volume put navlm-data data/sft/a2_*.jsonl /sft/
```

---

## 17. Glossary

| Term | Definition |
|---|---|
| **21-list** / `ATTRACTIONS_21` | The 21 hand-curated famous attractions from §2 |
| **Matched cohort** | The 1,219 frames where GPS-side and VLM-side lists share at least one name |
| **best_level** | Strongest match level for a frame: attraction > landmark > poi |
| **HMM-snapped GPS** | `gps_snapped` from `road_snap.py` — raw DINO GPS map-matched to the OSM walking graph |
| **Point target / Multi target** | Hybrid routing target type per §10 |
| **GT verb** | Ground-truth verb computed deterministically via `gt_verb_from_route()` |
| **Coincidence types** | `exact` / `substring` / `word_share` — three rules for matching names in STEP 3 |
| **heading v1** | `gps_recovery`'s all-4-cosine-weighted heading |
| **heading v2** | Gap-tiered heading (§7) |
| **EPSG:32632** | UTM Zone 32N — the projected CRS of `osm_walking.pkl` (positions in metres) |

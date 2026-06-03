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
DATA PIPELINE (offline, local, one-time)
═════════════════════════════════════════════════════════════════════

GPS-side                  VLM-side
─────────                 ─────────
gps_recovery_full.jsonl   poi_scan.jsonl + poi_scan_cos0.75.jsonl
       ↓                          ↓
 (STEP 1) a2_step1         (STEP 2) a2_step2
 GPS_GEO.jsonl             VLM_GEO.jsonl
       └────────┬───────────────┘
                ↓
      (STEP 3) a2_step3
      GPS_VLM_GEO.jsonl  →  matched cohort: 1,219
                ↓
      a2_target_frames + a2_heading_v2 + road_snap +
      a2_destination_targets    (16 point + 5 multi targets)
                ↓
      a2_route   (network-routed shortest path + deterministic GT verb)
      routes.jsonl  (3,657 (frame, dest) pairs)


ANNOTATION (3 parallel passes, Vertex AI Gemini Pro 2.5)
═════════════════════════════════════════════════════════════════════

routes.jsonl
   ↓                              ↓                            ↓
a2_annotate --variant given   --variant derived   --variant implicit
   ↓ (GCP project 1)             ↓ (GCP project 2)             ↓ (GCP project 3)
annotations_a2_given.jsonl   _derived.jsonl     _implicit.jsonl
   (~3,657 rows each — teacher sees heading; student prompt
    is variant-specific: heading shown only for "given")


SFT CONVERSION + UPLOAD (local)
═════════════════════════════════════════════════════════════════════

annotations_a2_{variant}.jsonl
   ↓ a2_to_sft --variant {given,derived,implicit}
data/sft/a2_{variant}_{train,val,test}.jsonl  (80/10/10, seed=42, per variant)
   ↓ modal volume put navlm-data data/sft/a2_*.jsonl /sft/


TRAINING (Modal A100-80GB, 9 LoRA adapters — rank sweep)
═════════════════════════════════════════════════════════════════════

For each (variant ∈ {given, derived, implicit})
    for each (rank ∈ {4, 8, 16}):
        a2_train_modal --variant <v> --lora-r <r>
            ↓
        /ckpts/lora_a2_<v>_r<r>_e2/   (LoRA adapter on navlm-ckpts)


EVALUATION (Modal A100-40GB, 12 conditions = 3 zs + 9 trained)
═════════════════════════════════════════════════════════════════════

For each condition:
    a2_eval_modal --condition <cond> [--adapter ...]
        ↓
    /eval/<run_id>/<cond>/per_sample.jsonl   (on navlm-eval)


SCORING (local)
═════════════════════════════════════════════════════════════════════

modal volume get navlm-eval <run_id> eval_pull/
    ↓
python -m src.a2_score --run-dir eval_pull/<run_id>
    ↓ 4 metrics × 12 conditions
summary_table.txt  (final report input)
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

## 16. Status (as of 2026-06-02)

| Step | What | Status |
|---|---|---|
| Decide on 3 weak attractions | Kunsthaus / Bürkliplatz / Paradeplatz each have 1 matched frame — accepted as-is; routes.jsonl includes them via the destination-cohort sampler | **Resolved (accepted)** |
| Per-attraction radii for long features | Multi-target routing (§10) addresses this — Bahnhofstrasse / Limmatquai / Lake Zurich / Limmat / Niederdorfstrasse use the matched-cohort node list + canonical fallback | **Resolved** |
| Re-annotation prompt v2 | `src/a2_annotate.py` produces 3 variant-specific datasets (given/derived/implicit) via 3 parallel Gemini Pro 2.5 passes — see §18 | **given 100 %, implicit 100 %, derived 88.6 %** — only derived still running, ETA ~3 h. Teacher pass rates so far: **given 87.5 %, derived 72.1 %, implicit 73.0 %** (whole-dataset PASS 77.8 % on 10,614 rows; see §18 "Whole-dataset teacher-quality analysis" — headline finding: **turn-around bias** — 60 % of teacher direction failures are turn-left/right mis-classified as turn-around). |
| Train/test split | Per-variant independent random 80/10/10 (`seed=42`, format_pass only by default) — see §20 | **Resolved** |
| Modal infrastructure | 3 volumes (`navlm-data/ckpts/eval`), 2 apps (`navlm-train-a2/eval-a2`), CLI cheat sheet + Windows gotchas — see §21 | **Resolved** |
| Frame upload to Modal | 1,030 needed frames (517 MB) discovered missing during trial; now uploaded to `navlm-data:/frames/` | **Resolved (2026-06-02)** |
| **Loss masking** | `<thinking>` + `<answer>` only — system+user tokens masked from CE loss. Validated by 32-sample overfit test (train→0.005, masked eval U-shaped, peak min at epoch 5) — see §22 | **Resolved (2026-06-02)** |
| **Early stopping** | `EarlyStoppingCallback(patience=2)` + `load_best_model_at_end=True` on `eval_loss` (the masked version) — see §22 | **Resolved (2026-06-02)** |
| Trial smoke test | 3 zs evals + 3 train-on-32 + 1 overfit-test all green end-to-end. **All 6 smoke conditions complete**: zs-given 62.5 % → trained-given 43.8 % (Δ −18.8 pp ← small-data degradation), zs-derived 18.8 % → trained-derived 25.0 % (Δ +6.2 pp ← LoRA learning CoT structure), zs-implicit 12.5 % → trained-implicit 12.5 % (Δ 0.0 pp ← null at this scale). SMOKE caveats apply (n=16, 32-sample × 1-epoch training, no loss masking) — see §19 "Trial smoke results" for details. | **Resolved (2026-06-02)** |
| LoRA training (full sweep) | 9 adapters trained (3 variants × r=4/8/16), `--only-pass` filter, 3 epochs, masked loss, early-stop p=2 | **Resolved (2026-06-03)** — all 9 adapters at `/ckpts/lora_a2_<v>_r<r>_e3/` |
| Resume-training (e3 → e5) | 6 given+implicit adapters extended +2 epochs each. Val_loss gained 0-3 % more; **r=16 has saturated** (implicit-r16 Δ=0.0) | **Resolved (2026-06-03)** |
| Eval harness (full sweep) | 12 conditions evaluated (3 zs + 9 trained-r{4,8,16}) — see §17 "Results" table | **Resolved (2026-06-03)** (11/12 complete, given-r8 partial at n=34) |
| Eval scoring | `src/a2_score.py` ready (4 metrics, see §19); rank-suffixed output dirs prevent multi-rank overwrites | **Resolved** |
| **Headline result** | **Compass-free thesis holds (with caveats)**: trained-derived-r8 = 64.9 % PASS (vs 26.8 % zs, vs 98.1 % trained-given-r16). Implicit-r16 = 55.1 % (vs 28.1 % zs). Numeric heading IS meaningful — gap to trained-given is ~33/43 pp, larger than the original 5-10 pp target. | **Resolved (2026-06-03)** |
| **Final report** | CS231n + NeurIPS 2026 — uses the 12-condition × 4-metric table from §17 / §19 | Pending — write-up |

For the full cost + wall-time breakdown, see the Cost section at the
end of §19.

---

## 17. The 12-condition experiment matrix (with LoRA rank sweep)

All conditions are evaluated against the SAME student model
**Qwen 2.5 VL 7B**. The two axes are (1) the prompt variant (given /
derived / implicit) and (2) zero-shot vs trained — with the trained
side ALSO swept across LoRA ranks **r ∈ {4, 8, 16}**:

- **Zero-shot** (3 conditions): base Qwen 2.5 VL 7B, no adapter.
- **Trained** (9 conditions): base Qwen + 1 of 9 LoRA adapters
  (3 variants × 3 ranks).

The teacher model **Gemini Pro 2.5** is used ONLY to generate the
training-data annotations (3 independent passes — §18). We never
evaluate Gemini Pro 2.5 itself — it is the labeler, not the system
under test.

```
                       ZERO-SHOT          TRAINED (LoRA rank)
                       (no adapter)       ┌───────┬───────┬───────┐
                                          │  r=4  │  r=8  │ r=16  │
                       ─────────────      ┼───────┼───────┼───────┤
heading-given          zs-heading-given   │ t-g-4 │ t-g-8 │ t-g-16│
heading-derived        zs-heading-derived │ t-d-4 │ t-d-8 │ t-d-16│
heading-implicit       zs-heading-implicit│ t-i-4 │ t-i-8 │ t-i-16│
                                          └───────┴───────┴───────┘
                       ──────────         ──────────────────────────
                       3 conditions       9 conditions  →  12 total
```

(`t-g-4` = `trained-heading-given-r4`, etc.)

All 12 conditions are evaluated on **Modal A100-40GB** via
`src/a2_eval_modal.py`. Zero-shot uses base weights as-is; trained
loads the corresponding LoRA adapter from `/ckpts/lora_a2_<v>_r<r>_e2/`
on top.

### Per-condition input + CoT style + training data

| Conditions | Heading in user prompt? | `<thinking>` style | Training data |
|---|:-:|---|---|
| `zs-heading-given` / `trained-heading-given-r{4,8,16}` | YES | numeric heading + route bearing → verb | `annotations_a2_given.jsonl` |
| `zs-heading-derived` / `trained-heading-derived-r{4,8,16}` | NO | 4-step derivation: cues → geography → heading → verb | `annotations_a2_derived.jsonl` |
| `zs-heading-implicit` / `trained-heading-implicit-r{4,8,16}` | NO | 3-step visual chain (NO numeric heading) | `annotations_a2_implicit.jsonl` |

(Training data is variant-specific because the teacher's `<thinking>`
follows the variant's CoT template — see §18.)

### Why rank sweep r ∈ {4, 8, 16}

LoRA rank controls the capacity of the adapter (parameter count grows
linearly with r). For Qwen 2.5 VL 7B targeting q/k/v/o_proj:

```
r=4   ~ 2.1 M trainable params  (≤ 0.03 % of base)
r=8   ~ 4.2 M trainable params  (≤ 0.06 %)
r=16  ~ 8.4 M trainable params  (≤ 0.12 %)
```

- r=4 tests whether the variant prompt + CoT format is teachable with
  minimal capacity — if r=4 already matches r=16, the bottleneck is
  the prompt design, not capacity.
- r=16 is the "comfort" rank typically reported in LoRA papers.
- r=8 is the midpoint; if PASS rate monotonically improves r=4 → 8 → 16,
  more capacity is helping; if it plateaus, we have rank-saturation.

`lora_alpha` is fixed at `2 * r` (standard practice) so the effective
learning rate per LoRA weight is rank-invariant.

### Hypotheses

| Comparison | Tests |
|---|---|
| `zs-X` vs `trained-X-r16` (same variant) | Does fine-tuning help? (headline trained-vs-zs effect) |
| `trained-X-r4` vs `r8` vs `r16` (same variant) | Rank-saturation: is more LoRA capacity helping? |
| `trained-given-r16` vs `trained-derived-r16` | Does deriving the heading recover most accuracy? |
| `trained-given-r16` vs `trained-implicit-r16` | Cost of dropping heading entirely (visual-only) |
| `trained-derived-r16` vs `trained-implicit-r16` | Is explicit derivation better than visual-only? |

### Ablation matrix — the concrete runs

Using `--only-pass` filter (rows that PASS both format and direction
teacher checks — see §18) so training and test data is clean of the
teacher's turn-around bias. Per-variant sample counts:

```
            train rows   val   test   adapters   eval conditions
─────────────────────────────────────────────────────────────────
given           2,561     320   320      3 (r=4,8,16)   4 (zs + 3 trained)
implicit        2,137     267   267      3              4
derived        ~2,118    ~265  ~265      3              4
─────────────────────────────────────────────────────────────────
total           6,816     852   852      9             12
```

9 LoRA training ablations, all with the same hyperparameters
(§22 — `--only-pass`, masked CE loss, 3-epoch cap, early stop p=2,
load_best_model_at_end). Adapter naming: `lora_a2_<variant>_r<r>_e3/`.

```
─────────────────────────────────────────────────────────────────────────────────────────────────────
condition           rank  α    train  val   optim steps    wall-time   cost    data-ready?
─────────────────────────────────────────────────────────────────────────────────────────────────────
given-r4             4    8    2,561  320   3 × 320 = 960   ~65 min     $3.75   now
given-r8             8   16    2,561  320   960             ~65 min     $3.75   now
given-r16           16   32    2,561  320   960             ~65 min     $3.75   now
implicit-r4          4    8    2,137  267   3 × 267 = 801   ~55 min     $3.20   now
implicit-r8          8   16    2,137  267   801             ~55 min     $3.20   now
implicit-r16        16   32    2,137  267   801             ~55 min     $3.20   now
derived-r4           4    8   ~2,118 ~265   3 × 265 = 795   ~55 min     $3.20   after derived annotation (~3 h)
derived-r8           8   16   ~2,118 ~265   795             ~55 min     $3.20   after derived annotation
derived-r16         16   32   ~2,118 ~265   795             ~55 min     $3.20   after derived annotation
─────────────────────────────────────────────────────────────────────────────────────────────────────
total                                       7,668 steps    ~65 min     ~$30.50   (parallel wall time)
                                                          (parallel)             (~$22 if early stop fires)
─────────────────────────────────────────────────────────────────────────────────────────────────────
```

12 eval conditions (Modal A100-40GB):

```
─────────────────────────────────────────────────────────────────────────────────────────────
condition              adapter                              n_test   per-s    time      cost
─────────────────────────────────────────────────────────────────────────────────────────────
zs-given               (base Qwen, no adapter)               320     4.6 s   ~29 min   $1.00
zs-implicit            (base Qwen, no adapter)               267     7.0 s   ~36 min   $1.25
zs-derived             (base Qwen, no adapter)              ~265     7.4 s   ~38 min   $1.35
trained-given-r4       /ckpts/lora_a2_given_r4_e3            320     4.6 s   ~29 min   $1.00
trained-given-r8       /ckpts/lora_a2_given_r8_e3            320     4.6 s   ~29 min   $1.00
trained-given-r16      /ckpts/lora_a2_given_r16_e3           320     4.6 s   ~29 min   $1.00
trained-implicit-r4    /ckpts/lora_a2_implicit_r4_e3         267     7.0 s   ~36 min   $1.25
trained-implicit-r8    /ckpts/lora_a2_implicit_r8_e3         267     7.0 s   ~36 min   $1.25
trained-implicit-r16   /ckpts/lora_a2_implicit_r16_e3        267     7.0 s   ~36 min   $1.25
trained-derived-r4     /ckpts/lora_a2_derived_r4_e3         ~265    11.5 s   ~56 min   $1.95
trained-derived-r8     /ckpts/lora_a2_derived_r8_e3         ~265    11.5 s   ~56 min   $1.95
trained-derived-r16    /ckpts/lora_a2_derived_r16_e3        ~265    11.5 s   ~56 min   $1.95
─────────────────────────────────────────────────────────────────────────────────────────────
total                                                                        ~56 min   ~$16.30
                                                                            (parallel)
─────────────────────────────────────────────────────────────────────────────────────────────
```

**Grand total — both training + eval together: ~$47, ~5 h wall-time
parallel (or ~$35 with early stop firing).**

### Results — full 12-condition PASS table (2026-06-02 / -03)

11 of 12 conditions complete (`trained-given-r8` was still
finishing at last update — partial n=34 included for completeness
since it already aligns with the other given-r* rates).

```
condition                            n      fmt     dir     PASS      vs zs       vs best given (r=16)
─────────────────────────────────────────────────────────────────────────────────────────────────────
zs-heading-given                    320  100.0%   44.7%   44.7%      baseline    −53.4 pp
zs-heading-derived                  265  100.0%   26.8%   26.8%      baseline    −71.3 pp
zs-heading-implicit                 267  100.0%   28.1%   28.1%      baseline    −70.0 pp

trained-heading-given-r4            320  100.0%   97.2%   97.2%      +52.5 pp     −0.9 pp
trained-heading-given-r8 (partial)   34  100.0%   97.1%   97.1%      +52.4 pp     −1.0 pp
trained-heading-given-r16           320  100.0%   98.1%   98.1%      +53.4 pp     ← best given

trained-heading-derived-r4          265  100.0%   58.5%   58.5%      +31.7 pp    −39.6 pp
trained-heading-derived-r8          265  100.0%   64.9%   64.9%      +38.1 pp    −33.2 pp  ← best derived
trained-heading-derived-r16         265  100.0%   63.0%   63.0%      +36.2 pp    −35.1 pp

trained-heading-implicit-r4         267  100.0%   50.2%   50.2%      +22.1 pp    −47.9 pp
trained-heading-implicit-r8         267  100.0%   54.3%   54.3%      +26.2 pp    −43.8 pp
trained-heading-implicit-r16        267  100.0%   55.1%   55.1%      +27.0 pp    −43.0 pp  ← best implicit
```

Heading-inference accuracy (derived conditions only — "facing X°"
within 22.5° of GT heading):
```
zs-heading-derived:           27.6 %  (n=134 — 51 % emitted parseable heading)
trained-heading-derived-r4:   58.1 %  (n=265 — 100 % emitted parseable heading)
trained-heading-derived-r8:   62.3 %  (n=265)
trained-heading-derived-r16:  60.4 %  (n=265)
```

#### Headline findings

1. **Given is basically solved** (97-98 % PASS). Base Qwen 2.5 VL 7B at
   44.7 % zero-shot → 98.1 % after r=16 LoRA. Trained model exceeds the
   teacher's own PASS rate (87.5 % on given annotations) — confirms the
   `--only-pass` test set is the cleaner subset.
2. **Derived (compass-free w/ CoT derivation) comes within ~33 pp of
   given** at r=8 (64.9 %). Model learns to read the photo's orientation
   cues (heading-inference acc 58-62 %).
3. **Implicit (purely visual) gains 27 pp** to 55.1 % at r=16 — the
   hardest variant, but still nearly doubles from zero-shot.
4. **Rank-saturation differs per variant**:
   - given: saturated at r=4 (+1 pp to r=16).
   - derived: r=8 best — r=16 slightly regresses (-1.9 pp from r=8).
   - implicit: still climbing monotonically, gains slowing (4 pp r=4→r=8, 1 pp r=8→r=16).
5. **Compass-free thesis holds**: trained-derived (64.9 %) and
   trained-implicit (55.1 %) both far exceed their zero-shot baselines.
   The headline "within ~5-10 pp of trained-given" target was not met
   (gap is 33 / 43 pp respectively) — the model recovers a substantial
   fraction of given's accuracy but not the full amount, suggesting
   numeric heading IS a meaningful signal Qwen 2.5 VL 7B can't fully
   substitute for from photo cues alone.

#### Per-variant rank-saturation curve (PASS by rank)

```
                  r=4         r=8         r=16
─────────────────────────────────────────────────
given (98 % zone)  97.2 %  →  ~97 %  →    98.1 %   ← saturated at r=4
derived            58.5 %  →  64.9 %  →   63.0 %   ← peaks at r=8
implicit           50.2 %  →  54.3 %  →   55.1 %   ← monotonic, slowing
```

#### Resume-training (e3 → e5) val-loss comparison

```
                  val_loss e3 (orig)    val_loss e5 (resumed)    Δ
───────────────────────────────────────────────────────────────────
given-r4          0.2224                0.2159                   −0.0065  (−2.9 %)
given-r8          0.2177                0.2131                   −0.0046  (−2.1 %)
given-r16         0.2130                0.2112                   −0.0018  (−0.8 %)
implicit-r4       0.4135                0.4009                   −0.0126  (−3.0 %)
implicit-r8       0.4031                0.3953                   −0.0078  (−1.9 %)
implicit-r16      0.3923                0.3924                   +0.0001  ( 0.0 %) ← saturated
```

Smaller ranks (r=4, r=8) gained 2-3 % more val_loss reduction with
+2 more epochs; **r=16 has saturated** — implicit-r16 didn't move at
all. The warm-restart cosine LR caused a temporary epoch-4 val_loss
bump in 4 of 6 runs before recovering by epoch 5 (classic
SGDR-warm-restart signature). e5 trained-eval not yet run; deferred
because the e3 PASS rates are already at the saturation ceiling.

### Actual launch commands (Windows — use PowerShell for ALL `modal` calls)

> **⚠ PowerShell required on Windows.** Any `modal` invocation whose
> argument list contains a `/`-prefixed path (volume put/get/ls/rm OR
> `--adapter /ckpts/...` OR `--run-id /...`) is silently mangled by
> Git Bash's path-conversion layer — `/ckpts/foo` becomes
> `C:/Program Files/Git/ckpts/foo` and PEFT then errors with
> `ValueError: Can't find 'adapter_config.json' at 'C:/Program Files/...'`.
> Hit three times during the 2026-06-02 ablation launch. Always run
> Modal commands from PowerShell (or set `MSYS_NO_PATHCONV=1` in Bash).

#### Step 1 — generate `--only-pass` SFT splits (PowerShell or Bash, both fine)

```powershell
python -m src.a2_to_sft --variant given    --only-pass
python -m src.a2_to_sft --variant implicit --only-pass
python -m src.a2_to_sft --variant derived  --only-pass
#  → data/sft/a2_{variant}_{train,val,test}.jsonl
#    given:    2,561 train / 320 val / 320 test
#    implicit: 2,137 train / 267 val / 267 test
#    derived:  2,127 train / 265 val / 265 test
```

#### Step 2 — upload SFT files + any new frames to Modal (PowerShell)

```powershell
$env:PYTHONIOENCODING        = "utf-8"
$env:PYTHONLEGACYWINDOWSSTDIO = "utf-8"
$modal = "C:\Users\z0502\anaconda3\envs\navlm_v2\Scripts\modal.exe"
cd "C:\Users\z0502\Desktop\cs231n\navlm_v2"

foreach ($v in @("given","implicit","derived")) {
  foreach ($s in @("train","val","test")) {
    & $modal volume put navlm-data "data/sft/a2_${v}_${s}.jsonl" `
                                   "/sft/a2_${v}_${s}.jsonl" --force
  }
}

# Frames must also be present at /frames/<video>/<frame>.jpg.
# If new test rows reference frames not yet on the volume, stage them:
& $modal volume put navlm-data _trial_snapshot/frames_to_upload /frames --force
```

#### Step 3 — launch 6 trainings + 2 zs evals in parallel (PowerShell + Start-Process)

`Start-Process -PassThru` keeps PIDs so you can wait on all 8 jobs.
Each `modal run` blocks; running them this way fans them out:

```powershell
$RunId = "ablation_$(Get-Date -Format yyyyMMdd_HHmmss)"
New-Item -ItemType Directory -Force -Path _ablation_logs | Out-Null
"RUN_ID=$RunId" | Out-File -Encoding utf8 "_ablation_logs/RUN_ID"
$jobs = @()

# 6 trainings (A100-80GB)
foreach ($v in @("given","implicit","derived")) {
  foreach ($r in @(4, 8, 16)) {
    $log = "_ablation_logs\train_${v}_r${r}.log"
    $p = Start-Process -FilePath $modal `
        -ArgumentList @("run","src/a2_train_modal.py",
                        "--variant",$v,"--lora-r","$r","--epochs","3") `
        -RedirectStandardOutput $log -RedirectStandardError "${log}.err" `
        -NoNewWindow -PassThru
    $jobs += $p
  }
}

# 3 zero-shot evals (A100-40GB, no adapter needed)
foreach ($v in @("given","implicit","derived")) {
  $log = "_ablation_logs\zs_${v}.log"
  $p = Start-Process -FilePath $modal `
      -ArgumentList @("run","src/a2_eval_modal.py",
                      "--condition","zs-heading-$v","--run-id",$RunId) `
      -RedirectStandardOutput $log -RedirectStandardError "${log}.err" `
      -NoNewWindow -PassThru
  $jobs += $p
}

$jobs | Wait-Process
```

#### Step 4 — once trainings finish, launch 9 trained-evals (PowerShell, MUST keep `/ckpts/` prefix)

```powershell
$jobs2 = @()
foreach ($v in @("given","implicit","derived")) {
  foreach ($r in @(4, 8, 16)) {
    $log = "_ablation_logs\trained_${v}_r${r}_ps.log"
    $p = Start-Process -FilePath $modal `
        -ArgumentList @("run","src/a2_eval_modal.py",
                        "--condition","trained-heading-$v",
                        "--adapter","/ckpts/lora_a2_${v}_r${r}_e3",
                        "--run-id",$RunId) `
        -RedirectStandardOutput $log -RedirectStandardError "${log}.err" `
        -NoNewWindow -PassThru
    $jobs2 += $p
  }
}
$jobs2 | Wait-Process
```

#### Step 5 — pull eval results to local + score (PowerShell)

```powershell
& $modal volume get navlm-eval $RunId eval_pull/ --force
python -m src.a2_score --run-dir "eval_pull/$RunId"
# writes per_sample_scored.jsonl + summary.json per condition;
# prints the 12-row PASS / fmt / dir / h_inf table.
```

### Lessons learned (real bugs hit during the 2026-06-02 ablation launch)

| # | Bug | Fix |
|---|---|---|
| 1 | `modal run a2_train_modal.py --lora-r 4` → `Error: No such option '--lora-r'` — the `local_entrypoint` `main()` didn't expose `lora_r` | Added `lora_r: int = 16, lora_alpha: int = 0` to `main()`, with `lora_alpha = 2*lora_r` default. Commit `6e70ffe`. |
| 2 | The `--only-pass` test set referenced 1,187 unique frames vs the 1,030 from the trial-2 trial cohort | Re-staged 1,216 frames (566 MB) and re-uploaded to `navlm-data:/frames/` via PowerShell. |
| 3 | Trained-eval failed with `Can't find 'adapter_config.json' at '/lora_a2_X'` | DEFAULT_ADAPTER paths in `a2_eval_modal.py` were missing the `/ckpts/` mount prefix. Fixed in commit `d1fb09b`. |
| 4 | Trained-eval re-launched from Bash → `Can't find 'adapter_config.json' at 'C:/Program Files/Git/ckpts/lora_a2_X'` | **Git Bash silently converts `/ckpts/...` arg to a Windows path.** Re-launch from PowerShell with `Start-Process`. **Now the universal rule: all Modal CLI from PowerShell.** |

### Hypotheses

| Comparison | Tests |
|---|---|
| `zs-X` vs `trained-X-r16` (same variant) | Does fine-tuning help? (headline trained-vs-zs effect) |
| `trained-X-r4` vs `r8` vs `r16` (same variant) | Rank-saturation: is more LoRA capacity helping? |
| `trained-given-r16` vs `trained-derived-r16` | Does deriving the heading recover most accuracy? |
| `trained-given-r16` vs `trained-implicit-r16` | Cost of dropping heading entirely (visual-only) |
| `trained-derived-r16` vs `trained-implicit-r16` | Is explicit derivation better than visual-only? |

**Main project result**: if `trained-heading-derived-r16` or
`trained-heading-implicit-r16` PASS is within ~5–10 % of
`trained-heading-given-r16`, the compass-free thesis holds.

---

## 18. Annotation prompt v2 — three independent teacher passes

### Design — supervised distillation with input asymmetry

The Gemini Pro 2.5 teacher labels each `(frame, destination)` pair
THREE TIMES, once per variant — in three independent parallel passes,
one per GCP project (for quota and parallelism). For every pass:

- The **teacher** is given the heading (so it can compute the verb
  correctly via geometry).
- The **student** prompt — what Qwen sees at training and at inference
  — is variant-specific: heading is **shown** for `given` and **hidden**
  for `derived` and `implicit`.

Both the system prompt and the user prompt differ across variants —
all three differ. Per-variant chat shape:

```
                        TEACHER (Gemini, annotation)        STUDENT (Qwen, train + eval)
system_prompt           system_prompt(variant)              system_prompt(variant)   ← same string
user_prompt             teacher_prompt(variant)             student_prompt(variant)
                        (heading ALWAYS shown)              (heading shown only for given)
assistant               teacher's <thinking> + <answer>     model generates this
```

The `system_prompt(variant)` returned by `src/a2_annotate.py` is the
same string at annotation, training, and inference for a given variant
— so the SFT data and the eval inputs share the system context the
teacher was trained against.

### System prompt — common head (shared across variants)

```
You are a Zurich walking-tour guide speaking directly to a tourist who
is looking at the photo right now. Help them take the next step.

Useful Zurich orientation facts you may rely on when reasoning:
- The Limmat river flows roughly south-to-north through central Zurich.
- Grossmünster (twin towers) sits on the EAST bank of the Limmat.
- Fraumünster (single tall spire, green roof) sits on the WEST bank.
- St. Peter (largest clock face in Europe) is on the WEST bank a bit
  north of Fraumünster.
- Bahnhofstrasse runs roughly south-to-north: Hauptbahnhof at the
  NORTH end, Paradeplatz mid-way, Bürkliplatz / Lake Zurich at the
  SOUTH end.
- At midday in Zurich the sun sits in the SOUTH (south-east in
  morning, south-west in afternoon).
- Tram tracks visible on a street tell you the street's axis.

Your reply has two parts: <thinking> for reasoning and <answer> for
the spoken instruction.

<answer> is one sentence speaking DIRECTLY to the walker (use "you"),
pointing to specific things they can see, then the action verb:
  "Can you see X?"  "Look at the X."  "Notice the X ahead."
Reference only landmarks from the "Visible landmarks" list. End with
the action verb on its own short sentence.

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

### System-prompt tail — variant-specific `<thinking>` style rule

```
THINKING_RULE["given"]:
  In <thinking>, write 1-2 short sentences reasoning from the GIVEN
  heading and the route's first-segment bearing to a verb. State both
  numerically (e.g., "I'm facing 95° (east); the route heads 270°
  (west), a 180° rotation, so turn around").

THINKING_RULE["derived"] (4-step):
  STEP 1 (visual cues): I can see [2-3 things in the photo].
  STEP 2 (apply geography): These cues indicate the camera is oriented
                            such that [reasoning].
  STEP 3 (estimated heading): I estimate I'm facing X° (direction).
  STEP 4 (route comparison): Route heads Y° — that's [N°] [direction],
                            so [verb].

THINKING_RULE["implicit"] (3-step, NO numeric heading):
  STEP 1 (what I see): The visible scene contains [2-3 specific things].
  STEP 2 (where the destination is relative to me): The destination is
         "ahead of me / behind me / to my left / to my right" because
         [visual cue]. NO numeric heading.
  STEP 3 (verb decision): Therefore the walker should [verb].
```

The combined `system_prompt(variant) = COMMON_HEAD + "\n" +
THINKING_RULE[variant]` — defined in `src/a2_annotate.py:139`.

### User prompts — teacher (always has heading) vs student

```
                      teacher_prompt(variant)                  student_prompt(variant)
                      ───────────────────────                  ─────────────────────────
HEADING LINE          ALWAYS prepended:                        Only prepended for given:
                      "You are at this location,               "You are at this location,
                       facing 95° (east)."                      facing 95° (east)." (only for given)

SHARED BODY           Destination + walking distance +         Same as teacher
                      OSM route first-segment bearing +
                      Visible landmarks list

TRAILING INSTRUCTION  Variant-specific CoT instruction         Variant-specific instruction
                      (the teacher is reminded to              (the student is told the
                       PRODUCE the variant's CoT style          heading is NOT provided for
                       even though it has the heading)          derived/implicit)
```

#### Example — `given`

```
TEACHER and STUDENT see the same prompt:

You are at this location, facing 95° (east).

Destination: Grossmünster (大教堂), about 287 m walking distance.

OSM walking route:
  First segment heads 270° (west) for 62 m, then 2 more turns
  over a total of 187 m.

Visible landmarks at this spot:
  Limmatquai, Münsterbrücke

Decide the next action verb.
```

#### Example — `derived`

```
STUDENT (heading hidden):

Destination: Grossmünster (大教堂), about 287 m walking distance.

OSM walking route:
  First segment heads 270° (west) for 62 m, then 2 more turns
  over a total of 187 m.

Visible landmarks at this spot:
  Limmatquai, Münsterbrücke

The walker's heading is NOT provided. In <thinking>, FIRST infer the
heading from the photo by stating "I estimate I'm facing X° (direction)",
THEN reason about the route and verb.

TEACHER (same body, BUT heading appended at top + a "write AS IF you
derived this" instruction so the produced CoT still follows the 4-step
template the student will be trained on):

You are at this location, facing 95° (east).

Destination: Grossmünster ... [shared body]

Decide the next action verb. In <thinking>, WRITE AS IF you derived
the heading from the photo: start with "I estimate I'm facing X°
(direction)." using the GIVEN heading value X. Then reason about the
route and verb. Cite visual cues from the photo that support the
heading estimate (shop signs, tram direction, sun position,
recognisable buildings).
```

#### Example — `implicit`

Same pattern: student sees the body alone with a "no numeric heading,
visual-only" instruction. Teacher sees `facing 95°` + a "purely visual,
no numeric heading" instruction. Teacher uses the heading internally to
geometric-derive the correct verb; the produced CoT stays visual-only.

### Why three independent passes (instead of one + transforms)

Earlier we considered running one Gemini pass and deriving the 3
variants by text transforms. That was abandoned because:

1. **Information leak.** The "visual" CoT we wanted for derived/implicit
   needs to be generated *without* heading-aware language; if generated
   from a heading-conditioned answer and edited locally, the visual
   reasoning is implicitly heading-aware.
2. **Verb consistency.** Letting Gemini regenerate the answer per
   variant lets the variant-specific CoT template (4-step / 3-step /
   1-step) actually shape the answer, not just the prefix.

The cost is 3× the teacher API calls, parallelised across 3 separate
GCP projects (`navlm-annot-1-26`, `navlm-annot-2-26`, `navlm-annot-3-26`).

### Per-row schema in `annotations_a2_{variant}.jsonl`

```jsonl
{
  "video": "...",  "frame_id": "...",
  "destination": "Grossmünster", "destination_zh": "大教堂",
  "variant": "given",
  "heading": 95.0,           "gt_verb": "turn around",
  "route_bearing_network": 270.0,
  "first_segment_length_m": 62.0, "n_segments": 3,
  "route_distance_m": 187.4, "sampling_band": "near",
  "visible_landmarks": ["Limmatquai", "Münsterbrücke"],

  "teacher_prompt": "<sent to Gemini>",
  "student_prompt": "<used at SFT and inference>",
  "response":  "<full <thinking>…</thinking><answer>…</answer> as returned>",
  "thinking":  "<extracted from response>",
  "answer":    "<extracted from response>",

  "first_verb": "turn around",
  "format_pass": true,
  "direction_pass": true,
  "PASS": true,
  "truncated": false,
  "derived_heading": null     // only set when the model wrote "facing X°"
}
```

### Measured teacher pass rates (as of 2026-06-02, derived still in flight)

Per-variant rates over the rows annotated so far. `PASS = format_pass
AND direction_pass`. The metrics here are the same 4 used to score the
student (§19) — applied to the teacher's own responses to characterise
the supervisor's reliability.

| Variant | n rows so far | format_pass | direction_pass | **PASS** | h_inf accuracy (n derived headings) |
|---|---:|---:|---:|---:|---:|
| **given** | 3,657 / 3,657 (100 %) | **100.0 %** | 87.5 % | **87.5 %** | n/a |
| **derived** | 3,239 / 3,657 (88.6 %) | 89.3 % | 73.0 % | **72.1 %** | **99.1 %** (n=2,928) |
| **implicit** | 3,657 / 3,657 (100 %) | 99.9 % | 73.1 % | **73.0 %** | n/a |

#### Reading the numbers

1. **`format_pass` ranks: given (100 %) ≫ implicit (99.9 %) > derived
   (89.3 %)**. The 4-step CoT template in derived occasionally
   truncates or misses the closing `</thinking>`/`</answer>` tag —
   ~10.7 % of derived responses fail the strict format check.

2. **`direction_pass` is roughly equal between derived and implicit
   (~73 %)** but well below given (~87.5 %). The teacher always has
   the heading, so the verb is geometrically derivable; the missing
   accuracy comes from the variant-specific CoT instruction nudging
   the model into a reasoning path that occasionally disagrees with
   the geometry (e.g. "the destination is behind me" can still
   resolve to "turn left" if the route's first edge is sideways).

3. **`derived` heading-inference accuracy is 99.1 % at n=2,928** —
   when the derived-variant teacher emits a "facing X°" statement, X
   matches the GT heading within 22.5° 99 % of the time. This is
   expected — the teacher IS shown the heading and just frames its
   reasoning as if derived. The number serves as a sanity check that
   the teacher prompt is producing the intended CoT structure (≥80 %
   of derived rows include the parseable heading statement).

4. **Implication for student PASS upper bound**: the student's
   `direction_pass` cannot exceed the teacher's by much without
   producing labels-disagreement, so realistic targets for the full
   sweep are:
   - `trained-heading-given` ≤ ~87.5 %
   - `trained-heading-derived` ≤ ~72 % (and 22.5°-accurate heading
     inference on a high fraction)
   - `trained-heading-implicit` ≤ ~73 %

   The zero-shot baseline (Qwen 2.5 VL 7B cold, no LoRA) — from the
   trial-2 16-sample eval — is currently:
   - `zs-heading-given` 62.5 %
   - `zs-heading-derived` 18.8 %
   - `zs-heading-implicit` 12.5 %

   So the headroom for LoRA to recover is largest on derived (~53 pp)
   and implicit (~60 pp). Given is closer to teacher (~25 pp headroom)
   and will be the hardest variant to demonstrate large gains on.

5. **The `format_pass` filter for the SFT splits drops ~11 % of
   derived rows** (vs. ~0 % for given/implicit) — `data/sft/a2_*.jsonl`
   will be uneven across variants. With `--only-pass` (additionally
   requires direction_pass), cohorts shrink further: given ~3,200,
   derived ~2,350, implicit ~2,675 (rough projections to full
   annotation completion).

### Whole-dataset teacher-quality analysis (10,614 rows across all 3 variants)

```
WHOLE DATASET
  format_pass    : 10,258 / 10,614  (96.6 %)
  direction_pass :  8,292 / 10,614  (78.1 %)
  overall PASS   :  8,261 / 10,614  (77.8 %)
```

#### By destination attraction — PASS rate

```
destination                n     fail   fail%   PASS%
─────────────────────────────────────────────────────
Lake Zurich              515     179    35 %    65 %   ← worst
Münsterbrücke            498     171    34 %    66 %
Fraumünster              518     150    29 %    71 %
Paradeplatz              573     151    26 %    74 %
Münsterhof               496     130    26 %    74 %
Wasserkirche             474     124    26 %    74 %
Kunsthaus                275      71    26 %    74 %
Helmhaus                 484     119    25 %    75 %
Grossmünster             433     104    24 %    76 %
Bürkliplatz              362      85    23 %    77 %
Bahnhofstrasse           637     146    23 %    77 %
Sechseläutenplatz        405      89    22 %    78 %
Niederdorfstrasse        657     143    22 %    78 %
Limmatquai               639     136    21 %    79 %
Opernhaus                465      94    20 %    80 %
Rathaus                  665     125    19 %    81 %
Lindenhof                587      97    17 %    83 %
Landesmuseum             420      67    16 %    84 %
St. Peter                520      76    15 %    85 %
Limmat river             569      61    11 %    89 %
Hauptbahnhof             422      35     8 %    92 %   ← easiest
```

**Pattern**: river/bank-adjacent destinations (Lake Zurich,
Münsterbrücke, Fraumünster — all on or beside the Limmat) fail most,
because routes there often require a bridge crossing whose verb
depends on which bank the walker is on. **Hauptbahnhof is easiest**
because it sits north of the matched cohort with an unambiguous
bearing.

#### By distance band — PASS rate

```
band                   n      fail   fail%   PASS%
──────────────────────────────────────────────────
near (<500m)         7,438   1,826    25 %    75 %   ← worst (counter-intuitive)
mid  (500-1000m)     2,428     407    17 %    83 %
far  (1000m+)          748     120    16 %    84 %
```

**Counter-intuitive but consistent**: the teacher fails MORE on near
destinations. Two compounding reasons:
1. The 80 % near-band sampling concentrates the `turn around` cases
   (destination just around the corner or behind the walker).
2. Near destinations have short routes (~50-500 m) so small
   heading-noise errors translate to wrong verbs — an 80 m segment is
   much more bearing-sensitive than a 1.2 km segment.

#### By ground-truth verb — PASS rate (most striking finding)

```
gt_verb              n        fail   fail%   PASS%
────────────────────────────────────────────────────
continue ahead    3,091      748    24 %    76 %
turn left         1,706      683    40 %    60 %   ← worst
turn right        2,436      758    31 %    69 %
turn around       3,381      164     5 %    95 %   ← best, but it's a BIAS — see below
```

The "95 % PASS on turn around" looks great in isolation but the
confusion matrix below reveals it's actually a **teacher over-
prediction bias**, not a genuine strength.

#### Confusion matrix — where teacher's verb disagrees with GT

```
GT verb         → teacher verb        count
────────────────────────────────────────────
turn right       → turn around          617    ← teacher over-predicts turn around
turn left        → turn around          593    ← teacher over-predicts turn around
continue ahead   → turn left            454
continue ahead   → turn right           122
continue ahead   → turn around           87
turn right       → continue ahead        60
turn left        → continue ahead        29
turn left        → turn right            18
turn right       → turn left             11
turn around      → continue ahead         3
```

#### **Key finding: TURN-AROUND BIAS**

1,210 of the 1,997 total direction failures (60 %) are cases where
the GT verb is `turn left` or `turn right` and the teacher mis-
predicted `turn around`. The teacher is trigger-happy with the
180°-rotation verb whenever the route bearing is noticeably backward
from the heading — even when geometrically it's only a 90° turn.

**Implications**:

- The 95 % PASS rate on `turn around` is real but it comes WITH a
  large false-positive rate from `turn left/right` cases. The
  precision is high but recall on the OTHER verbs is hurt.
- The SFT data has a **systematic over-representation of `turn around`
  labels** on rows that should be turn left/right. The student will
  inherit this bias.
- `--only-pass` filtering removes the 1,210 mis-classified rows
  (~12 % cohort shrinkage), trading data volume for label cleanliness.
  Worth considering for the final training run.

#### By variant × distance band

```
variant       near (<500m)      mid (500-1000m)     far (1000m+)
──────────────────────────────────────────────────────────────────
given          85 % (n=2571)     92 % (n=827)        96 % (n=259)   ← monotonic — far easiest
derived        71 % (n=2296)     76 % (n=774)        79 % (n=230)   ← monotonic
implicit       70 % (n=2571)     82 % (n=827)        76 % (n=259)   ← non-monotonic
```

`given` is robust across distance bands (monotonic 85 → 96 %).
`derived` follows the same monotonic shape but ~15 pp lower at every
band — consistent with the 4-step derivation costing accuracy
proportionally. `implicit` is non-monotonic: best at mid distance
where landmarks are still visible AND geometry is unambiguous, worse
at near (turn-around bias) and worse at far (no landmarks to anchor
visual reasoning).

### Direction-pass-rate analysis (clean reasoning-quality measure)

Restricting the denominator to **format-passing rows only** isolates
the teacher's reasoning quality from its output-format reliability.

```
FORMAT-PASSING rows: 10,271
  direction_pass rate: 8,270 / 10,271  (80.5 %)
```

#### Precision / recall split per verb — the most diagnostic view

```
RECALL — how often is each GT verb correctly identified
  gt_verb             n      ok     recall
  continue ahead    3,013   2,348    78 %
  turn left         1,666   1,025    62 %   ← worst recall (often mistaken for turn around)
  turn right        2,367   1,678    71 %
  turn around       3,225   3,219   100 %   ← perfect recall ... at a cost (see precision)

PRECISION — when teacher says verb V, how often is it correct?
  pred_verb           n      ok     precision
  continue ahead    2,440   2,348    96 %   ← when teacher says "continue ahead", almost always right
  turn left         1,492   1,025    69 %
  turn right        1,821   1,678    92 %
  turn around       4,518   3,219    71 %   ← worst precision — fires too often
```

**The 100 % recall + 71 % precision on `turn around` IS the bias.**
The teacher catches every real turn-around but ALSO falsely fires
"turn around" on 1,299 cases that should have been turn-left/right.

#### Per-destination direction-pass rate (worst → best)

```
destination                n      ok    dir%
─────────────────────────────────────────────
Lake Zurich              497     337    68 %   ← worst
Münsterbrücke            474     328    69 %
Fraumünster              492     368    75 %
Wasserkirche             465     350    75 %
Münsterhof               483     367    76 %
Paradeplatz              552     423    77 %
Kunsthaus                265     204    77 %
Helmhaus                 467     365    78 %
Grossmünster             417     329    79 %
Bahnhofstrasse           624     493    79 %
Niederdorfstrasse        645     515    80 %
Bürkliplatz              345     277    80 %
Sechseläutenplatz        393     316    80 %
Limmatquai               615     503    82 %
Rathaus                  646     541    84 %
Opernhaus                442     371    84 %
Lindenhof                571     490    86 %
Landesmuseum             407     353    87 %
St. Peter                501     444    89 %
Limmat river             556     508    91 %
Hauptbahnhof             414     388    94 %   ← easiest
```

#### Per-distance-band direction-pass rate

```
band                       n      ok    dir%
─────────────────────────────────────────────
near (<500m)           7,235   5,620    78 %   ← worst (where turn-around cases live)
mid  (500-1000m)       2,318   2,022    87 %
far  (1000m+)            718     628    87 %
```

### Four direction-failure modes — analogies + mechanism

#### 1. The 180° trigger-finger — `turn-around` over-prediction

```
turn-around precision: 71 % (4,518 fires, 1,299 wrong)
contribution to total direction failures: 60 % (1,210 of 1,997)
```

**Analogy**: A taxi driver who hears "destination is behind us
somewhere" and immediately yanks the wheel for a full U-turn — when
actually the destination is just slightly back-and-to-the-side and a
single 90° turn at the next corner would do.

**Mechanism**: the teacher's `<thinking>` compares heading to
first-segment bearing. When `|heading − bearing| > 100°`, the model
gravitates to "180° rotation needed → turn around", missing that a
100-160° gap is more often a 90° left-or-right at the upcoming
intersection.

#### 2. The right-handed gardener — `turn-left` under-recall

```
turn left:  recall 62 % (lowest of all 4 verbs)
turn right: recall 71 %
```

**Analogy**: A right-handed gardener trimming a hedge instinctively
works clockwise. When the route requires a left turn, the model's
preferred verbs are right (handled) or turn-around (the headline
bias) — turn-left is the disfavoured residual.

**Mechanism**: a combination of (a) a mild rightward bias on
ambiguous turn cases (likely from training-data composition) and (b)
when bearing-diff is `−90° to −135°`, the model fires turn-around
instead of left more often than the symmetric right case at
`+90° to +135°`.

#### 3. The "which bank am I on" confusion — river-edge destination failures

```
Lake Zurich     68%    Münsterbrücke   69%
Fraumünster     75%    Wasserkirche    75%    Münsterhof  76%
```

All five are on or beside the Limmat (river + lakeshore).
Hauptbahnhof, by contrast, is 800 m inland with an unambiguous
northward bearing → 94 % direction-pass.

**Analogy**: Asking a tourist on the Münsterbrücke "which way to the
Limmat?" — the river is *right there on both sides*. The verb depends
on whether you walk *along* the bank you're on or *cross* to the
other one. Pro 2.5 can see water in the photo but cannot reliably
tell which bank it's standing on, so when the route says "cross to
the east bank toward Grossmünster", the model sometimes says
"continue ahead toward the river" instead.

**Mechanism**: the multi-target routing (§10) for Lake Zurich and
Limmat picks the network-nearest node, which can be on the OPPOSITE
bank. The correct verb depends on whether a bridge is in the first
50 m of route — a fine detail the photo doesn't always disambiguate.

#### 4. The around-the-corner blindspot — near-distance failure spike

```
near (<500m)         78 % direction-pass
mid  (500-1000m)     87 %
far  (1000m+)        87 %
```

**Analogy**: A walking-tour guide who's great at the long view ("the
cathedral is ten minutes that way, you'll see it") but stumbles on
the immediate-corner case ("the cathedral is right around this
corner — left or right?"). The wide gestures work; the precise
micro-direction at <500 m doesn't.

**Mechanism**: short routes are bearing-sensitive — a 50 m first
segment requires the model to nail the heading to within ~10° to pick
the right verb. Long routes have many segments, so even if the
first-segment bearing is slightly off, the cumulative direction tends
toward "ahead" or a clear cardinal turn.

### Summary — the four direction-failure modes

| Mode | Symptom | Frequency | Where it hits worst |
|---|---|---|---|
| **Turn-around trigger-finger** | 71 % precision on turn-around → 1,299 false positives | Largest contributor (60 % of all direction failures) | All 3 variants, especially derived/implicit |
| **Left under-recall** | turn-left recall 62 % vs turn-right 71 % | 9 pp gap | All 3 variants, slightly worse in implicit |
| **River-bank ambiguity** | Lake Zurich / Münsterbrücke / Fraumünster bottom 3 by direction-pass | 5 destinations, ~25 % of dataset | Worst in derived/implicit (no heading anchor) |
| **Near-distance blindspot** | 9 pp gap near vs mid/far | Concentrated in 70 % of dataset (near band) | All 3, worst in implicit |

---

## 19. Evaluation metrics (v2)

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

### Per-condition reporting (12 rows — 3 zs + 9 trained)

```
                                       n     fmt    dir    PASS   h_inf  h_n
zs-heading-given                       ?      ?      ?      ?      n/a    0
zs-heading-derived                     ?      ?      ?      ?       ?     ?
zs-heading-implicit                    ?      ?      ?      ?      n/a    0
trained-heading-given-r4               ?      ?      ?      ?      n/a    0
trained-heading-given-r8               ?      ?      ?      ?      n/a    0
trained-heading-given-r16              ?      ?      ?      ?      n/a    0
trained-heading-derived-r4             ?      ?      ?      ?       ?     ?
trained-heading-derived-r8             ?      ?      ?      ?       ?     ?
trained-heading-derived-r16            ?      ?      ?      ?       ?     ?
trained-heading-implicit-r4            ?      ?      ?      ?      n/a    0
trained-heading-implicit-r8            ?      ?      ?      ?      n/a    0
trained-heading-implicit-r16           ?      ?      ?      ?      n/a    0
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

### Trial smoke results (2026-06-02, partial — 5 of 6 conditions)

**Strong caveat — these are SMOKE numbers, not the final-report numbers.**
The 3 trained adapters used here were trained with the **trial-2
config: 32 train samples × 1 epoch × r=16 × NO LOSS MASKING** (the
masking and early-stopping fixes were added AFTER these adapters were
trained, see §22). Evaluation is on the same 16-sample subset of the
test split for every condition (`limit=16`).

Real interpretable numbers will come from the full sweep: 9 adapters
trained on ~2,900 samples × 2 epochs with masked loss + early
stopping, evaluated on ~290 samples per condition.

```
condition                        n     fmt     dir    PASS   h_inf   h_n
──────────────────────────────────────────────────────────────────────────
zs-heading-given                 16  100.0%   62.5%   62.5%    n/a    0
trained-heading-given (r16/e1)   16  100.0%   43.8%   43.8%    n/a    0   ← Δ −18.8 pp

zs-heading-derived               16  100.0%   18.8%   18.8%   12.5%   8
trained-heading-derived (r16/e1) 16  100.0%   25.0%   25.0%    8.3%  12   ← Δ  +6.2 pp

zs-heading-implicit              16  100.0%   12.5%   12.5%    n/a    0
trained-heading-implicit (r16/e1)16  100.0%   12.5%   12.5%    n/a    0   ← Δ   0.0 pp
```

Run dir on Modal: `navlm-eval:/trial2_20260602_002923/`.

#### What the smoke says (and what it doesn't)

| Observation | Reading |
|---|---|
| `trained-given` PASS DROPPED 18.8 pp vs zs | Classic small-data + no-masking degradation. The LoRA over-fit to the 32 samples' surface format and unlearned base Qwen's prior geometric reasoning. Loss-masking + ~90× more data + early stopping in the production sweep should fix this. |
| `trained-derived` PASS rose +6.2 pp | LoRA learning the 4-step CoT structure even at this scale. Promising signal — with proper-scale training this delta should grow substantially. |
| `trained-derived` produced "facing X°" in 12/16 rows (vs 8/16 zs) | The variant-specific CoT template IS being adopted. Even though h_inf accuracy is only 8.3 % on those 12 (model writes a heading but it's not within 22.5° of GT), more rows now follow the expected structure. |
| `trained-implicit` PASS unchanged (12.5 % = zs) | Null result at this scale. Two readings: (1) implicit has the hardest visual-only task and 32 train rows are simply not enough to move the needle, or (2) the 3-step CoT template needs more capacity than r=16 can fit on tiny data. Production-scale training should disambiguate. |
| `format_pass = 100 %` everywhere | Base Qwen already obeys the `<thinking>…</thinking><answer>…</answer>` structure cold. The format metric will plateau at 100 % across the full sweep and won't be the differentiator. |

#### What we cannot conclude from smoke

- **Statistical significance** — n=16 has ±12.5 pp per-row precision (one
  flip moves PASS by 6.25 pp). Production eval (~290 samples) will be
  ~4× tighter (±3 pp).
- **Final ranking of variants** — rank-sweep + masked-loss + full data
  may invert the trained-given drop.
- **Heading-inference comparison** — h_inf at n=8-12 is noisy; the
  full eval will have n ≈ 290 and meaningful precision.

The headline scientific question — does compass-free
(`trained-heading-derived` / `-implicit`) come within 5-10 % of
`trained-heading-given` — cannot be answered from this smoke run. It
needs the full sweep, which is gated on annotation completion (~3 h
remaining as of writing).

### Cost & wall-time estimate (updated 2026-06-01)

Measured throughput from the in-flight annotation runs and from the
Modal pricing table (A100-80GB ≈ $3.46/hr, A100-40GB ≈ $2.10/hr).

#### Annotation (Gemini Pro 2.5, 3 parallel passes, one per variant)

```
Per-variant: ~3,657 calls × variant-specific input/output token counts.
Measured throughput:
  given     : ~268 rows/hr  → 13.6 h full pass · ~$28
  derived   : ~205 rows/hr  → 17.8 h full pass · ~$34 (longer 4-step CoT)
  implicit  : ~256 rows/hr  → 14.3 h full pass · ~$28
Parallel wall-time (3 GCP projects in parallel): ~18 h (derived bottleneck)
Combined cost: ~$90 (range $85-100 depending on token usage)
```

#### SFT conversion + upload

```
Local Python, < 1 min per variant · $0
modal volume put navlm-data data/sft/a2_*.jsonl /sft/  (~30 MB) · $0
```

#### Training — full LoRA rank sweep (9 adapters, Modal A100-80GB)

```
Per-LoRA adapter (one variant × one rank):
  - model load + warmup            : ~10 min
  - SFT training:
      ~2,900 train rows × 2 epochs
      per_device_batch=1 grad_accum=8  → ~725 optimisation steps
      ~3-5 s/step                     → ~40-60 min
  - save adapter + commit volume   : ~1 min
  TOTAL wall-time per adapter      : ~55-75 min (~1 h)
  Per-adapter cost (A100-80GB)     : $3.46-4.30

9 adapters (3 variants × 3 ranks):
  Sequential wall-time             : ~9-11 h
  Parallel wall-time (Modal)       : ~1-1.5 h (Modal can run 9 in parallel)
  Combined cost                    : ~$30-40
```

Rank does not change wall time meaningfully — r=4, 8, 16 differ only in
adapter param count, not forward/backward cost.

`eval_loss` from training is **masked-loss** (see §22) — it averages
cross-entropy only over `<thinking>` + `<answer>` tokens, NOT over the
system+user prompt. This makes the number directly comparable across
runs (variant × rank) and meaningful as a model-selection signal. The
2026-06-02 overfit test showed that the pre-masking eval_loss was
misleadingly low (it averaged in the trivial-to-predict system+user
tokens); the post-masking eval_loss reveals real overfitting as a
U-shaped curve over epochs.

#### Eval — 12 conditions on Modal A100-40GB

```
Per-condition (one of 12):
  - model load                     : ~5 min
  - adapter load (trained only)    : ~30 s
  - inference (~290 test samples)  :
      derived condition  : ~15-20 s/sample (4-step CoT, longer outputs)
                          → ~75-95 min
      given/implicit     : ~8-12 s/sample (shorter outputs)
                          → ~40-60 min
  TOTAL wall-time per condition    : ~50-100 min
  Per-condition cost (A100-40GB)   : $1.75-3.50

Zero-shot eval (3 conditions):
  Wall-time (parallel)             : ~1.5 h
  Cost                             : ~$6

Trained eval (9 conditions):
  Wall-time (parallel)             : ~1.5 h
  Cost                             : ~$20

Pull (`modal volume get navlm-eval ...`) + local scoring  : $0
```

#### Grand total

```
Annotation              ~$90
Training (9 LoRAs)      ~$35
Zero-shot eval (3)       ~$6
Trained eval (9)        ~$20
─────────────────────────────
Total Modal + Gemini    ~$150
Wall-time (parallel)    ~22 h (annotation bottleneck)
```

Gemini Pro 2.5 is only invoked during teacher annotation. All Qwen
evaluation (including zero-shot) runs on Modal A100s — same Modal app
as training, separate function.

---

## 20. Per-variant SFT conversion (`src/a2_to_sft.py`)

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

### Worked examples — one real training row per variant

Each row's `messages` array has 3 turns: `system` (the variant-specific
prompt from §18), `user` (image + variant-specific student prompt), and
`assistant` (the teacher's `<thinking>` + `<answer>`). At inference time
the assistant turn is stripped and the model generates it.

Examples below are real rows from the trial-snapshot SFT files
(`data/sft/a2_<variant>_train.jsonl`, snapshot at ~80% annotation
completion). System prompt is abbreviated as `<system_prompt(variant)>`
(see §18 for the full string).

#### Example — variant=`given`

```
video=old_town_limmat  frame=frame_01171
dest=Limmat river (利马特河)   heading=270°   gt_verb=turn around

[system] <system_prompt("given")>  ← COMMON_HEAD + THINKING_RULE["given"]

[user]   <IMAGE>
         You are at this location, facing 270° (west).

         Destination: Limmat river (利马特河), about 116 m walking distance.

         OSM walking route:
           First segment heads 89° (east) for 45 m, then 5 more turns
           over a total of 116 m.

         Visible landmarks at this spot:
           (no notable landmarks listed)

         Decide the next action verb.

[assistant]
<thinking>
I'm facing 270° (west); the route heads 89° (east), which is a 181°
rotation, so I need to turn around.
</thinking>
<answer>
There is no clear landmark in front of you. Turn around.
</answer>
```

The student sees the heading (270°) and the route bearing (89°) and
just has to do the angle subtraction. Heading-given is the easiest
variant.

#### Example — variant=`derived`

```
video=looks_perfect  frame=frame_02463
dest=Münsterbrücke (大教堂桥)   heading=140.1° (HIDDEN)   gt_verb=turn around

[system] <system_prompt("derived")>  ← COMMON_HEAD + 4-step THINKING_RULE

[user]   <IMAGE>
         Destination: Münsterbrücke (大教堂桥), about 258 m walking distance.

         OSM walking route:
           First segment heads 329° (north-northwest) for 32 m,
           then 11 more turns over a total of 258 m.

         Visible landmarks at this spot:
           (no notable landmarks listed)

         The walker's heading is NOT provided. In <thinking>, FIRST
         infer the heading from the photo by stating "I estimate I'm
         facing X° (direction)", THEN reason about the route and verb.

[assistant]
<thinking>
STEP 1 (visual cues): I can see the H. Schwarzenbach Colonialwaren
shop at number 19 on a narrow cobblestone street. In the distance, a
street sign for Spiegelgasse is visible. The buildings are colorful
and typical of Zurich's old town.
STEP 2 (apply geography): The shop H. Schwarzenbach is located at
Münstergasse 19, on the east bank of the Limmat. Looking down the
street from this address towards the intersection with Spiegelgasse
means I am looking southeast along Münstergasse.
STEP 3 (estimated heading): I estimate I'm facing 140° (southeast).
STEP 4 (route comparison): The route's first segment heads 329°
(north-northwest). This is almost exactly the opposite direction of
my current heading (140°). The required action is to turn 180 degrees.
The verb is turn around.
</thinking>
<answer>
There is no clear landmark in front of you. Turn around.
</answer>
```

The student does NOT see the heading. The teacher (which had access to
heading=140.1°) had to produce the 4-step derivation as if it had
derived 140° from the photo cues — and it did, exactly. Note that the
STEP-3 estimate (140°) matches the GT heading (140.1°) almost
perfectly, which is the strongest possible signal for the student:
"the photo's cues are sufficient to recover this heading."

#### Example — variant=`implicit`

```
video=looks_perfect  frame=frame_01886
dest=Limmatquai (利马特河滨道)   heading=220.2° (HIDDEN)   gt_verb=turn around

[system] <system_prompt("implicit")>  ← COMMON_HEAD + 3-step purely-visual rule

[user]   <IMAGE>
         Destination: Limmatquai (利马特河滨道), about 98 m walking distance.

         OSM walking route:
           First segment heads 67° (east-northeast) for 16 m,
           then 5 more turns over a total of 98 m.

         Visible landmarks at this spot:
           Fraumünster, Münsterhof

         The walker's heading is NOT provided. Reason from visual cues
         in the photo about where the destination is and which verb is
         needed. Do NOT state a numeric heading.

[assistant]
<thinking>
STEP 1 (what I see): The scene shows the large, open cobblestone
square of Münsterhof, with a fountain and buildings across the way.
The corner of the Fraumünster church is visible on the far left.
STEP 2 (where the destination is relative to me): The destination,
Limmatquai, is the street along the river. The river is located behind
me, on the other side of the Fraumünster. I am currently facing away
from it.
STEP 3 (verb decision): Therefore the walker should turn around.
</thinking>
<answer>
You are looking across the open square of Münsterhof. Turn around.
</answer>
```

No numeric heading anywhere — neither in the prompt nor in the
reasoning. The model navigates by spatial reasoning over the
"Visible landmarks" list ("Fraumünster on the left + Limmatquai is
behind that" → turn around).

### What "test row" looks like

The test rows have the **same 3-message structure**, but at inference
the harness strips the assistant turn:

```python
# src/a2_eval_modal.py:152
messages_for_inference = [m for m in row["messages"]
                          if m["role"] != "assistant"]
```

…feeds `[system, user]` to the model, and decodes only the
newly-generated tokens. The teacher's response in the test row's
`messages[2]` is kept solely so the row remains self-contained for
offline inspection — it is never fed to the model.

The `gt_verb` field on each row is computed deterministically from
the OSM route (§11) — independent of what the teacher wrote — and is
what `src/a2_score.py` compares the model's `first_verb` against.

---

## 21. Modal infrastructure — upload, download, run

All training and evaluation happens on **Modal** (serverless GPU
compute). The local machine produces SFT data + frame images and pulls
eval results back; everything else lives on Modal volumes.

### 21.1 The 3 navlm Modal volumes

| Volume | Mounted at | What lives there |
|---|---|---|
| `navlm-data` | `/data` (in containers) | SFT JSONL: `/sft/a2_<v>_<split>.jsonl`<br>Frame images: `/frames/<video>/<frame_id>.jpg` |
| `navlm-ckpts` | `/ckpts` | LoRA adapters: `/ckpts/lora_a2_<v>_r<r>_e<e>/{adapter_model.safetensors, adapter_config.json, summary.json, history.json}` (volume root has them at `lora_a2_<v>_r<r>_e<e>/`; the `/ckpts/` prefix is the in-container mount path needed by `--adapter`) |
| `navlm-eval` | `/eval` | Per-condition eval outputs: `/<run_id>/<condition>/{per_sample.jsonl, summary.json}` |

When `navlm-data` is mounted at `/data` inside a Modal container, its
`/frames/<video>/...jpg` files appear at `/data/frames/<video>/...jpg`
— which is what `FRAMES_ROOT = "/data/frames"` in
`src/a2_{train,eval}_modal.py` expects.

```python
# src/a2_train_modal.py + src/a2_eval_modal.py
ckpts    = modal.Volume.from_name("navlm-ckpts", create_if_missing=True)
data_vol = modal.Volume.from_name("navlm-data",  create_if_missing=True)
eval_vol = modal.Volume.from_name("navlm-eval",  create_if_missing=True)
```

### 21.2 The 2 Modal apps

| App name | Defined in | Function | GPU | What it does |
|---|---|---|---|---|
| `navlm-train-a2` | `src/a2_train_modal.py` | `train_lora` | A100-80GB | SFT one LoRA adapter for one (variant, rank) |
| `navlm-eval-a2` | `src/a2_eval_modal.py` | `evaluate_condition` | A100-40GB | Run inference for one of 12 conditions |

Both apps mount all 3 volumes (read for inputs, write for outputs).

### 21.3 CLI cheat sheet (PowerShell, recommended on Windows)

Set this once per shell session — Modal CLI emits Unicode glyphs
(`✓`, box-drawing) that crash Windows `cp1252` stdout without it:

```powershell
$env:PYTHONIOENCODING        = "utf-8"
$env:PYTHONLEGACYWINDOWSSTDIO = "utf-8"
$modal = "C:\Users\z0502\anaconda3\envs\navlm_v2\Scripts\modal.exe"
```

#### Upload (local → volume)

Single file:
```powershell
& $modal volume put navlm-data data/sft/a2_given_train.jsonl /sft/a2_given_train.jsonl --force
```

Whole directory (recursive):
```powershell
& $modal volume put navlm-data _trial_snapshot/frames_to_upload /frames --force
```

`--force` overwrites existing files on the volume.

#### List (inspect what's on a volume)

```powershell
& $modal volume ls navlm-data /sft                   # SFT splits present
& $modal volume ls navlm-data /frames                # top-level video dirs
& $modal volume ls navlm-data /frames/old_town_limmat  # per-video frames
& $modal volume ls navlm-ckpts /                     # trained adapters
& $modal volume ls navlm-eval  /                     # eval run_ids
```

#### Download (volume → local)

```powershell
& $modal volume get navlm-eval  <run_id> ./eval_pull/
& $modal volume get navlm-ckpts /lora_a2_given_r16_e2 ./adapters/
```

#### Delete

```powershell
& $modal volume rm navlm-data /sft/old.jsonl
& $modal volume rm navlm-data /some_dir --recursive
```

#### Run a Modal-app function

```powershell
& $modal run src/a2_train_modal.py --variant given --lora-r 16 --epochs 2
& $modal run src/a2_eval_modal.py  --condition zs-heading-given --limit 16
& $modal run src/a2_eval_modal.py  --condition trained-heading-derived `
                                    --adapter /ckpts/lora_a2_derived_r8_e2
```

`modal run` BLOCKS until the function returns. Launch multiple in
parallel by backgrounding each one (Bash `&` + `wait`, or several
PowerShell terminals).

### 21.4 The training script — what `modal run` actually triggers

`src/a2_train_modal.py`:

```python
app = modal.App("navlm-train-a2")
train_image = (modal.Image.debian_slim(python_version="3.11")
               .pip_install("torch==2.5.1", "transformers>=4.49", "peft>=0.13",
                            "bitsandbytes>=0.44", ...))

@app.function(image=train_image, gpu="A100-80GB", timeout=6*3600,
              volumes={"/ckpts": ckpts, "/data": data_vol},
              secrets=[modal.Secret.from_name("huggingface")])
def train_lora(variant, epochs=2, lr=2e-4, lora_r=16, lora_alpha=32, limit=0):
    train_path = Path(f"/data/sft/a2_{variant}_train.jsonl")
    val_path   = Path(f"/data/sft/a2_{variant}_val.jsonl")
    # 1. load 4-bit base Qwen 2.5 VL 7B
    # 2. attach LoRA (target q/k/v/o_proj, r=lora_r, alpha=lora_alpha)
    # 3. HF Trainer with per-image collate that opens /data/frames/<video>/<frame>.jpg
    # 4. save adapter to /ckpts/lora_a2_<v>_r<r>_e<e>/
    ckpts.commit()       # persist writes back to the navlm-ckpts volume

@app.local_entrypoint()
def main(variant="given", epochs=2, lr=2e-4, limit=0):
    result = train_lora.remote(variant=variant, epochs=epochs, lr=lr,
                               limit=limit)
    # prints the adapter path + the `modal volume get` command to pull it
```

The `.remote()` call serialises the args, ships them to a freshly-spun
A100-80GB container, and BLOCKS until `train_lora` returns. Logs from
the remote function stream back to your terminal in real time.

`ckpts.commit()` is critical — without it, writes to `/ckpts/...`
inside the container would be lost when the container terminates.

### 21.5 The inference script — what eval does

`src/a2_eval_modal.py`:

```python
app = modal.App("navlm-eval-a2")
# same image set as train_image plus inference deps

@app.function(image=eval_image, gpu="A100-40GB", timeout=3*3600,
              volumes={"/ckpts": ckpts, "/data": data_vol, "/eval": eval_vol},
              secrets=[modal.Secret.from_name("huggingface")])
def evaluate_condition(condition, run_id, adapter="", max_new_tokens=4096,
                        temperature=0.0, limit=0):
    variant = CONDITION_TO_VARIANT[condition]      # given / derived / implicit
    is_trained = condition.startswith("trained-")
    rows = [json.loads(l) for l in
            Path(f"/data/sft/a2_{variant}_test.jsonl").open()]
    # 1. load 4-bit base Qwen
    # 2. if trained: PeftModel.from_pretrained(model, adapter)
    # 3. per-row: strip assistant turn, apply chat template, run model.generate
    # 4. write /eval/<run_id>/<condition>/per_sample.jsonl
    eval_vol.commit()
```

Each row's output goes one-line-JSON into `per_sample.jsonl` with
fields the local scorer needs: `model_response`, `gt_verb`, `heading`,
`condition`, `video`, `frame_id`.

### 21.6 End-to-end data flow

```
LOCAL                                       MODAL
─────────────────────────                   ─────────────────────────────

data/sft/a2_<v>_<split>.jsonl  ─────┐
data/cities/zurich/frames/<v>/*.jpg ─┤
                                     │
                              modal volume put (upload)
                                     │
                                     ▼
                                  navlm-data
                                  ├ /sft/a2_<v>_<split>.jsonl
                                  └ /frames/<v>/*.jpg

                                  modal run src/a2_train_modal.py --variant <v> --lora-r <r>
                                     │
                                     ▼  reads /data/sft/, /data/frames/
                                  navlm-ckpts:/lora_a2_<v>_r<r>_e<e>/

                                  modal run src/a2_eval_modal.py --condition <cond> [--adapter ...]
                                     │
                                     ▼  reads /data/sft/, /data/frames/,
                                        and /ckpts/lora_a2_... (if trained)
                                  navlm-eval:/<run_id>/<cond>/per_sample.jsonl

                              modal volume get (download)
                                     │
                                     ▼
eval_pull/<run_id>/<cond>/per_sample.jsonl
                                     │
                              python -m src.a2_score
                                     │
                                     ▼
eval_pull/<run_id>/<cond>/per_sample_scored.jsonl
                          summary.json
                          summary_table.txt   ← final-report input
```

### 21.7 Windows gotchas (real ones we hit)

#### Git Bash path conversion

Git Bash auto-converts an argument that starts with `/` into a Windows
path before passing it to `modal.exe`. So this:

```bash
modal volume put navlm-data ./frames /frames     # Git Bash
```

uploads to a folder literally named `C:/Program Files/Git/frames` on
the Modal volume (not the intended `/frames`). Symptoms: training jobs
fail with `No such file or directory: /data/frames/...`.

Fixes:

```bash
modal volume put navlm-data ./frames //frames    # Git Bash: double slash bypasses conversion
MSYS_NO_PATHCONV=1 modal volume put ...          # or set this env var
```

```powershell
& $modal volume put navlm-data ./frames /frames  # PowerShell — no conversion
```

**Recommended: do all `modal volume` operations from PowerShell.**

#### UTF-8 stdout

Set `PYTHONIOENCODING=utf-8` + `PYTHONLEGACYWINDOWSSTDIO=utf-8` before
running Modal CLI on Windows. Without it, the `✓` glyph in the
"uploaded" confirmation crashes the CLI on PowerShell's default
`cp1252`. The upload may still finish, but the exit code is non-zero
and you lose the path confirmation in the output.

#### Frames must be uploaded separately from SFT JSONL

SFT JSONL files reference frames by `image_rel = "<video>/<frame>.jpg"`;
the train/eval scripts join that against `FRAMES_ROOT = "/data/frames"`.
**Uploading the SFT files alone is insufficient** — you must also push
the referenced frame images to `navlm-data:/frames/`. The trial-run
discovery on 2026-06-02 was that this step had never been done for the
Attempt-2 matched cohort; ~70-100 % of test rows skipped with
`No such file or directory` until the 517 MB / 1,030-frame upload
fixed it.

---

## 22. LoRA training pipeline (`src/a2_train_modal.py`)

Modal A100-80GB, 4-bit NF4 base + BF16 LoRA. One Modal function call
per (variant, rank) — 9 calls total for the full sweep.

### Hyperparameters — full setup

```
─────────────────────────────── MODEL / QUANTISATION ──────────────────────
base model               : Qwen/Qwen2.5-VL-7B-Instruct
quantisation             : NF4 (4-bit) base, BF16 LoRA
  bnb_4bit_quant_type    : "nf4"
  bnb_4bit_compute_dtype : bfloat16
  bnb_4bit_use_double_quant: true
processor max_pixels     : 448 × 448

─────────────────────────────── LORA (PEFT) ───────────────────────────────
task_type                : CAUSAL_LM
target_modules           : q_proj, k_proj, v_proj, o_proj
r (rank, SWEPT)          : 4 / 8 / 16
alpha (= 2 × r)          : 8 / 16 / 32   (rank-invariant per-weight LR)
dropout                  : 0.05

─────────────────────────────── OPTIMISER ─────────────────────────────────
optim                    : adamw_torch (HF default — NOT 8-bit adam)
adam_beta1, beta2, eps   : 0.9, 0.999, 1e-8 (HF defaults — not overridden)
weight_decay             : 0.0 (HF default — NOT regularised; relying on LoRA-r as capacity bottleneck + early stop)
max_grad_norm            : 1.0 (HF default — gradient clipping enabled)

─────────────────────────────── LR SCHEDULE ───────────────────────────────
learning_rate            : 2e-4
lr_scheduler_type        : "cosine"          ← annealing 2e-4 → 0 over total steps
warmup_ratio             : 0.03              ← 3 % linear warmup 0 → 2e-4
                                              (e.g. 24 warmup steps on an 800-step run)

─────────────────────────────── BATCHING ──────────────────────────────────
per_device_train_batch_size: 1
per_device_eval_batch_size : 1
gradient_accumulation_steps: 8                → effective batch size 8
bf16                       : true (forward + backward + optim states)

─────────────────────────────── LOSS — SUPERVISOR ONLY ON OUTPUT ─────────
labels = input_ids.clone()
labels[pad_token]                    = -100
labels[<|image_pad|>, <|vision_*|>]  = -100
labels[< system tokens >]            = -100  ← masked
labels[< user tokens >]              = -100  ← masked
labels[< assistant tokens >]         = original tokens (loss flows here)

  → cross-entropy is computed ONLY over tokens inside the assistant turn
    (<thinking>…</thinking><answer>…verb.</answer>), per `collate()` in
    src/a2_train_modal.py:110-149. See "Loss masking" subsection below.

─────────────────────────────── EARLY STOP + BEST CKPT ────────────────────
num_train_epochs           : 3 (CAP — early-stop may halt sooner)
eval_strategy              : "epoch"             (eval each epoch)
save_strategy              : "epoch"             (1 checkpoint per epoch)
save_total_limit           : 3                   (oldest pruned)
load_best_model_at_end     : true                (restore best ckpt)
metric_for_best_model      : "eval_loss"         (the MASKED version)
greater_is_better          : false
callbacks                  : [EarlyStoppingCallback(early_stopping_patience=2)]

─────────────────────────────── DATA / FILTER ─────────────────────────────
train data filter          : --only-pass (format_pass AND direction_pass)
per-variant counts         : given 2,561 train / 320 val / 320 test
                             implicit 2,137 / 267 / 267
                             derived ~2,118 / ~265 / ~265 (projected)
```

LoRA alpha is set to `2 × rank` (standard practice) so per-weight
effective learning rate is rank-invariant; we are sweeping capacity,
not effective LR. Weight decay is `0` and LoRA dropout is `0.05` — the
only regularisation comes from the small LoRA rank and early stop on
the masked val loss.

### Loss masking — `<thinking>` + `<answer>` only

The chat-template-rendered training row has 3 turns: `system`, `user`,
and `assistant`. Naively, HuggingFace Trainer computes the loss over
ALL non-padding tokens — including the (identical-across-rows) system
prompt and the user prompt. That makes the loss number dominated by
constant prompt text the model learns to predict in a few steps, and
makes `eval_loss` near-monotone-decreasing even when the model is
overfitting on the assistant tokens we actually care about.

The collate function therefore masks the system + user tokens out of
the loss by finding `<|im_start|>assistant\n` in each row's
`input_ids` and setting `labels[:asst_start] = -100`. After this, the
loss is computed over **only the assistant turn** — i.e. exactly the
`<thinking>...</thinking><answer>...verb.</answer>` tokens.

```python
# src/a2_train_modal.py — collate()
asst_prefix_ids = processor.tokenizer.encode(
    "<|im_start|>assistant\n", add_special_tokens=False)
# inside collate, per row:
for i in range(labels.shape[0]):
    seq = enc["input_ids"][i].tolist()
    for j in range(len(seq) - len(asst_prefix_ids) + 1):
        if seq[j:j+len(asst_prefix_ids)] == asst_prefix_ids:
            labels[i, :j + len(asst_prefix_ids)] = -100
            break
```

Empirical impact (overfit smoke test, 32 train / 4 val / 30 epochs,
variant=given, r=16):

| Epoch | UNMASKED train | UNMASKED eval | MASKED train | MASKED eval |
|---:|---:|---:|---:|---:|
| 1  | 4.84  | 4.38  | 0.60  | 0.77 |
| 5  | 1.87  | 1.32  | 0.36  | **0.52** ← min |
| 9  | 0.19  | 0.18  | 0.13  | 0.55 |
| 15 | 0.08  | 0.12  | 0.04  | 0.64 |
| 20 | 0.06  | 0.12  | 0.02  | 0.82 |
| 30 | 0.05  | 0.13  | 0.005 | 0.87 |

The unmasked eval_loss looked like a healthy plateau; the masked
eval_loss reveals classic overfitting — minimum at epoch ~5 then
climbing. Early stopping (patience=2) would have caught this at
epoch ~7.

### Early stopping

`EarlyStoppingCallback(early_stopping_patience=2)` halts training when
`eval_loss` (the masked version) fails to improve for 2 consecutive
epochs. `load_best_model_at_end=True` then restores the checkpoint
with the lowest eval_loss before `model.save_pretrained()`. For the
full sweep (~2,900 train rows) we set `epochs=2` as the cap; if the
val loss starts climbing within those 2 epochs, the best checkpoint
is what gets saved.

### Measured training + validation loss (2026-06-02 ablation run, 6 completed adapters)

All 6 trainings ran the full 3-epoch cap — **early stop never fired**
because masked `eval_loss` improved monotonically every epoch
(diff was small in the last epoch, suggesting we're near but not at
the overfit cliff at 3 epochs on real-scale data).

#### Per-epoch masked val_loss (lower = better)

```
                  epoch 1     epoch 2     epoch 3 (final)    Δ (ep1→3)
─────────────────────────────────────────────────────────────────────
given-r4          0.2517      0.2263      0.2224              -0.0293
given-r8          0.2451      0.2216      0.2177              -0.0274
given-r16         0.2422      0.2168      0.2130              -0.0292
implicit-r4       0.4507      0.4184      0.4135              -0.0372
implicit-r8       0.4383      0.4066      0.4031              -0.0352
implicit-r16      0.4268      0.3942      0.3923              -0.0345
```

#### Final (best) val_loss + rank-saturation effect

```
                  r=4         r=8         r=16        Δ (r=4→r=16)
─────────────────────────────────────────────────────────────────
given             0.2224      0.2177      0.2130       -0.0094  (4.2%)
implicit          0.4135      0.4031      0.3923       -0.0212  (5.1%)
```

Doubling rank from 4 → 16 (4× parameters) buys only 4-5 % val-loss
reduction. **Mild rank-saturation past r=8** — the LoRA capacity is
not the bottleneck. Larger gains would likely come from more epochs,
more data, or richer prompts rather than more rank.

#### Training loss trajectory (decimated to per-quarter-epoch)

Train loss is computed on the assistant-token-only labels (same
mask as eval). Values shown are the most recent logged train loss
sampled at each quarter-epoch mark.

```
              ep 0.0   ep 0.5   ep 1.0   ep 1.5   ep 2.0   ep 2.5   ep 3.0
─────────────────────────────────────────────────────────────────────────
given-r4      0.79     0.23     0.23     0.21     0.21     0.19     0.19
given-r8      0.78     0.23     0.23     0.21     0.20     0.18     0.18
given-r16     0.77     0.22     0.23     0.21     0.19     0.17     0.18
implicit-r4   1.01     0.49     0.44     0.41     0.39     0.37     0.38
implicit-r8   1.01     0.48     0.44     0.41     0.37     0.34     0.36
implicit-r16  0.99     0.47     0.42     0.38     0.35     0.31     0.33
```

#### Observations

1. **Steepest learning happens in the first quarter-epoch** — train
   loss drops from ~0.8 (given) or ~1.0 (implicit) to ~0.2 / ~0.5
   within 60-80 optimisation steps. After that, gradual annealing.

2. **Train vs val gap is small** — at epoch 3, train loss is roughly
   85 % of val loss (e.g. given-r16: train 0.18 vs val 0.21). No
   overfit detected at 3 epochs on this data size; 4-5 epochs is
   likely safe before the curve U-turns.

3. **Implicit's loss is ~2× given's at every checkpoint** — confirms
   the implicit task is intrinsically harder (no heading anchor → the
   model must spend representational capacity on visual-only spatial
   reasoning, which is genuinely more uncertain).

4. **Rank-saturation onset is around r=8** — the r=8 → r=16 step buys
   roughly half of what r=4 → r=8 did. r=4 is probably the right
   choice for production if minimising LoRA size matters.

5. **Cosine LR schedule does its job** — late-epoch train loss
   continues to drop noticeably (epoch 2.5 → 3.0: ~5 % drop), which
   wouldn't happen with a flat or step LR.

### Continue-training support — `--resume-adapter` (added 2026-06-02)

Because all 6 e3 trainings showed monotonic val_loss decrease (no
overfit U-turn, no early-stop fired), there was reason to suspect
more epochs would still buy meaningful loss reduction. The training
script was extended with a `--resume-adapter <path>` CLI flag (Mode B
— "load saved LoRA + fresh optimizer/LR + continue training"):

```bash
modal run src/a2_train_modal.py \
    --variant given --lora-r 4 \
    --resume-adapter /ckpts/lora_a2_given_r4_e3 \
    --epochs 2
# → loads existing r=4 e3 adapter, trains +2 more epochs (fresh cosine LR)
# → writes /ckpts/lora_a2_given_r4_e5/   (orig 3 + new 2 = 5 total)
```

Behaviour:
- LoRA weights start from the saved adapter (not random init).
- **Optimizer + LR schedule are fresh** — cosine LR restarts from peak
  2e-4 over the new total step count (a "warm restart" — sometimes
  helps escape local minima, sometimes destabilises; comparable to
  SGDR's warm restart).
- The adapter's own `r`, `alpha`, `dropout`, `target_modules`
  override the CLI flags so you can't accidentally rank-mismatch.
- Output dir auto-derived: parses `_e<N>` from the input adapter path
  and writes to `/ckpts/lora_a2_<variant>_r<r>_e<orig+new>/`.
- `summary.json` records `resume_adapter`, `resume_orig_epochs`,
  `epochs_this_run`, `total_epochs` for lineage.

NOT supported:
- True checkpoint resume (HF Trainer `resume_from_checkpoint=` to
  continue optimizer state + step counter). Would be a small
  additional CLI flag if needed for crash resilience.

### In-flight experiment: e3 → e5 resume across all 6 given+implicit adapters

Launched 2026-06-02. Each adapter from the original 3-epoch run is
being extended by 2 more epochs (PowerShell `Start-Process` parallel
fan-out — 6 jobs concurrent on Modal A100-80GB):

```
condition         input adapter (e3)         output adapter (e5)
─────────────────────────────────────────────────────────────────
given-r4 +e2      lora_a2_given_r4_e3     →  lora_a2_given_r4_e5
given-r8 +e2      lora_a2_given_r8_e3     →  lora_a2_given_r8_e5
given-r16 +e2     lora_a2_given_r16_e3    →  lora_a2_given_r16_e5
implicit-r4 +e2   lora_a2_implicit_r4_e3  →  lora_a2_implicit_r4_e5
implicit-r8 +e2   lora_a2_implicit_r8_e3  →  lora_a2_implicit_r8_e5
implicit-r16 +e2  lora_a2_implicit_r16_e3 →  lora_a2_implicit_r16_e5
```

Per-adapter wall-time: ~45-55 min (vs. ~65 min for fresh `--epochs 5`
since the first 3 epochs of compute are reused).
Total cost: ~$15-20.

Reporting expectations (will be filled in once the run finishes):
| | val_loss e3 (existing) | val_loss e5 (resumed +2) | Δ |
|---|---:|---:|---:|
| given-r4 | 0.2224 | ? | ? |
| given-r8 | 0.2177 | ? | ? |
| given-r16 | 0.2130 | ? | ? |
| implicit-r4 | 0.4135 | ? | ? |
| implicit-r8 | 0.4031 | ? | ? |
| implicit-r16 | 0.3923 | ? | ? |

If `Δ_e3→e5 ≈ 0.005-0.01` per adapter, the additional epochs were
worth it (similar marginal benefit to e2 → e3). If `Δ ≈ 0` or
positive (loss climbing), e3 was the sweet spot and we should keep
e3 as the production adapter. The headline 12-condition PASS table
will use whichever epoch count gives the lowest masked val_loss
per (variant, rank) pair.

### Adapter naming convention

`/ckpts/lora_a2_<variant>_r<rank>_e<epochs>/`

Examples:
```
/ckpts/lora_a2_given_r16_e2/        ← variant=given, rank=16, 2 epochs
/ckpts/lora_a2_derived_r8_e2/
/ckpts/lora_a2_implicit_r4_e2/
```

### Per-adapter wall-time estimate

```
model load + 4-bit quantise + PEFT wrap     : ~10 min
SFT (~2,900 train rows × 2 epochs, eff      : ~40-60 min
  batch 8 → ~725 optimisation steps)
save adapter + commit volume                : ~1 min
─────────────────────────────────────────────────────────
TOTAL per adapter (A100-80GB)               : ~55-75 min
Cost per adapter                            : $3.46-4.30
```

Rank does not affect wall time — forward/backward dominated by the
4-bit base, not the LoRA. r=4 and r=16 take the same time within
noise.

### Run

Full sweep (9 trainings, sequential — for parallel use 9 terminals
since `modal run` blocks):

```bash
for v in given derived implicit; do
  for r in 4 8 16; do
    modal run src/a2_train_modal.py --variant $v --lora-r $r --epochs 2
  done
done
```

Smoke (~5 min, A100-80GB cost ~$0.30):

```bash
modal run src/a2_train_modal.py --variant given --limit 32 --epochs 1
```

`--limit N` truncates train_rows to N (val to N/8). Use for harness
verification before launching the full sweep.

### Outputs per training run

```
/ckpts/lora_a2_<v>_r<r>_e2/
   adapter_model.safetensors    ← LoRA weights (the only file eval needs)
   adapter_config.json
   summary.json                 ← {variant, epochs, lr, lora_r,
                                   lora_alpha, n_train, n_val,
                                   final_eval_loss}
   history.json                 ← per-step loss + per-epoch val loss
```

`final_eval_loss` from `summary.json` is the per-epoch held-out val
loss after the last epoch — useful for an early signal on which rank
is helping before running the full 12-condition eval.

---

## 23. Inference pipeline (`src/a2_eval_modal.py`)

Modal A100-40GB, 4-bit NF4 base, BF16 LoRA when loaded. One Modal
function call per condition — 12 calls total for the full sweep.

### Condition → variant → adapter mapping

```python
CONDITION_TO_VARIANT = {
    "zs-heading-given"         : "given",
    "zs-heading-derived"       : "derived",
    "zs-heading-implicit"      : "implicit",
    "trained-heading-given"    : "given",     # rank passed via --adapter
    "trained-heading-derived"  : "derived",
    "trained-heading-implicit" : "implicit",
}
```

Each condition reads its variant's test split:
`/sft/a2_{variant}_test.jsonl`.

Zero-shot conditions: no adapter loaded; base Qwen 2.5 VL 7B answers
the variant-specific student prompt cold.

Trained conditions: pass `--adapter /ckpts/lora_a2_<v>_r<r>_e2`
explicitly — the `/ckpts/` prefix is REQUIRED because that is where
the `navlm-ckpts` volume mounts inside the container; PEFT's
`from_pretrained` opens this path directly (default in
`DEFAULT_ADAPTER` is `r16`; override to evaluate r=4 / 8 adapters by
changing the rank suffix).

The script drops the assistant turn from each test row before
generating:

```python
messages_for_inference = [m for m in row["messages"] if m["role"] != "assistant"]
```

then decodes only the newly-generated tokens after the prompt.

### Inference hyperparameters

```
max_new_tokens          : 4096    (room for 4-step CoT — usually ~200)
temperature             : 0.0     (greedy, deterministic)
do_sample               : False
```

### Per-condition wall-time estimate

```
model load + 4-bit quantise                 : ~5 min
adapter load (trained conditions only)      : ~30 s
inference per sample:
   given / implicit (short outputs)         : ~8-12 s
   derived (4-step CoT, longer outputs)     : ~15-20 s
inference (~290 test samples):
   given / implicit                         : ~40-60 min
   derived                                  : ~75-95 min
─────────────────────────────────────────────────────────
TOTAL per condition (A100-40GB)             : ~50-100 min
Cost per condition                          : $1.75-3.50
```

Trained and zero-shot conditions take the same wall time — adapter
load is negligible vs. base load + inference.

### Run

Full sweep (12 conditions):

```bash
# 3 zero-shot:
for v in given derived implicit; do
  modal run src/a2_eval_modal.py --condition zs-heading-$v
done

# 9 trained (rank sweep):
for v in given derived implicit; do
  for r in 4 8 16; do
    modal run src/a2_eval_modal.py \
        --condition trained-heading-$v \
        --adapter /ckpts/lora_a2_${v}_r${r}_e2 \
        --run-id sweep_r${r}_$(date +%Y%m%d)
  done
done
```

Smoke (~3 min, A100-40GB cost ~$0.15):

```bash
modal run src/a2_eval_modal.py --condition zs-heading-given --limit 16
```

### Outputs per eval run

```
/eval/<run_id>/<condition>/
   per_sample.jsonl   ← one row per test sample with model_response,
                        gt_verb, heading, image_rel, ...
   summary.json       ← {condition, variant, is_trained, adapter,
                          n_samples, wall_time_s, out_path}
```

### Pull + score

```bash
mkdir -p eval_pull
modal volume get navlm-eval <run_id> eval_pull/

python -m src.a2_score --run-dir eval_pull/<run_id>
```

Writes `per_sample_scored.jsonl` + `summary.json` per condition, plus
prints the full 12-condition table (and writes
`summary_table.txt`). See §19 for metric definitions.

---

## 24. Full reproducibility checklist (annotate → train × 9 → eval × 12)

Assumes the data pipeline §1-§14 has already produced
`routes.jsonl` and the matched cohort (1,219 frames).

### Step A — Annotation (3 parallel Gemini passes, ~18 h, ~$90)

Prereq: 3 GCP projects with Vertex AI enabled + service-account JSON
keys (see `_setup_3_gcp_projects.sh`). Then:

```bash
bash _launch_3_annotation_passes.sh
# Tails: logs/annot_{given,derived,implicit}.log
# Outputs: data/cities/zurich/a2/annotations_a2_{variant}.jsonl
```

Each variant writes ~3,657 rows. The 3 GCP projects let you run in
parallel without hitting any one project's quota.

### Step B — SFT conversion (local, < 1 min, $0)

```bash
python -m src.a2_to_sft --variant given
python -m src.a2_to_sft --variant derived
python -m src.a2_to_sft --variant implicit
```

Produces `data/sft/a2_{variant}_{train,val,test}.jsonl` — 80/10/10
random split per variant, format_pass rows only.

### Step C — Upload to Modal (one-time, < 5 min, $0)

```bash
modal volume put navlm-data data/sft/a2_*.jsonl /sft/
```

Frame images already on `navlm-data:/data/frames/` from Attempt 1 —
no re-upload needed unless cohort changed.

### Step D — Training (9 LoRA adapters, ~1-1.5 h parallel, ~$35)

```bash
for v in given derived implicit; do
  for r in 4 8 16; do
    modal run src/a2_train_modal.py --variant $v --lora-r $r --epochs 2 &
  done
done
wait
```

(Modal runs each `modal run` as a separate function on its own A100;
the `&` + `wait` shell pattern launches 9 in parallel from one shell.
Modal will queue if all 9 GPUs aren't immediately available.)

Outputs `/ckpts/lora_a2_<v>_r<r>_e2/` × 9 on `navlm-ckpts`.

### Step E — Evaluation (12 conditions, ~1.5 h parallel, ~$26)

```bash
RUN_ID=$(date +%Y%m%d_%H%M%S)_a2_sweep

# 3 zero-shot:
for v in given derived implicit; do
  modal run src/a2_eval_modal.py --condition zs-heading-$v \
      --run-id $RUN_ID &
done

# 9 trained (rank sweep):
for v in given derived implicit; do
  for r in 4 8 16; do
    modal run src/a2_eval_modal.py \
        --condition trained-heading-$v \
        --adapter /ckpts/lora_a2_${v}_r${r}_e2 \
        --run-id ${RUN_ID}_r${r} &
  done
done
wait
```

Outputs `/eval/<run_id>/<condition>/per_sample.jsonl` per condition.

### Step F — Pull + score (local, < 5 min, $0)

```bash
mkdir -p eval_pull/$RUN_ID
modal volume get navlm-eval $RUN_ID eval_pull/
modal volume get navlm-eval ${RUN_ID}_r4 eval_pull/
modal volume get navlm-eval ${RUN_ID}_r8 eval_pull/
modal volume get navlm-eval ${RUN_ID}_r16 eval_pull/

# combine per-rank run dirs into one tree then score
python -m src.a2_score --run-dir eval_pull/$RUN_ID
python -m src.a2_score --run-dir eval_pull/${RUN_ID}_r4
python -m src.a2_score --run-dir eval_pull/${RUN_ID}_r8
python -m src.a2_score --run-dir eval_pull/${RUN_ID}_r16
```

Each `a2_score` invocation prints the conditions in that run dir.
For the final report, merge the four `summary.json` lists per
condition into one 12-row table — `summary_table.txt` in each run dir
is the printable view.

### Total wall-time + cost

```
Step A  annotation       18 h   $90
Step B  SFT conversion   <1 min  $0
Step C  upload           <5 min  $0
Step D  training         1-1.5 h $35
Step E  evaluation       1.5 h   $26
Step F  pull + score     <5 min  $0
───────────────────────────────────────
Total wall-time          ~22 h
Total cost               ~$151
```

(Annotation dominates wall time; training + eval together are ~3 h.)

---

## 25. Code walkthrough — sanity-check the key scripts

Five Python files implement everything in §17-§23. This section gives
file-and-line pointers to the critical implementation moments so you
can verify that the code actually does what the manual claims. Listed
in execution order.

### 25.1 `src/a2_annotate.py` — teacher annotation (3-pass Gemini Pro 2.5)

| Implements | Where | What it does |
|---|---|---|
| 3-variant system prompts (`given`, `derived`, `implicit`) | `THINKING_RULE` dict at line 85 | Per-variant tail appended to `SYSTEM_PROMPT_COMMON_HEAD` |
| **Teacher prompt = ALWAYS includes heading** | `build_teacher_prompt()` line 164 | `base = f"You are at this location, facing {heading:.0f}° ..."` is prepended for EVERY variant |
| **Student prompt = heading hidden for derived/implicit** | `build_student_prompt()` line 196 | For `variant == "given"` only, the heading line is prepended; otherwise omitted |
| Format-pass detection | `parse_answer()` line 225 | `t_open ≥ 0 AND a_open ≥ 0 AND first_verb is not None` |
| Direction-pass detection | computed downstream after parsing | `parsed["first_verb"] == gt_verb` |

**Sanity check — student should not see heading for derived/implicit**:
```bash
python -c "
import json
r = json.loads(open('data/cities/zurich/a2/annotations_a2_implicit.jsonl', encoding='utf-8').readline())
print('CAMERA heading in student_prompt?',
      'You are at this location, facing' in r['student_prompt'])
"   # MUST print False for derived + implicit
```

### 25.2 `src/a2_to_sft.py` — SFT split conversion

| Implements | Where | What it does |
|---|---|---|
| `--only-pass` filter | line 84 | `rows = [r for r in rows if r.get("direction_pass")]` |
| **Per-row content uses STUDENT prompt, NOT teacher** | line 99 | `{"type": "text", "text": r["student_prompt"]}` ← critical line |
| Variant-specific system prompt | line 88 | `sys_prompt_text = system_prompt(args.variant)` (imported from `a2_annotate.py`) |
| Random 80/10/10 split with `seed=42` | lines 119-132 | `rng = random.Random(args.seed); rng.shuffle(qwen_rows)` |

**Sanity check — SFT user-text matches student_prompt (not teacher)**:
```python
sft = json.loads(open('data/sft/a2_implicit_train.jsonl', encoding='utf-8').readline())
user_text = next(c['text'] for c in sft['messages'][1]['content'] if c.get('type') == 'text')
ann = ... # look up original annotation row by (video, frame_id, destination)
assert user_text == ann['student_prompt']         # MUST hold
assert user_text != ann['teacher_prompt']         # MUST hold (teacher has heading; student does not for derived/implicit)
```

### 25.3 `src/a2_train_modal.py` — LoRA SFT on Modal A100-80GB

| Implements | Where | What it does |
|---|---|---|
| Base model — Qwen 2.5 VL 7B in 4-bit NF4 | line 100 | `Qwen2_5_VLForConditionalGeneration.from_pretrained(BASE_MODEL, quantization_config=bnb, ...)` |
| LoRA target = q,k,v,o projection | line 44 | `LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj"]` |
| Fresh LoRA init OR resume | lines 104-129 | `if resume_adapter: PeftModel.from_pretrained(model, resume_adapter, is_trainable=True) else: get_peft_model(model, LoraConfig(...))` |
| **LOSS MASKED to assistant tokens only** | lines 113-150 | Per-row scan for `<\|im_start\|>assistant\n` tokens; `labels[:asst_start] = -100`. Pad + image-marker tokens also masked. |
| Hyperparams (lr=2e-4, cosine, warmup=3%, bf16, batch=1×grad8) | lines 158-172 | `TrainingArguments(...)` |
| Early stop + best-model load | lines 167-174 | `save_strategy="epoch"`, `load_best_model_at_end=True`, `metric_for_best_model="eval_loss"`, `EarlyStoppingCallback(patience=2)` |
| Adapter naming (with resume support) | line 156 | `out_dir = f"/ckpts/lora_a2_{variant}_r{lora_r}_e{total_epochs}"` where `total_epochs = resume_orig_epochs + epochs` |

**Sanity check — verify loss is masked to assistant tokens**:
```python
# In a fresh Python shell inside the Modal container (or run a dry collate test locally):
from transformers import AutoProcessor
proc = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct")
asst_prefix = proc.tokenizer.encode("<|im_start|>assistant\n", add_special_tokens=False)
# For a sample row, verify labels[:N] == -100 for all N before the assistant marker.
```

### 25.4 `src/a2_eval_modal.py` — inference on Modal A100-40GB

| Implements | Where | What it does |
|---|---|---|
| Variant → test split mapping | line 109 | `test_path = Path(f"/data/sft/a2_{variant}_test.jsonl")` |
| Adapter loading (trained conditions) | lines 130-133 | `if is_trained: model = PeftModel.from_pretrained(model, adapter)` |
| **Assistant turn STRIPPED before inference** | line 152 | `messages_for_inference = [m for m in row["messages"] if m["role"] != "assistant"]` |
| Greedy decoding | lines 160-166 | `model.generate(..., do_sample=False, temperature=0)` |
| **Decode ONLY newly-generated tokens** | line 168 | `gen_ids = gen[0][inputs["input_ids"].shape[1]:]` (excludes prompt) |
| Rank-suffixed output dir (multi-rank-safe) | lines 137-147 | Parses `_r<N>_e<M>` from adapter path → output `/eval/<run_id>/<condition>_r<r>_e<e>/per_sample.jsonl` |

**Sanity check — assistant turn is removed at inference**:
```python
import json
row = json.loads(open('data/sft/a2_given_test.jsonl', encoding='utf-8').readline())
roles = [m['role'] for m in row['messages']]
assert roles == ['system', 'user', 'assistant']            # original 3-turn
roles_inf = [m['role'] for m in row['messages'] if m['role'] != 'assistant']
assert roles_inf == ['system', 'user']                     # what model actually sees
```

### 25.5 `src/a2_score.py` — local scoring of eval outputs

| Implements | Where | What it does |
|---|---|---|
| Verb-extraction (longest-match first) | `parse_response()` lines 56-79 | `for v in sorted(VERBS, key=len, reverse=True)` — prefers "continue ahead" over "ahead" |
| Truncation-robust parsing | lines 64-74 | If `</answer>` missing, takes rest of text after `<answer>`. Pro 2.5 often omits the closing tag. |
| Format-pass = both tags + verb | line 85-86 | `t_open >= 0 AND a_open >= 0 AND first_verb is not None` |
| Direction-pass = match GT verb | line 120-121 | `parsed["first_verb"] == gt_verb` |
| **PASS = format AND direction** | line 122 | `PASS = parsed["format_pass"] and direction_pass` |
| Heading-inference accuracy (derived only) | lines 100-107, 131-138 | Regex `r"facing\s+(\d{1,3}(?:\.\d+)?)\s*°"` in `<thinking>`; circular diff < 22.5° |
| Per-condition table | `main()` lines 184-246 | Glob `*/per_sample.jsonl` under run_dir, score each, print 6-column table |

**Sanity check — scorer recognizes all 4 verbs from a synthetic response**:
```python
from src.a2_score import parse_response
r = parse_response("<thinking>foo</thinking><answer>Turn around.</answer>")
assert r["first_verb"] == "turn around" and r["format_pass"] is True
r = parse_response("garbage no tags")
assert r["format_pass"] is False
```

### 25.6 The 5-line sanity-check checklist (run before each big sweep)

```bash
# 1. Annotation files have all required fields:
python -c "import json; r=json.loads(open('data/cities/zurich/a2/annotations_a2_given.jsonl', encoding='utf-8').readline()); print(sorted(r.keys()))"
#    MUST include: student_prompt, teacher_prompt, response, format_pass, direction_pass, PASS, gt_verb, first_verb, heading

# 2. SFT splits exist and reference STUDENT prompt:
python -m src.a2_to_sft --variant given --only-pass    # (regen if you've re-annotated)
ls data/sft/a2_*.jsonl
# Then manually verify one row: heading hidden for derived/implicit, present for given

# 3. Modal volumes have what training needs:
modal volume ls navlm-data /sft     # 9 files for 3 variants × 3 splits
modal volume ls navlm-data /frames  # 8 video dirs

# 4. Training-script loss masking is in place:
grep -A2 "asst_prefix_ids" src/a2_train_modal.py
#    MUST find: "labels[i, :j + apl] = -100"   (assistant-prefix mask)

# 5. Eval-script strips assistant turn:
grep -n "messages_for_inference" src/a2_eval_modal.py
#    MUST find: `[m for m in row["messages"] if m["role"] != "assistant"]`
```

If all five pass, the pipeline matches the manual. If any fails,
inspect the relevant section above and the file at the listed line.

---

## 26. Glossary

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

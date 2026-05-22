

Q1: 3.1. we don't need these two right?
Purpose of zurich_landmarks_gps.py — 31 hand-curated landmark name→(lat, lon, aliases), manually verified from OSM. v1 used it for OCR sign-text → GPS. v2 (OCR removed): it is the high-precision table the VLM geo-localizer uses to resolve a named place → GPS, and a routing destination table.

Purpose of scenery_pois.py — 13 hand-written entries for features OSM tags as ways/polygons (streets, the Limmat, the lake, bridges) that point-node extraction misses. Each carries a custom radius_m (streets ~300 m, lake 600 m, bridges ~80 m) so proximity checks use a feature-appropriate radius instead of a fixed 50 m.

Q2: how the 26 POI is decided? is this L1 POI?
Gemma POI scan — reference/toolbox/scan_video_pois_multi.py:
Model: google/gemma-4-31b-it via local vLLM, one request per frame, every 20th frame.
The fixed 26-candidate POI list is hardcoded as CANDIDATE_POIS in scan_video_pois_multi.py:24-36 — a hand-picked short list of the most iconic Zurich landmarks + scenery (Hauptbahnhof, Lindenhof, Paradeplatz, Fraumünster, Grossmünster, …, Bahnhofstrasse, Limmat, Lake Zurich). It is a subset of the 453, kept short so Gemma picks from a manageable set.
Input: one frame image + a prompt embedding the 26 candidates ("identify ALL landmarks clearly visible … reply POI: <name>, or POI: none").
Output: parsed to _video_poi_multi.jsonl, {video, frame_id, visible_pois[]} — 1,358 rows, 25 distinct POIs.
[v2] Keep the existing Gemma POI scan output as-is — do NOT rerun with Gemini. The 3 tables and _video_poi_mult

Q3, this street view grid should be based on the POI bbox derived from the video frames right? 
VLM geo-localization, this is not to indicate the localization, but to indicate a place -- derive it as some GPS then compare it with DNOv2's Q4. decision 
I think use a weighted score between DINVOv2'S cosine simiarity score and the varaiance between dinov2's mapped gps and vlm's derived gps, check if there is any menthod to weight the score. I don't any of this is capbale enough to solely determine, what should gemini fast output in this case?

Q5. for Then OSM + HMM road-snapping (Newson-Krumm Viterbi over the osmnx walking graph) smooths the accepted GPS onto walkable geometry. this one would the route mapping consist of angle? or it only map the gps location, then how are we using the route to create instruction tuning, if a pic is facing front, the route is at the back, then the instruction tuning content will be different right, how to determine the pircture facing together with the route

Q6, 
VLM geo-check — candidate frames - pro
USE Annotation — 3 dest/frame pro

Q7
these should be seperate way of doing ablation, meaning the training will be either train on 7 and test on 1, or traiing one some POI and test on other POS
hold out saturday_morning video 
a set of destination POIs (§5)

Q8
explain the method for blur (Laplacian var) + exposure + pHash dedup

now you can remove the v1 content, and indicate the method used in detailed without mentioning same as v1 





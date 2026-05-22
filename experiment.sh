Q1: who determine this 
Scenery entries — 13 hand-added entries for features OSM tags as ways/polygons, which point extraction misses: streets (Bahnhofstrasse, Niederdorfstrasse…), the Limmat, Lake Zurich, bridges. Each carries a custom radius_m (streets ~300 m, lake 600 m, bridges ~80 m) so proximity checks use a feature-appropriate radius. These matter for navigation ("walk along Bahnhofstrasse") and are folded into the table.

Q2
POI scan of the video frames — scan_video_pois_multi.py had a VLM (Gemma) look at frames and list visible POIs from a fixed 26-candidate list. That list (CANDIDATE_POIS) is a hand-picked shortlist of the most iconic Zurich landmarks + scenery — effectively the tier-1 / "L1" iconic set (Hauptbahnhof, Lindenhof, Paradeplatz, Fraumünster, Grossmünster, Bahnhofstrasse, the Limmat, Lake Zurich, …). It is short on purpose, so the VLM picks from a manageable menu. The existing scan output (_video_poi_multi.jsonl) is kept as-is — not rerun. 
can you list down all 26 of the POIs and visualize them on map with a signature icon map? is there a code to obtain the 26 places?
when you list the 26 places in table please also list the chinese name 

Q3.
what is the code for this?
it is the bounding box for the 26 POI right? 
Crawl bbox (Q3). The grid should cover where the videos actually walk, not an arbitrary box. The route extent is bootstrapped from the GPS bounding box of the POIs the video POI-scan found (§2.3), plus a ~300 m margin so a POI on the edge — or a route segment that leaves the box — still has reference imagery. The metadata scan is free and incremental, so the box can be expanded later if GPS recovery shows routes near an edge.

Q4.
how is this top k calculated?
Heading (Q5). Each Street View crop was rendered at a known heading (0/90/180/270°). The frame's camera heading ≈ the heading of the matched crop; averaging the headings of the top-k matched crops (circular mean, outlier-filtered — the geometry is in reference/toolbox/compute_frame_heading.py) gives a heading estimate plus a spread-based confidence.

Q5. 
How to determine this to be 30 degree? any theory 
Every sample is gated by a closed-loop verifier: parse the action verb from the answer, check |heading + ACTION_DELTA[verb] − route_bearing| < 30°. Samples that fail are dropped.

Q6.
I think you didn't mentioned for this Instruction annotation — 3 dest/frame: about how the instruction is formed, like the propotion of the prompts: within 500m, 80%, within 1000m, 10%, with 1500m, 10%, this is to say with each pic, how this was done 

Q7.
for this one, I think you need to also mention how to split the frames, determine each frame is associate with which POI, may be need to visualize the polgon masks also by area
POI generalization — train on one set of destination POIs, test on a disjoint set. Tests routing to places never seen as a training destination.

Q8
for 5. Visualizations
add the POI region for the frames with GPS to show the areas 

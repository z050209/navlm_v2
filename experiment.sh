Slide 1 — Technical Approach Overview
NavLM v2: Self-Supervised View-Grounded Walking Directions

Goal: fine-tune a VLM to give short spoken walking directions from a phone photo + GPS, without relying on a compass.

Input at inference time

Current street-view / phone image
Coarse GPS location
Destination question, e.g. “How do I walk to Grossmünster?”

Core idea

Use maps for route planning
Use vision for camera-facing direction + visible landmark grounding
Train the VLM to translate abstract route geometry into first-person instructions

Pipeline

Walking-tour videos → frame extraction → GPS/heading recovery → OSM route planning → teacher instruction generation → verifier filtering → LoRA fine-tuning

Output

2–4 concise, TTS-ready walking instructions
Uses relative actions: “continue ahead,” “turn left,” “turn right,” “turn around”
Grounds directions in visible objects or landmarks

Example output

“Turn right toward the bridge ahead. Walk along the riverside path, keeping the church tower on your left.”

Slide 2 — Design Intuition
Why This Method Should Work

Challenge: route planners know where to go, but not what the user is currently facing.

A map route gives an absolute bearing, but a human needs a relative instruction:

relative action = route bearing − camera heading

So the central technical problem is recovering the camera heading from a normal image.

Design choice 1 — Separate geometry from language

OSM / route planner handles shortest-path walking geometry
VLM handles visual grounding and natural language generation
This reduces the burden on the VLM: it does not need to solve full graph routing

Design choice 2 — Recover heading using two complementary signals

DINOv2 retrieval: match video frames to Street View reference images
strong for visual similarity
may fail on repeated streets / visually similar scenes
VLM place naming: identify visible landmarks or street names
semantically aware
less precise in exact GPS

We combine them with an agreement-based score, then road-snap the accepted GPS sequence onto the OSM walking graph.

Design choice 3 — Use closed-loop filtering for label quality

Teacher VLM generates reasoning + answer
Verifier checks whether the answer’s action matches the planned route bearing
Incorrect or ambiguous samples are removed before LoRA training

Expected benefit
The model learns not just to describe a scene, but to infer “which way the camera is facing” and convert map routes into human-friendly instructions.

Slide 3 — Dataset, Baselines, and Evaluation
Dataset Construction

Raw data

8 public Zürich walking-tour videos
No GPS metadata
Zürich Old Town POIs and OSM walking graph

Frame preprocessing

Sample frames from videos
Remove blurry / overexposed / underexposed frames
Deduplicate near-identical frames using perceptual hashing

Self-supervised sample generation
For each trusted frame:

Recover GPS + heading
Sample destination POIs by distance band
Compute OSM walking route
Generate teacher answer
Verify direction correctness
Keep only clean instruction-tuning tuples
Baselines

B1 — Zero-shot Qwen2.5-VL

Image + destination prompt only
Tests whether the base model can infer direction without training

B2 — Zero-shot with explicit reasoning prompt

Ask the model to first infer visible landmarks / likely heading
Tests whether prompting alone is enough

B3 — With-heading oracle / ceiling

Provide camera heading directly
Measures how much of the problem is caused by heading recovery

Ours — Qwen2.5-VL + LoRA

Fine-tuned on self-supervised view-grounded walking instructions
Main setting: no compass, image + GPS + destination only
Evaluation Metrics

1. Direction correctness

Compare predicted action with OSM route bearing
Strict pass if angular error is within the action tolerance

2. Visible-anchor grounding

Check whether the named object / landmark is visible in the input image

3. Spoken-format usability

2–4 concise sentences
No compass words
No long metric-heavy route descriptions

4. Generalization tests

Held-out video: tests new camera trajectories
Held-out POIs: tests unseen destinations / map regions
Slide 4 — Current Implementation Progress and Plan
Current Progress

Implemented

Repo scaffold: config.py, src/, tests/, logs/, viz/
Video acquisition / local video organization
Frame extraction module
Quality filtering:
blur detection using Laplacian variance
exposure filtering using mean luminance
perceptual-hash deduplication
Candidate POI table with GPS and visualization map
Unit-testable pipeline modules

In progress

Street View reference crawl
DINOv2 image matching against Street View crops
VLM place-name verification
GPS + heading recovery
OSM / HMM road-snapping

Next implementation steps

Build Street View reference index
Match extracted frames to Street View images with DINOv2
Use VLM place naming to cross-check localization
Recover trusted GPS + heading frames
Generate OSM routes and teacher instructions
Run 5-sample annotation sanity check
Scale instruction generation
Fine-tune Qwen2.5-VL with LoRA
Evaluate zero-shot, oracle-heading, and LoRA models

Main risk

GPS / heading recovery may be noisy in visually repetitive streets

Mitigation

Use agreement between DINOv2 retrieval, VLM place naming, POI constraints, and HMM road-snapping before accepting training samples.

REQUIREMENTS:
Milestone 2 — Technical Approach (May 22)
The goal of this milestone is to solidify your method and plan for implementation. You should present your proposed approach, including the key design choices and the intuition behind them. It would also be helpful to explain why you expect your method to work, especially in comparison to your baselines.

Please also describe how you plan to use your dataset, what your experimental setup will look like, and how you'll evaluate performance.

Suggested slides: method overview (diagrams encouraged), design intuition, baselines, and experimental setup, along with your current implementation plan or progress.
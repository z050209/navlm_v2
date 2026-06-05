# Compass-Free Pedestrian Navigation with a Vision-Language Model:
# A LoRA Ablation Study on Qwen 2.5 VL 7B in Zurich

**CS231n Final Report — Spring 2026**
*Author: Yi (z050209). Affiliation: Stanford CS231n.*

---

## Abstract

We study whether a 7-billion-parameter vision-language model (VLM)
can replace an explicit numeric compass heading with **photo-derived
spatial reasoning** when issuing turn-by-turn pedestrian navigation
instructions in a real city. We construct 3,657 (frame, destination)
pairs across 21 famous Zurich attractions from an existing
DINOv2-localised street-view dataset, compute deterministic
ground-truth verbs via an OpenStreetMap walking graph, and use
Gemini Pro 2.5 in three parallel passes to annotate three input
*variants*: `heading-given` (the camera heading is given numerically),
`heading-derived` (the camera heading is hidden but the student model
is asked to derive it explicitly), and `heading-implicit` (no
numeric heading anywhere; purely visual reasoning). We then sweep
LoRA rank ∈ {4, 8, 16} × epoch ∈ {3, 5} for each variant on
Qwen 2.5 VL 7B. The headline result: PASS rate (well-formed
`<thinking>+<answer>` plus first-verb match to OSM ground truth)
improves from **44.7 % → 98.4 %** for heading-given, **26.8 % →
68.7 %** for heading-derived, and **28.1 % → 58.4 %** for
heading-implicit — gains of +30 to +54 percentage points over
zero-shot. The derived variant's heading-inference accuracy at 22.5°
tolerance rises from 27.6 % to 64.9 %, with a strikingly bimodal
prediction distribution (~65 % nearly-exact, ~30 % wildly off, almost
no fuzzy middle). Compass-free navigation works non-trivially, but
the ~30 pp gap to heading-given confirms that a numeric heading
remains a meaningful signal the model cannot fully substitute for
from photo cues alone at this scale.

---

## 1. Introduction (0.5-1 page)

Walking-tour apps and pedestrian wayfinding systems need to produce
per-step verb-level instructions ("continue ahead", "turn left",
"turn right", "turn around") from a single first-person photo of a
walker's current view. The hard subproblem is **disambiguating the
walker's facing direction**: traditional systems rely on an
IMU-derived compass heading, but consumer-phone compasses drift in
dense urban canyons, indoors, and near steel structures (Falaki et
al., 2010). A vision-language model that recovers heading from the
photo alone — recognising the cathedral on the east bank, the tram
tracks running south, the sun position at noon — would degrade more
gracefully and require no extra hardware.

We pose two empirical questions: **(Q1)** *How much navigation
accuracy do we lose if the model is never given the numeric
heading?* **(Q2)** *If the model is asked to first derive the heading
from the photo, does it learn to do so reliably?*

**Input/output formalisation.** The input to our algorithm is a tuple
$x = (I, p, h, d, B, V)$ where $I \in \mathbb{R}^{H \times W \times 3}$
is the first-person photo at a Zurich walking-tour frame,
$p \in \mathbb{R}^2$ is the walker's GPS position (HMM-map-matched
to the OSM walking graph), $h \in [0°, 360°)$ is the walker's camera
heading (shown or hidden depending on the variant), $d$ is the
destination's name (one of 21 famous attractions),
$B \in [0°, 360°)$ is the bearing of the first edge of the
shortest-path OSM route to $d$, and $V$ is the list of named
landmarks visible in $I$. The output is a verb
$\hat{y} \in \mathcal{V} = \{\text{continue ahead},
\text{turn left}, \text{turn right}, \text{turn around}\}$
preceded by a `<thinking>` chain of reasoning. We use **Qwen 2.5 VL
7B** (Bai et al., 2025) with LoRA adapters (Hu et al., 2021) trained
on 6,825 supervised-fine-tuning rows annotated by **Gemini Pro 2.5**
(Google DeepMind, 2025) as a teacher. We evaluate on a held-out
test split with the deterministic ground-truth verb
$y^\star$ computed from OSM geometry.

**Contributions.** (i) A reproducible OSM-routed dataset of 3,657
(frame, destination) pairs across 21 hand-curated famous Zurich
attractions with deterministic GT verbs (§3-4). (ii) A 3-variant
teacher-annotation protocol with input-asymmetric distillation
(teacher always sees heading, student variant-specific) realised as
3 parallel Gemini passes (§3.3-3.4). (iii) A 12-condition ablation on
Qwen 2.5 VL 7B + LoRA showing +30-54 pp PASS over zero-shot, with
rank-saturation at $r=4$ for the easy variant and continued
$r=16$ gains for harder ones (§5). (iv) A heading-inference analysis
revealing the trained model's heading-derivation distribution is
**bimodal**, with concrete implications for further improvement
(§5.3, §6).

---

## 2. Related Work (0.5-1 page)

**Vision-and-Language Navigation (VLN).** R2R (Anderson et al., 2018)
introduced photo-realistic indoor navigation in Matterport3D, where
an agent follows a natural-language instruction and emits per-step
actions. REVERIE (Qi et al., 2020) extends this to goal-finding
with referring expressions. Both assume access to the agent's
discrete pose (including heading) in a simulator. Our work removes
this assumption — the model must recover its own heading from a
single street-view photo.

**Vision-Language Models for spatial tasks.** CLIP (Radford et al.,
2021) established large-scale contrastive image-text pretraining;
LLaVA (Liu et al., 2023) and Qwen-VL (Bai et al., 2023; Bai et al.,
2025) added instruction-tuned multimodal chat. SpatialBot (Cai et
al., 2024) studied spatial-reasoning ability of VLMs but did not
target navigation. We fine-tune Qwen 2.5 VL 7B with parameter-
efficient adapters rather than full fine-tuning.

**Parameter-efficient fine-tuning.** LoRA (Hu et al., 2021)
decomposes weight updates into low-rank matrices, enabling
fine-tuning of multi-billion-parameter models on commodity GPUs.
QLoRA (Dettmers et al., 2023) couples LoRA with 4-bit NF4
quantisation of the frozen base — the recipe we use. Adapters
(Houlsby et al., 2019) and prefix-tuning (Li & Liang, 2021) are
related families.

**Teacher–student distillation with input asymmetry.** Standard
knowledge distillation (Hinton et al., 2015) trains a student to
mimic a teacher with matched inputs. Our setting is closer to
"privileged information" learning (Vapnik & Vashist, 2009): the
teacher sees the GT heading; the student does not. This guarantees
correct labels while forcing the student to learn the
photo-to-heading mapping.

**Visual localisation.** DINOv2 (Oquab et al., 2023) produces
self-supervised visual features that we use upstream to match
first-person frames against a StreetView panorama grid for GPS
recovery (the source of $p$ and $h$). OSMnx (Boeing, 2017) supplies
the walking-graph routing and nearest-node queries.

**Map-matching.** Hidden-Markov-model GPS-to-road snapping
(Newson & Krumm, 2009) yields cleaner per-frame positions than raw
DINOv2 GPS; we use this for both the matched cohort selection and
for the source node in our shortest-path routing.

**LR scheduling.** Cosine annealing with warm restart (SGDR;
Loshchilov & Hutter, 2016) is the basis for our resume-training
behaviour: when we extend a 3-epoch run by 2 more epochs, the cosine
schedule restarts from peak LR. AdamW (Loshchilov & Hutter, 2017)
is our optimiser.

---

## 3. Methods (2 pages)

### 3.1 Ground-truth verb from OSM geometry

For each (frame, destination) pair we compute the GT verb
deterministically with no model in the loop. Let $G = (N, E)$ be the
Zurich pedestrian walking graph from OpenStreetMap, edge-weighted by
length. Let $n_{\text{src}} = \arg\min_{n \in N} \|n - p\|_2$ be the
node nearest the walker's HMM-map-matched GPS (UTM-projected, EPSG:
32632), and let $n_{\text{dst}}$ be the destination's snapped node
(or, for multi-target destinations like Lake Zurich, the closest of
$K$ candidate nodes by Dijkstra distance). The shortest path
$\pi = (n_{\text{src}}, n_1, \ldots, n_{\text{dst}})$ is computed
with $\pi = \arg\min_{\pi} \sum_i \ell(n_i, n_{i+1})$, and the first-
edge bearing is

$$B = \mathrm{bearing}(n_{\text{src}} \to n_1) \in [0°, 360°)$$

Define the verb-rotation map $\Delta : \mathcal{V} \to \mathbb{R}$:

$$\Delta(\text{cont.ahead}) = 0°, \quad \Delta(\text{turn left}) = -90°,$$
$$\Delta(\text{turn right}) = +90°, \quad \Delta(\text{turn around}) = 180°$$

The GT verb is the verb whose post-action heading minimises the
angular distance to the route bearing:

$$y^\star = \arg\min_{v \in \mathcal{V}} |\angle((h + \Delta(v)) \bmod 360°, B)|$$

where $|\angle(\cdot, \cdot)|$ is the circular angular distance.
This yields 3,657 GT-verb-labelled samples across the 21-attraction
destination set, distance-banded 80 % near (50-500 m) / 10 % medium /
10 % far with `random.seed(42)`.

### 3.2 Three input variants

We define three variants $v \in \{\text{given, derived, implicit}\}$
that differ in **whether and how the camera heading $h$ appears in
the student's prompt**:

- **given**: $h$ is shown as `"You are at this location, facing 95°
  (east)"`. Student reasons numerically about $h$ vs $B$.
- **derived**: $h$ is hidden. Student is asked to first emit
  `"I estimate I'm facing X°"` from photo cues (4-step CoT: visual
  → geography → heading → verb), then reason about the verb.
- **implicit**: $h$ is hidden, AND the student is told not to emit
  any numeric heading (3-step CoT: what I see → destination position
  relative to me → verb).

### 3.3 Teacher annotation under input asymmetry

We use Gemini Pro 2.5 (Google DeepMind, 2025) as a teacher labeller.
Critically, **the teacher always sees $h$** (so it can compute
$y^\star$ correctly by geometry); the student's prompt is
variant-specific. The teacher's response is a `<thinking>...
</thinking><answer>...VERB.</answer>` block following the variant's
CoT template. Three independent passes (one per variant) ran in
parallel via three GCP service-account projects to bypass per-project
Vertex AI quota limits.

Total annotations: $10{,}614$ rows ($\approx 3{,}657$ per variant) at
$\sim\$95$ over 17 h wall-time. Measured teacher quality:
**format_pass = 96.6 %** (4-step derived CoT occasionally truncates
the closing tag), **direction_pass = 78.1 %**, **PASS = 77.8 %**.
The teacher exhibits a *turn-around bias*: of 1,997 direction
failures, 1,210 (60 %) are turn-left/right cases mis-labelled as
turn-around. We mitigate this with the `--only-pass` filter for
training (keep only rows where the teacher's verb matches $y^\star$).

### 3.4 LoRA supervised fine-tuning

The student is **Qwen 2.5 VL 7B-Instruct** (Bai et al., 2025) with
the base weights frozen in 4-bit NF4 (Dettmers et al., 2023) and
trainable LoRA adapters (Hu et al., 2021) on the query, key, value,
and output projections of every attention layer:

$$W_q' = W_q + \frac{\alpha}{r} B A, \quad A \in \mathbb{R}^{r \times d},\, B \in \mathbb{R}^{d \times r}$$

with $\alpha = 2r$ (constant per-weight effective LR across ranks),
dropout $0.05$, and $r \in \{4, 8, 16\}$ in our ablation. Each SFT
row is a 3-turn chat: $\text{system} = \text{system\_prompt}(v)$,
$\text{user} = (I, \text{student\_prompt}(v))$, $\text{assistant} =
$ teacher's `<thinking>+<answer>`.

**Loss function — masked cross-entropy.** Let $T$ be the input token
sequence after applying Qwen's chat template, and $a$ the index of
the first assistant-content token (located by searching for
`<|im_start|>assistant\n`). The training loss is the mean cross-
entropy over assistant tokens only:

$$\mathcal{L} = -\frac{1}{|T| - a} \sum_{t=a}^{|T|-1} \log P_\theta(T_t \mid T_{<t}, I)$$

with $\theta$ = LoRA parameters. Tokens before position $a$ (system +
user, including pad and image-marker tokens) are masked with
label $-100$. **This masking is critical**: without it, the loss is
dominated by the (identical-across-rows) prompt and the val-loss
curve looks artificially smooth; with the mask, val_loss reveals
real overfitting dynamics, as confirmed by a 32-sample overfit-test
control (train loss $\to 0.005$, masked val loss $\to$ U-shape with
minimum at epoch ~5).

**Optimiser.** AdamW (Loshchilov & Hutter, 2017) with $\beta_1 =
0.9$, $\beta_2 = 0.999$, $\epsilon = 10^{-8}$, weight decay $0$,
max gradient norm $1.0$. Learning rate $2 \times 10^{-4}$, cosine
schedule with 3 % warmup. Per-device batch size 1 with gradient
accumulation 8 (effective batch 8). BF16 mixed precision.
EarlyStoppingCallback with patience 2 epochs on masked val_loss,
`load_best_model_at_end=True`.

### 3.5 Inference and evaluation

At inference, the assistant turn is stripped from the test row and
the model generates it via greedy decoding (`do_sample=False`,
`max_new_tokens=4096`). We parse the response with a truncation-
robust regex (Pro 2.5's `</answer>` is often omitted) to extract
`first_verb` $\hat{y}$. Four metrics per row:

- **format_pass** $\in \{0,1\}$: response opens both `<thinking>`
  and `<answer>` and yields a parseable verb.
- **direction_pass** $\in \{0,1\}$: $\hat{y} = y^\star$.
- **PASS** = format_pass $\wedge$ direction_pass.
- **heading_inference_acc** (derived-variant only): regex-extract
  `"facing X°"` from `<thinking>` and check
  $|\angle(X, h)| < 22.5°$ (the angular threshold at which the
  argmin verb would flip).

### 3.6 Continue-training support

To test whether 3 epochs are enough, we added a `--resume-adapter`
flag that loads a saved adapter via `PeftModel.from_pretrained(
model, adapter, is_trainable=True)`, then trains 2 more epochs with
a fresh cosine LR (a warm restart in the SGDR sense; Loshchilov &
Hutter, 2016). Output adapter is named `..._e<orig+new>/`.

### 3.7 Codebase

We wrote all code in this project ourselves. We used **public
libraries** (HuggingFace `transformers`, `peft`, `bitsandbytes`;
PyTorch; OSMnx; matplotlib; Modal serverless GPU runtime), but no
existing project-level codebase was extended. The full source is at
https://github.com/z050209/navlm_v2 (~2,000 lines of Python).

---

## 4. Dataset and Features (0.5-1 page)

### 4.1 Source

We reuse a Zurich walking-tour video corpus from a prior project:
8 first-person videos covering central Zurich at ~1 fps, with raw
per-frame GPS recovered via DINOv2 (Oquab et al., 2023) panorama
matching against a Google StreetView grid. Each frame has a
candidate GPS position and a heading derived from the best-matching
StreetView panorama.

### 4.2 Curated destination vocabulary

A prior project used the top-30 OSM POI names from the cohort
($\sim 98\%$ street names) as the destination set — a poor match for
tourist queries. We replace it with **21 hand-curated famous
attractions** from three authoritative sources (Zürich Tourism,
PlanetWare, Switzerland Tourism), categorised as 4 churches
(Grossmünster, Fraumünster, St. Peter, Wasserkirche), 3 streets
(Bahnhofstrasse, Niederdorfstrasse, Limmatquai), 2 water features
(Lake Zurich, Limmat river), and 12 museums/civic/squares. Each
attraction has an alias table to fold VLM-produced name variants
(`"Zürich Hauptbahnhof"` → `Hauptbahnhof`).

### 4.3 Splits and preprocessing

We band-sample 3 destinations per matched frame, weighted 80 % near
(50-500 m straight-line) / 10 % medium / 10 % far, producing **3,657
(frame, destination) pairs**. After teacher annotation and the
`--only-pass` filter (keep rows where the teacher's verb matches the
OSM ground truth — drops the teacher's turn-around bias), we apply a
per-variant random 80/10/10 split with `random.seed(42)`:

| Variant | annotated | after --only-pass | train | val | test |
|---|---:|---:|---:|---:|---:|
| given | 3,657 | 3,201 | 2,561 | 320 | 320 |
| derived | 3,657 | 2,657 | 2,127 | 265 | 265 |
| implicit | 3,657 | 2,671 | 2,137 | 267 | 267 |
| **Total** | 10,971 | 8,529 | 6,825 | 852 | 852 |

### 4.4 Image preprocessing and input format

Images are JPGs at native resolution (median ~960 × 640 px), capped
by the Qwen 2.5 VL processor at $448 \times 448$ via the
`max_pixels=448*448` setting. No data augmentation (the task is
geometric — random crops would invalidate the heading). The model's
chat template inserts `<|vision_start|>`, $K$ image-pad tokens, and
`<|vision_end|>` markers between text turns; these are masked out
of the loss along with system + user tokens (§3.4).

Each SFT row contains: `image_rel` (relative frame path),
`messages` (3-turn chat as JSON), plus diagnostic carry-over fields
`video`, `frame_id`, `destination`, `gt_verb`, `heading`,
`direction_pass`. See the **worked examples** in `DEV_MANUAL_v2.md
§20** for one real training row per variant.

---

## 5. Experiments / Results / Discussion (2-3 pages)

### 5.1 Hyperparameter choices

- **LR = $2 \times 10^{-4}$, cosine, 3 % warmup**: a standard QLoRA
  default (Dettmers et al., 2023); we did not perform a separate LR
  sweep due to compute budget.
- **Effective batch size = 8** (per-device 1 × grad-accum 8):
  selected to maximise effective batch while fitting Qwen 7B + LoRA
  + image tensors in A100-80GB memory.
- **LoRA rank $r \in \{4, 8, 16\}$**: the headline ablation,
  testing whether capacity is the bottleneck.
- **Epochs $\in \{3, 5\}$**: 3 chosen as a conservative starting
  point; 5 added via `--resume-adapter` after observing monotonic
  val_loss decrease through epoch 3.
- **EarlyStoppingCallback(patience=2)** on masked val_loss, with
  `load_best_model_at_end`: never fired at the 3-epoch cap on the
  real-scale data (only on the 32-sample overfit control).
- **No cross-validation**: a single 80/10/10 split with seed 42 per
  variant. Confidence intervals at $n = 265-320$ are $\pm 3-6$ pp.

### 5.2 Primary metric

Our headline metric is **PASS rate** = mean over the test set of
$\mathbb{1}[\text{format\_pass}] \cdot
\mathbb{1}[\hat{y} = y^\star]$.

We also report direction_pass (verb accuracy ignoring format) and,
for derived only, heading-inference accuracy at 22.5° tolerance (the
angular threshold at which $\arg\min$-verb flips).

### 5.3 Quantitative results — the 12-condition matrix

The full 12-condition ablation result (best of e3/e5 per
condition):

| Variant | zero-shot | best LoRA | $\Delta$ | best adapter |
|---|---:|---:|---:|---|
| heading-given | 44.7 % | **98.4 %** | +53.7 pp | r=8 e5 or r=16 e5 |
| heading-derived | 26.8 % | **68.7 %** | +41.9 pp | r=16 e5 |
| heading-implicit | 28.1 % | **58.4 %** | +30.3 pp | r=16 e5 |

![Figure 2: Zero-shot vs best LoRA](figures/fig2_zs_vs_trained.png)
*Figure 2: PASS rate (%) per variant, zero-shot Qwen 2.5 VL 7B vs.
best LoRA-trained. All three variants gain 30-54 pp from LoRA
fine-tuning; the largest gain is on heading-given (basically solved
at 98.4 %), the smallest on heading-implicit (purely visual, 58.4 %).*

**Rank-saturation.** Figure 1 shows PASS vs. rank for each
variant × epoch count. Given saturates at $r = 4$ (97.2 → 97.8 →
98.1 with $\Delta < 1$ pp per rank doubling); derived's behaviour
inverts between e3 (peak at $r = 8$, 64.9 %) and e5 (peak at
$r = 16$, 68.7 %); implicit climbs monotonically through $r = 16$.

![Figure 1: Rank-saturation curve](figures/fig1_rank_saturation.png)
*Figure 1: PASS rate vs LoRA rank for each variant. Dashed lines =
e3 adapters, solid = e5 adapters. Horizontal dotted lines mark the
zero-shot baselines. Given is saturated at $r=4$; implicit gains
monotonically; derived's optimum rank depends on epoch count.*

**Continue-training (e3 → e5).** Across 8 of 9 trained adapters,
+2 epochs improved PASS by 0.0 to +5.7 pp; only derived-r8 regressed
(-1.1 pp). The biggest e5 gains were on implicit-r4 (+5.6 pp) and
derived-r16 (+5.7 pp). Notably, **val_loss is an imperfect proxy for
PASS**: implicit-r16 had zero val_loss change from e3 to e5 but PASS
jumped +3.3 pp. Token-level cross-entropy can plateau while the
model continues refining its verb-choice behaviour. Future training
recipes should periodically sanity-check PASS during training rather
than stopping purely on val_loss.

### 5.4 Heading-inference quality (derived-variant only)

For derived, we parse `"facing X°"` from `<thinking>` and compare to
the GT heading $h$ via circular distance.

| Condition | n_emit | within 5° | within 22.5° | median \|err\| | mean \|err\| |
|---|---:|---:|---:|---:|---:|
| zs-derived | 134/265 (51 %) | 27.6 % | 27.6 % | 90.0° | 98.9° |
| trained-derived-r4 e3 | 265/265 (100 %) | 58.1 % | 58.1 % | 0.0° | 52.2° |
| trained-derived-r16 e5 | 265/265 (100 %) | **64.5 %** | **64.9 %** | **0.0°** | **39.3°** |

![Figure 3: Heading-inference scatter](figures/fig3_heading_scatter.png)
*Figure 3: Predicted vs ground-truth heading for the derived
condition, comparing zero-shot Qwen (left) to the best LoRA model
(right, r=16 e5). The trained model produces a strongly bimodal
distribution: 64.9 % of predictions lie within $\pm 22.5°$ of the
identity line (green), only $\sim 2.3$ % land in the
$22.5° < \cdot < 90°$ "close-ish" zone (orange), and $\sim 32.8$ %
are >90° off (red). Median absolute error is 0° — over half the
predictions are essentially exact.*

**Three findings**:

(1) **The trained model is bimodal**: within-5° $\approx$ within-22.5°
for every trained condition; median |err| = 0°. The model is either
nearly-exact (lookup-table-like) or wildly off, with essentially no
fuzzy interpolation in between.

(2) **LoRA forces commitment**: zero-shot Qwen emits "facing X°" in
only 51 % of derived rows; after LoRA, 100 %. A substantial fraction
of derived's PASS gain comes from this "forced commitment", not
purely from better orientation reasoning.

(3) **e5 reliably shortens the wrong-direction tail**: r=16 e3 → e5
reduces mean |err| by 8° (47° → 39°) but `within-22.5°` only +4.5
pp. The threshold metric misses tail improvements that the mean
absorbs.

### 5.5 Qualitative examples (success and failure)

**Success — trained-given-r16 e5, a `turn around` case**:
```
<thinking>I'm facing 95° (east); the route heads 270° (west), which
is a 175° rotation, so the destination is behind me. I should turn
around.</thinking>
<answer>Notice the Limmatquai with tram tracks behind you. Turn
around.</answer>
```

**Failure — zs-implicit defaulting to "continue ahead"**:
```
<thinking>STEP 1: The visible scene shows a cobblestone street with
shops. STEP 2: The destination is ahead of me, as the street extends
forward. STEP 3: Therefore the walker should continue ahead.
</thinking>
<answer>Continue ahead.</answer>
```
The model has no way to verify the destination's actual direction so
it defaults to "ahead" — the canonical implicit failure mode (6 of
14 implicit failures in our smoke test followed this pattern).

**Failure — trained-given-r16 e5 left/right confusion (rare)**:
```
<thinking>You're currently facing west (270°). The route heads
south (185°), which means you need to turn right to follow the
correct direction.</thinking>
<answer>Look at the building on your right. Turn right.</answer>
```
GT verb is `turn left` (west → south is a -85° rotation = left). The
trained model occasionally inverts left/right despite having the
heading. These represent ~1.5 % of trained-given failures and are
not eliminated by more LoRA capacity.

Full per-condition examples (2 success + 2 failure × 6 conditions =
24 cases) are in `docs/qualitative_examples.md`.

### 5.6 Overfitting analysis

The masked val_loss decreased monotonically through epoch 3 for all
9 adapters — early-stop did not fire — suggesting we are NOT
overfitting at the 3-epoch cap on real-scale data. The 32-sample
overfit-test control (30 epochs × 32 train samples) showed a
clear U-shaped val_loss with minimum at epoch ~5, confirming the
masking and early-stop pipeline both work as intended. Train-vs-val
gap at e3 is small (train $\sim 85$ % of val), and e5 PASS gains on
8 of 9 conditions confirm there is still useful generalisation
signal in the gradient — i.e. the model is in the "still learning"
regime, not "memorising train". Mitigations in place: LoRA $r \in
\{4, 8, 16\}$ as a capacity bottleneck (vs full fine-tuning), LoRA
dropout 0.05, early-stop patience 2.

### 5.7 Discussion

**Q1: Compass-free navigation works non-trivially.** Both
heading-derived (68.7 %) and heading-implicit (58.4 %) far exceed
their zero-shot baselines (26.8 / 28.1 %). The model genuinely
learns to extract orientation cues from photos — derived's heading-
inference accuracy rises from 27.6 % to 64.9 % at 22.5° tolerance.

**Q2: But numeric heading is still a meaningful signal.** The gap to
heading-given (98.4 %) is 30 pp for derived and 40 pp for implicit.
The bimodal heading-prediction distribution implies the model is
*pattern-matching against memorised landmarks* rather than
interpolating an orientation function: when the photo contains a
recognised landmark (Grossmünster's east-bank twin towers, the
Limmat axis), the heading is exact; otherwise it's wildly off.
**To narrow the gap**, future work should expand the recognised-
landmark set rather than refine a non-existent interpolation
function.

---

## 6. Conclusion / Future Work (1-3 paragraphs)

We showed that LoRA fine-tuning of Qwen 2.5 VL 7B on 6,825
OSM-grounded supervised samples brings pedestrian-navigation PASS
rates from $\sim 45$ % to $98.4$ % when the camera heading is
provided numerically. Removing the numeric heading and asking the
model to derive it from photo cues yields **68.7 %** — a $+41.9$ pp
gain over zero-shot but $30$ pp below heading-given. Removing all
heading scaffolding (purely visual) reaches **58.4 %**. The
heading-derivation distribution is strikingly bimodal, suggesting
the model has learnt a discrete landmark $\to$ heading lookup
rather than a continuous orientation function. LoRA rank 4 suffices
for the easy variant; harder variants benefit from $r = 16$ and 5
epochs.

**For future work** with more compute, we would (i) test
generalization out-of-distribution by holding out a video, a
destination attraction, and a different city; (ii) instrument
training with periodic PASS-during-training sanity checks given our
val_loss/PASS dissociation; (iii) extend from single-step verb
emission to multi-step rollouts of 5-20 instructions per walk
(compounding-error analysis); and (iv) explore a hybrid system in
which a noisy compass is supplied alongside the photo and the model
learns *when to trust which signal*. The bimodal heading-prediction
finding also motivates targeted training data that explicitly
covers under-represented landmark-orientation pairs.

---

## 7. Appendices

(Optional and not counted toward the 8-page main body limit.)

- **Appendix A**: 21-attraction list with Chinese names and tourism-
  source citations (see `DEV_MANUAL_v2.md §2`).
- **Appendix B**: OSM walking-graph UTM-projection (EPSG:32632)
  details and multi-target routing for long features
  (`DEV_MANUAL_v2.md §10`).
- **Appendix C**: Whole-dataset teacher-quality analysis —
  per-destination and per-distance-band breakdowns with
  the four direction-failure-mode analogies (the "180° trigger-
  finger", the "right-handed gardener", the "which-bank-am-I-on
  confusion", the "around-the-corner blindspot").
  `DEV_MANUAL_v2.md §18`.
- **Appendix D**: Loss-masking implementation and sanity-check
  snippets (`DEV_MANUAL_v2.md §22, §25`).
- **Appendix E**: Modal infrastructure (3 volumes, 2 apps), Windows
  PowerShell gotchas, full reproducibility checklist
  (`DEV_MANUAL_v2.md §21, §24`).

---

## 8. Contributions & Acknowledgements (not in page limit)

### Contributions

This is a single-author project. **Yi (z050209)** carried out all of:
problem formulation, dataset construction (21-attraction curation,
OSM routing pipeline, multi-target node selection, GT-verb
algorithm), teacher annotation (3-pass Gemini Pro 2.5 setup with 3
GCP service-account projects), Qwen 2.5 VL 7B LoRA fine-tuning
(Modal A100s, rank sweep, e5 resume training), evaluation pipeline,
scoring metrics, figures, qualitative-example extraction, and
report writing.

### Generative AI usage statement

Generative AI (Claude Code, an interactive software-engineering
assistant from Anthropic, model `claude-opus-4-7`) was used
extensively for **software-engineering assistance**: writing the
training, evaluation, and scoring scripts; debugging Modal volume
mount and PowerShell path-conversion issues; structuring the
development manual; generating matplotlib figures from the eval
outputs; and proofreading this report. All scientific decisions
(experimental design, ablation matrix, the 3-variant teacher-student
input-asymmetry design, the rank/epoch sweep, the loss-masking
choice, the conclusion that the heading-prediction is bimodal) and
all interpretation of results were directed by the human author.
Gemini Pro 2.5 (Google DeepMind) was used as the teacher model for
SFT data annotation — this is a method-level use, fully documented
in §3 and not a writing aid. All code, prompts, and metrics in this
project are the author's own work; no external project codebase was
extended.

### Acknowledgements

Stanford CS231n teaching staff for course materials and the project
template. The Hugging Face PEFT and Transformers teams for the LoRA
implementation. Alibaba Cloud for the open-weights release of
Qwen 2.5 VL 7B. Modal Labs for serverless GPU compute. Google
DeepMind for the Vertex AI Gemini Pro 2.5 API access. OpenStreetMap
contributors for the Zurich walking graph used in routing.

### Code availability

All code, prompts, and per-condition evaluation outputs are at
**https://github.com/z050209/navlm_v2**. The development manual
(`DEV_MANUAL_v2.md`) contains a full §24 reproducibility checklist:
a fresh clone + the documented commands reproduces the entire result
table in $\sim 22$ h wall-time at $\sim \$165$ ($\$95$ Gemini +
$\$70$ Modal). This report was prepared exclusively for CS231n and
is not shared with another class.

---

## 9. References / Bibliography (no page limit)

Anderson, P., Wu, Q., Teney, D., Bruce, J., Johnson, M., Sünderhauf, N., Reid, I., Gould, S., and van den Hengel, A. (2018). Vision-and-Language Navigation: Interpreting Visually-Grounded Navigation Instructions in Real Environments. *CVPR*.

Bai, J., Bai, S., Yang, S., Wang, S., Tan, S., Wang, P., Lin, J., Zhou, C., and Zhou, J. (2023). Qwen-VL: A Versatile Vision-Language Model for Understanding, Localization, Text Reading, and Beyond. *arXiv:2308.12966*.

Bai, S., Chen, K., Liu, X., Wang, J., Ge, W., Song, S., Dang, K., Wang, P., Wang, S., Tang, J., et al. (2025). Qwen2.5-VL Technical Report. *arXiv:2502.13923*.

Boeing, G. (2017). OSMnx: New Methods for Acquiring, Constructing, Analyzing, and Visualizing Complex Street Networks. *Computers, Environment and Urban Systems*, 65, 126-139.

Cai, W., Ponomarenko, I., Yuan, J., Li, X., Yang, W., Dong, H., and Zhao, B. (2024). SpatialBot: Precise Spatial Understanding with Vision Language Models. *arXiv:2406.13642*.

Dettmers, T., Pagnoni, A., Holtzman, A., and Zettlemoyer, L. (2023). QLoRA: Efficient Finetuning of Quantized LLMs. *NeurIPS*.

Falaki, H., Mahajan, R., Kandula, S., Lymberopoulos, D., Govindan, R., and Estrin, D. (2010). Diversity in smartphone usage. *MobiSys*.

Google DeepMind (2025). Gemini 2.5 Pro Technical Report. *Google DeepMind Technical Report*.

Hinton, G., Vinyals, O., and Dean, J. (2015). Distilling the Knowledge in a Neural Network. *NeurIPS Deep Learning Workshop*.

Houlsby, N., Giurgiu, A., Jastrzebski, S., Morrone, B., De Laroussilhe, Q., Gesmundo, A., Attariyan, M., and Gelly, S. (2019). Parameter-Efficient Transfer Learning for NLP. *ICML*.

Hu, E. J., Shen, Y., Wallis, P., Allen-Zhu, Z., Li, Y., Wang, S., Wang, L., and Chen, W. (2021). LoRA: Low-Rank Adaptation of Large Language Models. *ICLR 2022*.

Kingma, D. P. and Ba, J. (2014). Adam: A Method for Stochastic Optimization. *ICLR 2015*.

Li, X. L. and Liang, P. (2021). Prefix-Tuning: Optimizing Continuous Prompts for Generation. *ACL*.

Liu, H., Li, C., Wu, Q., and Lee, Y. J. (2023). Visual Instruction Tuning. *NeurIPS*.

Loshchilov, I. and Hutter, F. (2016). SGDR: Stochastic Gradient Descent with Warm Restarts. *ICLR 2017*.

Loshchilov, I. and Hutter, F. (2017). Decoupled Weight Decay Regularization. *ICLR 2019*.

Newson, P. and Krumm, J. (2009). Hidden Markov Map Matching Through Noise and Sparseness. *Proc. ACM SIGSPATIAL GIS*.

Oquab, M., Darcet, T., Moutakanni, T., Vo, H., Szafraniec, M., Khalidov, V., Fernandez, P., Haziza, D., Massa, F., El-Nouby, A., et al. (2023). DINOv2: Learning Robust Visual Features without Supervision. *Transactions on Machine Learning Research*.

Qi, Y., Wu, Q., Anderson, P., Wang, X., Wang, W. Y., Shen, C., and van den Hengel, A. (2020). REVERIE: Remote Embodied Visual Referring Expression in Real Indoor Environments. *CVPR*.

Radford, A., Kim, J. W., Hallacy, C., Ramesh, A., Goh, G., Agarwal, S., Sastry, G., Askell, A., Mishkin, P., Clark, J., et al. (2021). Learning Transferable Visual Models From Natural Language Supervision. *ICML*.

Vapnik, V. and Vashist, A. (2009). A new learning paradigm: Learning using privileged information. *Neural Networks*, 22(5-6), 544-557.

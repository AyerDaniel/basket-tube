# BasketTube — Approaches & Findings

## Core Thesis
The ball encodes all meaningful game information in its motion. The pipeline tracks the ball, identifies motion patterns, and classifies them as semantic events — unsupervised, with human labels applied after the fact.

---

## Pipeline Architecture (Stable)

### Sweep 1 — Ball Trace Extraction
- Split video into frames at delta-t intervals (30fps, 1918x1078)
- Detect ball per frame using RF-DETR (batch=16, RTX 3080)
- Output: `df_balls.csv` — bounding boxes + confidence per frame
- Annotated frames saved to `game_1/ball_frames/`

### Sweep 2 — Semantic Event Segmentation
- Compute motion vectors: `dx, dy, d_area` from consecutive ball centroids
- DISJOINT tokens mark gaps too large to bridge
- Goal: identify event boundaries and classify event types

### Sweep 3 — Player Attribution (planned)
- YOLOv8-pose at event boundaries
- Spatial KNN: nearest player = actor/recipient
- Deferred pending better event segmentation

---

## Sweep 2 Approaches — Chronological

### Approach 1 — Log Magnitude Bucketing
**Idea:** Bucket frames by `log10(sqrt(dx²+dy²))` into 3 groups based on observed histogram populations:
- Bucket 0: near-zero / stationary (`log10 < -2`)
- Bucket 1: normal motion (`-2 ≤ log10 ≤ 1`)
- Bucket 2: high-motion transitions (`log10 > 1`)

**Findings:** Bucket 0 = noise. Bucket 1 = mixed events (pass, shot, dribble at similar magnitudes). Bucket 2 = noisy Bucket 1. Magnitude alone is insufficient — it captures how much motion but not what kind.

---

### Approach 2 — Nearest Player as Camera Bias Corrector
**Idea:** Human bias contaminates the video signal. Camera follows human attention, not ball physics. Adding nearest-player distance as a feature might compensate for camera bias.

**Implementation:** Run RF-DETR person detection (class_id=1) on `ball_frames/`, compute distance from ball centroid to each player centroid per frame. Save to `df_players.csv`, annotated frames to `ball_player_frames/`.

**Additional hypothesis:** Nearest-player stability over a window is a strong event discriminator:
- High stability (same player throughout) → dribble
- One transition → pass
- Ball briefly nearest nobody → shot arc

**Why abandoned:** Requires persistent player tracking (ByteTrack/DeepSORT) for stability feature. Player identity flickers frame-to-frame with per-frame detection only. Also, adding player signal introduces human-bias assumptions into a ball-motion-only pipeline.

---

### Approach 3 — Feature Engineering (angle, verticality)
**Idea:** Add `angle = atan2(dy, dx)` and `verticality = |dy|/magnitude` as discriminating features. Dribbles are vertically dominated (angle ≈ ±90°), passes are horizontal.

**GMM threshold learning:** Fit 1D Gaussian Mixture Model on verticality score to learn dribble/not-dribble threshold from data.

**Why abandoned:** These are algebraic rearrangements of `dx` and `dy` — no new information. A model operating on raw vectors should discover these relationships itself. Feature engineering risks introducing bias without adding signal.

---

### Approach 4 — Polar Coordinate Embedding `[angle, magnitude]`
**Idea:** Embed each frame as `(θ, r)` in polar space. Sequence of `(θ, r)` should trace characteristic orbits per event type.

**Why abandoned:** Same issue as Approach 3 — angle is redundant with raw vectors. Also per-frame embeddings don't capture temporal structure. Sequences needed, not individual frames.

---

### Approach 5 — Ball-Area Normalized Vectors
**Insight:** Raw `dx, dy` are camera-contaminated — same event looks different at different zoom levels. Normalize by apparent ball size as proxy for zoom level:
```
norm_dx = dx / sqrt(area)
norm_dy = dy / sqrt(area)
```
**Status:** Implemented as `compute_normed_vectors()`. Imperfect but better than raw pixel displacement without full homography.

---

### Approach 6 — DTW + UMAP on Raw Sequences
**Idea:** 
- Extract overlapping windows of `[norm_dx, norm_dy]` at 2s, 3s, 4s, 5s
- Compute pairwise DTW distance matrix (handles speed variation between same event types)
- UMAP on distance matrix → 2D scatter plot revealing natural clusters

**Implementation:** `run_dtw_umap()` with subsample=2000, n_neighbors=30, stride=1.

**Findings:** 
- Window=90 (3s) showed most compact structure
- Coloring by magnitude bucket showed complete intermixing — magnitude bucket does not correspond to motion pattern clusters
- This confirms Approach 1 finding: magnitude is not a proxy for event type
- The embedding may have real structure but we lack ground-truth labels to reveal it

---

### Approach 7 — TCC-style Self-Supervised Learning (considered, rejected)
**Idea:** Use Temporal Cycle-Consistency Learning to learn embeddings without labels.

**Problem:** TCC requires two independent views of the same event (multi-camera). Single broadcast footage provides only one view.

**Alternatives considered:**
- Mirrored sequences as second view → rejected: model would learn the mirror transformation, not event semantics
- Overlapping windows as positive pairs → rejected: just temporally shifted same data, trivially similar
- Augmentation (noise, time-warp, dropout) → viable but weak signal

**Conclusion:** Genuinely self-supervised sequence learning without a second view or labels is not feasible at this problem.

---

## Core Blocker — Camera Motion Entanglement

The fundamental problem: `norm_dx, norm_dy` conflate ball motion with camera motion (pan, zoom, cut). A shot from a stationary camera looks completely different from a shot during a pan or zoom. Any clustering approach will find clusters like "shot + stationary camera" and "shot + panning camera" as separate groups rather than "shot" as a unified event type.

**Proposed solution: Court Registration (Homography)**
- Use court markings (3-point line, paint, center circle) to map ball/player positions to canonical court coordinates via homography
- Each frame independently anchored — handles camera cuts, pans, zooms
- Transforms vectors into true world-space ball displacement
- Makes the same event look the same regardless of camera state
- **Status: deferred, but likely prerequisite for all downstream segmentation work**

---

## Current State
- Ball detection and tracing working (`df_balls.csv`, `ball_frames/`, `create_tracing_video()`)
- Motion vectors computed (`df_vectors` with `norm_dx`, `norm_dy`)
- Magnitude bucketing implemented but insufficient
- DTW + UMAP implemented, structure unclear without better labels or camera correction
- Next logical step: court registration to remove camera motion from vectors

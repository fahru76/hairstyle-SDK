# AGENTS.md — notes for AI coding agents

This file exists so any AI agent (Claude, Codex, etc.) picking up this repo can get oriented fast without re-deriving context from scratch.

## Project intent

AR hairstyle try-on for an online barbershop app. Two distinct capabilities, don't conflate them:

1. **Hair color change** — fully working. Uses MediaPipe's `hair_segmenter` model to get a confidence mask of hair pixels, then shifts hue/saturation toward a target color in HSV space while preserving the original Value channel (so shading/highlights/texture stay intact).
2. **Hairstyle (cut/shape) change** — skeleton only, not production-ready. MediaPipe cannot generate new hair geometry; it only segments existing hair. The approach implemented here is a classic AR pattern: anchor a 2D PNG asset (a pre-rendered hairstyle, transparent background) to face landmarks via homography, then warp+blend it onto the live frame each frame.

## Current state (see PROGRESS.md for the live log)

- `src/hair_color/` — complete and testable once models are downloaded.
- `src/wig_overlay/` — three-script skeleton (landmark picker → asset annotator → runtime overlay). NOT yet combined with hair segmentation, so real hair is not removed/hidden before the wig is placed — this will look wrong if the user's real hair pokes out from under the overlay. That's the next planned step, see PROGRESS.md.

## Design decisions worth knowing before you change things

- **API surface**: uses the current MediaPipe **Tasks** API (`mediapipe.tasks.python.vision`), not the deprecated standalone `hair_segmentation.tflite` solution. If you see old-style `mp.solutions.*` code for hair segmentation anywhere, it's stale — don't copy it in.
- **Landmark IDs for hairline/temple anchors** (`CANDIDATE_IDS` in `pick_landmarks.py`) are **not verified ground truth** — they're community-sourced starting points. MediaPipe's face mesh is sparsest exactly in the hairline region. Always re-verify visually against `landmarks_check.jpg` before trusting them, especially if you change which anchor points are used.
- **Homography, not full mesh warp**: `wig_overlay.py` uses `cv2.findHomography` (planar, rigid-ish). This is a known simplification — good enough for near-frontal poses, breaks down on extreme head tilt. If accuracy needs to improve, look at Delaunay triangulation + piecewise affine warp (see README for a reference link) before reaching for something heavier.
- **Recoloring math**: HSV Hue+Saturation blend toward target, Value channel untouched — this is deliberate, it's what keeps recolored hair looking like real hair instead of a flat color fill.

## What NOT to do

- Don't add a paid/commercial SDK dependency (Banuba, GlamAR, DeepAR, etc.) without discussion — the whole point of this repo is the free/open-source path.
- Don't hardcode landmark IDs into `wig_overlay.py` without going through `pick_landmarks.py` verification first for any new anchor scheme.
- Don't remove the confidence-mask feathering (`cv2.GaussianBlur` on the mask) — it's there to prevent jaggy edges at the hair/background boundary.

## Next steps (also tracked in PROGRESS.md)

1. Combine `hair_segmenter` mask into `wig_overlay.py` so real hair is masked out before the wig PNG is blended on top.
2. Build out a small asset library in `data/wigs/` (a handful of styles + their CSVs) to prove the pipeline end-to-end.
3. Evaluate VIDEO running mode (vs per-frame IMAGE mode) for better real-time performance.
4. Consider Delaunay-triangulation warp as a follow-up if homography-only quality isn't good enough.

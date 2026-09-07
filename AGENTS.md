# AGENTS.md — notes for AI coding agents

This file exists so any AI agent (Claude, Codex, etc.) picking up this repo can get oriented fast without re-deriving context from scratch.

## Project intent

AR hairstyle try-on for an online barbershop app. Two distinct capabilities, don't conflate them:

1. **Hair color change** — fully working. Uses MediaPipe's `hair_segmenter` model to get a confidence mask of hair pixels, then shifts hue/saturation toward a target color in HSV space while preserving the original Value channel (so shading/highlights/texture stay intact).
2. **Hairstyle (cut/shape) change** — skeleton only, not production-ready. MediaPipe cannot generate new hair geometry; it only segments existing hair. The approach implemented here is a classic AR pattern: anchor a 2D PNG asset (a pre-rendered hairstyle, transparent background) to face landmarks via homography, then warp+blend it onto the live frame each frame.

## Current state (see PROGRESS.md for the live log)

- `src/hair_color/` — complete and testable once models are downloaded.
- `src/wig_overlay/` — landmark picker → asset annotator → runtime overlay, now combined with `hair_segmenter` so real hair is masked out (via `cv2.inpaint`) before the wig PNG is blended on top. Untested on real hardware (no webcam/mediapipe in the sandbox this was written in) — see PROGRESS.md for what still needs verification.
- `data/wigs/` — has one placeholder procedurally-generated bob-shape test asset (`wig_bob_test.png` / `.csv`, via `generate_sample_wig.py`) for exercising the pipeline before sourcing real hairstyle assets.

## Design decisions worth knowing before you change things

- **API surface**: uses the current MediaPipe **Tasks** API (`mediapipe.tasks.python.vision`), not the deprecated standalone `hair_segmentation.tflite` solution. If you see old-style `mp.solutions.*` code for hair segmentation anywhere, it's stale — don't copy it in.
- **Landmark IDs for hairline/temple anchors** (`CANDIDATE_IDS` in `pick_landmarks.py`) are **not verified ground truth** — they're community-sourced starting points. MediaPipe's face mesh is sparsest exactly in the hairline region. Always re-verify visually against `landmarks_check.jpg` before trusting them, especially if you change which anchor points are used.
- **Hairline landmarks are corrected via the hair mask, not used raw** — confirmed by real testing (2026-09-07) that landmark 10 (and its neighbors 109/338) sit mid-forehead, not at the actual hairline, because MediaPipe's face mesh doesn't extend into hair-covered area at all — landmark 10 is a fixed anatomical proportion relative to the eyes/eyebrows, not a hairline detection, so it's systematically wrong for anyone whose forehead height differs from the model's template. Fix: keep the landmark's X, override its Y by scanning the `hair_segmenter` mask **upward from the raw landmark's Y** (which is known to sit in forehead skin) until the first hair-covered row (`find_hairline_y()` in both `pick_landmarks.py` and `wig_overlay.py`). Temple (127/356) and chin (152) were visually confirmed accurate and are used as raw landmarks, unmodified.
  - **Gotcha already hit once**: the first version of `find_hairline_y()` scanned from the TOP of the image downward instead, which finds the crown (top of the head) instead of the hairline -- caught by re-testing with `pick_landmarks.py` after the first fix. If you touch this function, re-verify with a real photo before assuming it's right; a synthetic-mask unit test alone isn't enough (it can't tell you whether "upward from a known-skin start point" is the right search direction for real face geometry, only that the search itself is implemented correctly).
- **Homography, not full mesh warp**: `wig_overlay.py` uses `cv2.findHomography` (planar, rigid-ish). This is a known simplification — good enough for near-frontal poses, breaks down on extreme head tilt. If accuracy needs to improve, look at Delaunay triangulation + piecewise affine warp (see README for a reference link) before reaching for something heavier.
- **Recoloring math**: HSV Hue+Saturation blend toward target, Value channel untouched — this is deliberate, it's what keeps recolored hair looking like real hair instead of a flat color fill.
- **Real-hair removal**: `wig_overlay.py` now runs `hair_segmenter` every frame too (not just `face_landmarker`), thresholds the confidence mask, dilates it slightly (`HAIR_MASK_DILATE_PX`) so edges are fully covered, and fills it with `cv2.inpaint` (Telea algorithm) before the wig is warped on. This is a generic fill, not scene-aware — it doesn't know an ear or shirt collar is under there, so quality will vary. Running two models + inpaint per frame is noticeably heavier than either alone; if FPS is a problem, downscaling the frame before segmentation or switching to VIDEO running mode are the first things to try (see Next steps).

## What NOT to do

- Don't add a paid/commercial SDK dependency (Banuba, GlamAR, DeepAR, etc.) without discussion — the whole point of this repo is the free/open-source path.
- Don't hardcode landmark IDs into `wig_overlay.py` without going through `pick_landmarks.py` verification first for any new anchor scheme.
- Don't remove the confidence-mask feathering (`cv2.GaussianBlur` on the mask) — it's there to prevent jaggy edges at the hair/background boundary.

## Next steps (also tracked in PROGRESS.md)

1. Verify the hairline correction (`find_hairline_y`) with `pick_landmarks.py` on a real face — confirm the blue (corrected) points land at the actual hairline, not just closer than the red (raw) ones.
2. Test `wig_overlay.py` end-to-end on real webcam footage now that the hairline fix is in.
3. Replace the placeholder test asset (`wig_bob_test.*`) with real hairstyle assets; build out a small library in `data/wigs/`.
4. Tune/evaluate the `cv2.inpaint`-based hair removal quality on real footage — threshold, dilation, and inpaint radius constants in `wig_overlay.py` were chosen without empirical testing.
5. Evaluate VIDEO running mode (vs per-frame IMAGE mode) for better real-time performance, especially now that two models run per frame.
6. Consider Delaunay-triangulation warp as a follow-up if homography-only quality isn't good enough.

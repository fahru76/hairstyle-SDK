# PROGRESS.md

Running status log. Newest entries on top.

## 2026-09-07

- Repo rebuilt/pushed to `github.com/fahru76/hairstyle-SDK` (previous scaffolding was written in a throwaway session sandbox and never made it to git).
- Added: README, AGENTS.md, requirements.txt, .gitignore, LICENSE, models/README + download script, `src/hair_color/` (recolor_image.py, recolor_webcam.py), `src/wig_overlay/` (pick_landmarks.py, annotate_asset.py, wig_overlay.py), `data/wigs/.gitkeep`.
- Code has **not been executed** in any sandbox (mediapipe/opencv not installed, no camera access) — written directly against current official MediaPipe Tasks docs. Needs local testing.

## Open items / not yet done

- [ ] Test `recolor_image.py` / `recolor_webcam.py` on a real machine with mediapipe + opencv installed.
- [ ] Verify hairline/temple landmark IDs visually with `pick_landmarks.py` before trusting `wig_overlay.py`.
- [ ] Combine `hair_segmenter` output into `wig_overlay.py` to mask out real hair before blending the wig PNG (currently the overlay just draws on top — real hair can show through the edges).
- [ ] Populate `data/wigs/` with at least one real test asset + CSV to validate the end-to-end pipeline.
- [ ] Decide on VIDEO vs IMAGE running mode for webcam performance tuning.
- [ ] Longer-term: evaluate Delaunay triangulation + piecewise warp if homography-only overlay quality isn't sufficient for shipping.

## Background

Originated from a chat exploring free AR SDK options for hairstyle try-on (see conversation digest — MediaPipe chosen over commercial SDKs like Banuba/GlamAR/DeepAR because those are trial/limited, not free long-term).

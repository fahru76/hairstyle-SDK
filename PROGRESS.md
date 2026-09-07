# PROGRESS.md

Running status log. Newest entries on top.

## 2026-09-07 (5)

- **Bug found in the hairline fix itself, from testing the fix**: after (4) below, `pick_landmarks.py` output showed the "corrected" blue points landing at the very top of the head (crown), not the hairline. Root cause: `find_hairline_y()` scanned from the TOP of the image downward for the first hair-covered row -- that finds the crown (topmost point of the whole hair region), not the hairline (the boundary between hair and forehead skin).
- Fix: scan the other direction -- start at the raw landmark's Y (known to sit in forehead skin) and move UPWARD until hitting the first hair-covered row. That's the actual hairline boundary. Function signature changed to `find_hairline_y(mask, x, start_y, band=...)` in both `pick_landmarks.py` and `wig_overlay.py`.
- Verified with a synthetic-mask unit test (couldn't re-run against a real photo -- `storage.googleapis.com` isn't reachable from this sandbox to download models) confirming the new logic returns the hair/skin boundary correctly instead of row 0. **Still needs re-verification against a real photo** via `pick_landmarks.py` -- do this before trusting `wig_overlay.py` again.

## 2026-09-07 (4)

- **Found + fixed a real bug via testing**: `pick_landmarks.py` output on a real photo showed landmarks 10/109/338 sitting clearly mid-forehead, not at the hairline (visible gap between the red dots and where hair actually starts). Root cause: MediaPipe's face mesh doesn't extend into hair-covered area at all — landmark 10 is a fixed anatomical proportion relative to eyes/eyebrows, not a hairline detection, so it's systematically off for any forehead height that differs from the model's template. Temple (127/356) and chin (152) were confirmed accurate in the same test.
- Fix: `find_hairline_y()` added to both `pick_landmarks.py` and `wig_overlay.py` — keeps the landmark's X, replaces its Y by scanning the `hair_segmenter` confidence mask upward at that X for where hair pixels actually start (row-coverage threshold to avoid stray-pixel false positives, falls back to the raw landmark Y if no hair is found in that column). `pick_landmarks.py` now draws both the raw (red) and corrected (blue) points so the fix can be visually verified before trusting it.
- Not yet re-tested on real hardware after this change — next step is to re-run `pick_landmarks.py` and confirm the blue points actually land on the hairline.

## 2026-09-07 (3)

- **Confirmed working on real hardware**: `src/hair_color/recolor_image.py` tested on a real photo (Windows, Python 3.x venv). Hair color changed cleanly (black → dark burgundy), edges clean at the hairline, original texture/shading preserved as intended. First real-world validation of anything in this repo.
- Windows testing note: PowerShell aliases `curl` to `Invoke-WebRequest`, which doesn't accept curl-style flags (`-L -o`). Use `curl.exe` explicitly or `Invoke-WebRequest -Uri ... -OutFile ...` instead. `models/download_models.sh` itself needs Git Bash/WSL to run as-is on Windows.
- Next: `recolor_webcam.py` (real-time), then `pick_landmarks.py` + `wig_overlay.py`.

## 2026-09-07 (2)

- `src/wig_overlay/wig_overlay.py` now removes real hair before blending the wig: runs `hair_segmenter` each frame alongside `face_landmarker`, thresholds + dilates the confidence mask, and fills it with `cv2.inpaint` (Telea). Tunable via `HAIR_MASK_THRESHOLD`, `HAIR_MASK_DILATE_PX`, `INPAINT_RADIUS` constants at the top of the file.
- Added `data/wigs/wig_bob_test.png` + `.csv` — a procedurally generated placeholder bob-silhouette asset (see `data/wigs/generate_sample_wig.py`) so the overlay pipeline has something to run against before a real hairstyle asset is sourced. `wig_overlay.py`'s defaults now point at this test asset.
- Still **not executed on real hardware** — no mediapipe/opencv/webcam in the sandboxes this was written in. Everything below is unverified until run.

## 2026-09-07 (1)

- Repo rebuilt/pushed to `github.com/fahru76/hairstyle-SDK` (previous scaffolding was written in a throwaway session sandbox and never made it to git).
- Added: README, AGENTS.md, requirements.txt, .gitignore, LICENSE, models/README + download script, `src/hair_color/` (recolor_image.py, recolor_webcam.py), `src/wig_overlay/` (pick_landmarks.py, annotate_asset.py, wig_overlay.py), `data/wigs/.gitkeep`.
- Code has **not been executed** in any sandbox (mediapipe/opencv not installed, no camera access) — written directly against current official MediaPipe Tasks docs. Needs local testing.

## Open items / not yet done

- [ ] Test `recolor_image.py` / `recolor_webcam.py` on a real machine with mediapipe + opencv installed.
- [ ] Verify hairline/temple landmark IDs visually with `pick_landmarks.py` before trusting `wig_overlay.py`.
- [ ] Test the new hair-removal step in `wig_overlay.py` on real webcam footage — tune threshold/dilate/inpaint-radius constants against what actually looks right, current values are unverified guesses.
- [ ] Check real-time FPS with two models + inpaint running per frame; downscale-before-segment or VIDEO running mode if it's too slow.
- [ ] Replace `wig_bob_test.png` placeholder with a real hairstyle asset (or a small library of them) once the pipeline itself is confirmed working.
- [ ] Longer-term: evaluate Delaunay triangulation + piecewise warp if homography-only overlay quality isn't sufficient for shipping.

## Background

Originated from a chat exploring free AR SDK options for hairstyle try-on (see conversation digest — MediaPipe chosen over commercial SDKs like Banuba/GlamAR/DeepAR because those are trial/limited, not free long-term).

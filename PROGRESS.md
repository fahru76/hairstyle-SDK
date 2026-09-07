# PROGRESS.md

Running status log. Newest entries on top.

## 2026-09-07 (9)

- Tuned hair-removal constants using `tune_hair_removal.py` on real webcam footage. New values in `wig_overlay.py`: `HAIR_MASK_THRESHOLD = 0.26` (down from 0.5 -- the untuned guess was apparently too strict for this camera/lighting/hair color, missing real hair), `HAIR_MASK_DILATE_PX = 7`, `INPAINT_RADIUS = 9`, method stays Telea. These are fit to one test setup (one camera, one lighting condition, black hair) -- re-tune if any of those change enough to look wrong.

## 2026-09-07 (8)

- Added `src/wig_overlay/tune_hair_removal.py` — interactive webcam tool (OpenCV trackbars) to tune the hair-removal step live: threshold, dilate px, inpaint radius, inpaint method (Telea/NS), shown side by side as [original | mask | inpainted]. Press 'p' to print current values to paste into `wig_overlay.py`'s constants. The original `HAIR_MASK_THRESHOLD`/`HAIR_MASK_DILATE_PX`/`INPAINT_RADIUS` in `wig_overlay.py` were unverified guesses; this replaces "edit constants blind, re-run, repeat" with immediate visual feedback.
- `wig_overlay.py` also gained an `INPAINT_METHOD` constant (was hardcoded to `cv2.INPAINT_TELEA`) so a value found via the tuner can be dropped straight in.
- Not yet run/tuned against real footage by the user — next step.

## 2026-09-07 (7)

- **Full pipeline confirmed working end-to-end on real webcam footage** (~10s test clip). All pieces validated together for the first time:
  - Face tracking stable across frames, no jitter/misalignment as the head moves.
  - Hair removal clean -- no real hair visible poking out from under the wig edges, inpainted area looks like natural skin tone.
  - Homography warp + blend correctly keeps the face (eyes, nose, mouth) visible through the wig's face-hole, tracking head movement.
  - Hairline anchor sits at a reasonable position on the forehead (not too high/low).
  - Placeholder asset unsurprisingly reads as a "swim cap" rather than a hairstyle -- expected, it's just a test ring shape, not a real wig asset.
- **This closes out pipeline-mechanics testing.** Remaining work is now about asset quality (real hairstyle PNGs) and polish (inpaint tuning, performance), not core functionality bugs.
- Didn't evaluate FPS/performance numerically from the clip -- worth asking if it felt laggy in person before deciding whether to optimize.

## 2026-09-07 (6)

- **Hairline fix confirmed working on real photo**: re-ran `pick_landmarks.py` after the scan-direction fix -- blue (corrected) points now land right at the actual hair/forehead boundary, matching visually. Hairline anchor logic is now considered solid.
- Anchor points fully validated: hairline (10/109/338, mask-corrected) ✅, temple (127/356) ✅, chin (152) ✅. All 6 anchors used by `wig_overlay.py` are now confirmed accurate on a real face.
- Next: run `wig_overlay.py` end-to-end (webcam) with the placeholder bob asset and see how the actual warp + wig placement + hair removal looks together.

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

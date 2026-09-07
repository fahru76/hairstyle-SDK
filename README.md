# hairstyle-SDK

For AR use in any online Barber Shop.

Free/open-source AR hairstyle try-on toolkit built on **Google MediaPipe Tasks** — no paid SDK, no watermark, no usage limits. Runs on-device.

## What this does

| Feature | Status | How |
|---|---|---|
| **Hair recoloring** (change hair color, keep texture/shading) | ✅ Working, confirmed on real footage | `hair_segmenter` model + HSV hue/saturation shift on the segmented region |
| **Hairstyle change** (change cut/shape — e.g. long → bob, add fringe) | ✅ Pipeline confirmed working end-to-end; 🚧 assets are placeholders | 2D wig-overlay: `face_landmarker` + `hair_segmenter` anchor points + homography warp of a PNG asset, real hair removed via inpaint first |

MediaPipe's segmentation model can tell you *which pixels are hair* — it cannot generate a new hairstyle. Changing the actual cut/shape requires either:
1. Overlaying a pre-made 2D/3D hair asset anchored to face landmarks (what this repo implements as a skeleton), or
2. Generative AI (diffusion inpainting) — not implemented here.

## Repo layout

```
hairstyle-SDK/
├── README.md
├── AGENTS.md              # notes for AI coding agents picking this up
├── PROGRESS.md            # running status log
├── requirements.txt
├── .gitignore
├── LICENSE
├── models/
│   ├── README.md          # which models, where to download
│   └── download_models.sh
├── data/
│   └── wigs/               # wig PNG assets + their landmark-anchor CSVs go here
│       ├── generate_sample_wig.py    # generates wig_bob_test.png/.csv
│       └── generate_wig_library.py   # generates crew_cut/pompadour/undercut
└── src/
    ├── hair_color/
    │   ├── recolor_image.py    # static image hair recoloring
    │   └── recolor_webcam.py   # real-time webcam hair recoloring
    └── wig_overlay/
        ├── pick_landmarks.py      # visually verify face landmark IDs for anchors
        ├── annotate_asset.py      # click-anchor points on a wig PNG -> CSV
        ├── tune_hair_removal.py   # interactive sliders for hair-removal quality
        └── wig_overlay.py         # runtime: homography warp + blend wig onto webcam feed
```

## Quick start

```bash
pip install -r requirements.txt
bash models/download_models.sh

# hair color try-on (static image; needs an input.jpg in the repo root)
python src/hair_color/recolor_image.py

# hair color try-on (webcam, real-time)
python src/hair_color/recolor_webcam.py
```

**Windows (PowerShell) note:** `models/download_models.sh` is a bash script — it won't run as-is in PowerShell/CMD without Git Bash or WSL. Also, PowerShell aliases `curl` to `Invoke-WebRequest`, which doesn't accept curl's `-L -o` flags. Either use `curl.exe` explicitly, or download the models directly:

```powershell
curl.exe -L -o models\hair_segmenter.tflite https://storage.googleapis.com/mediapipe-models/image_segmenter/hair_segmenter/float32/latest/hair_segmenter.tflite
curl.exe -L -o models\face_landmarker.task https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task
```

Wig-overlay (hairstyle shape change) needs manual setup per asset — see `src/wig_overlay/` and `AGENTS.md` for the pipeline.

```bash
# try the full hairstyle-change pipeline (needs both models downloaded)
python src/wig_overlay/pick_landmarks.py input.jpg   # verify anchor points first
python src/wig_overlay/wig_overlay.py                # live webcam try-on
```

### Wig library (starter assets)

`data/wigs/` ships with a few **placeholder** hairstyle silhouettes, all procedurally generated (not photorealistic — see Known limitations) and all sharing the same anchor scheme, so any of them drops straight into `wig_overlay.py`'s `WIG_PNG_PATH` / `WIG_CSV_PATH`:

| File | Style |
|---|---|
| `wig_bob_test.png` / `.csv` | Bob (framing ring, open at chin) |
| `wig_crew_cut.png` / `.csv` | Short, tight, all-over crop |
| `wig_pompadour.png` / `.csv` | Crew-cut-like sides + swept-up front quiff |
| `wig_undercut.png` / `.csv` | Shaved sides (no coverage near temples) + fuller top |

Regenerate or add more with `python data/wigs/generate_wig_library.py` (edit the `STYLES` functions in that file, or `generate_sample_wig.py` for the bob) — or replace any of these with a real rendered/photographed hairstyle PNG (transparent background) plus a CSV made with `annotate_asset.py`.

## Known limitations

- **Wig overlay uses a rigid/planar homography** — it scales/rotates/shears a flat PNG but doesn't deform per-strand. Extreme head pitch can look pasted-on.
- **Bundled wig assets are procedural placeholders, not real hairstyles** — simple geometric silhouettes (see table above) for testing the pipeline's mechanics (tracking, warp, hair removal). They read as caps/rings, not as actual haircuts. Swap in real assets before anything user-facing.
- **Each hairstyle needs its own asset + CSV** — not scalable to hundreds of styles. Commercial SDKs (Banuba, GlamAR, etc.) use 3D hair mesh + physics instead.
- **Hairline/temple landmark IDs**: `face_landmarker`'s mesh doesn't extend into hair-covered area, so the hairline anchor is corrected against the `hair_segmenter` mask rather than used raw (see `find_hairline_y()` in `wig_overlay.py` / `pick_landmarks.py`). Confirmed accurate on one real face; always re-verify with `pick_landmarks.py` on a new face before trusting it.
- **Hair-removal quality (`cv2.inpaint`) is tuned for one test setup** (camera/lighting/hair color) — re-tune with `src/wig_overlay/tune_hair_removal.py` if results look wrong on a different setup.

## License

See `LICENSE`.

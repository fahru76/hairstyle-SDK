# hairstyle-SDK

For AR use in any online Barber Shop.

Free/open-source AR hairstyle try-on toolkit built on **Google MediaPipe Tasks** — no paid SDK, no watermark, no usage limits. Runs on-device.

## What this does

| Feature | Status | How |
|---|---|---|
| **Hair recoloring** (change hair color, keep texture/shading) | ✅ Working | `hair_segmenter` model + HSV hue/saturation shift on the segmented region |
| **Hairstyle change** (change cut/shape — e.g. long → bob, add fringe) | 🚧 Skeleton only | 2D wig-overlay: `face_landmarker` anchor points + homography warp of a PNG asset |

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
└── src/
    ├── hair_color/
    │   ├── recolor_image.py    # static image hair recoloring
    │   └── recolor_webcam.py   # real-time webcam hair recoloring
    └── wig_overlay/
        ├── pick_landmarks.py    # visually verify face landmark IDs for anchors
        ├── annotate_asset.py    # click-anchor points on a wig PNG -> CSV
        └── wig_overlay.py       # runtime: homography warp + blend wig onto webcam feed
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

## Known limitations

- **Wig overlay uses a rigid/planar homography** — it scales/rotates/shears a flat PNG but doesn't deform per-strand. Extreme head pitch can look pasted-on.
- **Original hair isn't removed** during wig overlay yet — needs combining with `hair_segmenter` to mask out real hair first (not yet implemented, see `PROGRESS.md`).
- **Each hairstyle needs its own asset + CSV** — not scalable to hundreds of styles. Commercial SDKs (Banuba, GlamAR, etc.) use 3D hair mesh + physics instead.
- **Hairline/temple landmark IDs are not precisely verified** — MediaPipe's landmark density is sparsest around the hairline. Always confirm with `pick_landmarks.py` before trusting an anchor ID.

## License

See `LICENSE`.

# Models

This repo does not commit model binaries (see `.gitignore`). Run `download_models.sh` to fetch them from Google's official MediaPipe model storage.

| Model | File | Used by | Purpose |
|---|---|---|---|
| Hair segmenter | `hair_segmenter.tflite` | `src/hair_color/*` | 512x512 input, binary mask (0=background, 1=hair) |
| Face landmarker | `face_landmarker.task` | `src/wig_overlay/*` | 478-point face mesh + facial transformation matrix (head pose) |

Both are pulled directly from `storage.googleapis.com/mediapipe-models/...` — official Google-hosted models, free, no API key required.

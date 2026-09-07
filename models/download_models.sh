#!/usr/bin/env bash
# Downloads the official MediaPipe models this repo needs.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "Downloading hair_segmenter.tflite..."
curl -L -o hair_segmenter.tflite \
  https://storage.googleapis.com/mediapipe-models/image_segmenter/hair_segmenter/float32/latest/hair_segmenter.tflite

echo "Downloading face_landmarker.task..."
curl -L -o face_landmarker.task \
  https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task

echo "Done. Models saved in $SCRIPT_DIR"

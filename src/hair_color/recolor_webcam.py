"""
Real-time hair recoloring from a webcam feed using MediaPipe Tasks
ImageSegmenter (hair_segmenter model).

Usage:
    python recolor_webcam.py

Press 'q' to quit.

Requires: models/hair_segmenter.tflite (run models/download_models.sh first)

Performance note: the model expects 512x512 input. For smoother webcam
performance, consider resizing the frame down before segment() and
resizing the mask back up to the original frame size before blending,
otherwise high-resolution webcam feeds may lag.
"""

import os
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

MODEL_PATH = os.path.join(
    os.path.dirname(__file__), "..", "..", "models", "hair_segmenter.tflite"
)

TARGET_COLOR_BGR = (40, 25, 140)  # change to whatever color you want to try


def recolor_hair(frame_bgr, alpha_mask, target_bgr, intensity=0.85):
    target_hsv = cv2.cvtColor(np.uint8([[target_bgr]]), cv2.COLOR_BGR2HSV)[0][0].astype(np.float32)
    frame_hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
    frame_hsv[..., 0] = np.clip(frame_hsv[..., 0] * (1 - intensity) + target_hsv[0] * intensity, 0, 179)
    frame_hsv[..., 1] = np.clip(frame_hsv[..., 1] * (1 - intensity) + target_hsv[1] * intensity, 0, 255)
    recolored = cv2.cvtColor(frame_hsv.astype(np.uint8), cv2.COLOR_HSV2BGR).astype(np.float32)
    mask_3ch = cv2.merge([alpha_mask] * 3)
    return (frame_bgr.astype(np.float32) * (1 - mask_3ch) + recolored * mask_3ch).astype(np.uint8)


def main():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Model not found at {MODEL_PATH}. Run models/download_models.sh first."
        )

    options = vision.ImageSegmenterOptions(
        base_options=mp_python.BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=vision.RunningMode.IMAGE,  # per-frame; consider VIDEO mode for better perf
        output_category_mask=False,
        output_confidence_masks=True,
    )

    with vision.ImageSegmenter.create_from_options(options) as segmenter:
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            raise RuntimeError("Could not open webcam (index 0).")

        while cap.isOpened():
            ok, frame = cap.read()
            if not ok:
                break

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            result = segmenter.segment(mp_image)

            hair_mask = result.confidence_masks[1].numpy_view().astype(np.float32)
            hair_mask = cv2.GaussianBlur(hair_mask, (0, 0), sigmaX=2)

            output = recolor_hair(frame, hair_mask, TARGET_COLOR_BGR)
            cv2.imshow("Hair Color Try-On", output)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

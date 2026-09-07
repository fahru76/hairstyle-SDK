"""
Hair recoloring on a static image using MediaPipe Tasks ImageSegmenter
(hair_segmenter model).

Usage:
    python recolor_image.py [input.jpg] [output.jpg]

Requires: models/hair_segmenter.tflite (run models/download_models.sh first)
"""

import sys
import os
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

MODEL_PATH = os.path.join(
    os.path.dirname(__file__), "..", "..", "models", "hair_segmenter.tflite"
)

# Example target color (BGR): dark burgundy. Change as needed.
TARGET_COLOR_BGR = (40, 25, 140)


def recolor_hair(frame_bgr, alpha_mask, target_bgr, intensity=0.85):
    """
    alpha_mask: float32 array 0..1 (hair confidence), same H/W as frame
    target_bgr: target color, e.g. (30, 20, 150) for dark red
    intensity: how strongly to pull toward target color (0..1)

    Shifts Hue + Saturation toward the target color while keeping the
    original Value (brightness) channel, so texture/highlights/shadow
    in the real hair are preserved -> looks realistic instead of flat.
    """
    target_hsv = cv2.cvtColor(np.uint8([[target_bgr]]), cv2.COLOR_BGR2HSV)[0][0].astype(np.float32)
    frame_hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV).astype(np.float32)

    frame_hsv[..., 0] = frame_hsv[..., 0] * (1 - intensity) + target_hsv[0] * intensity
    frame_hsv[..., 1] = frame_hsv[..., 1] * (1 - intensity) + target_hsv[1] * intensity
    frame_hsv[..., 0] = np.clip(frame_hsv[..., 0], 0, 179)
    frame_hsv[..., 1] = np.clip(frame_hsv[..., 1], 0, 255)

    recolored = cv2.cvtColor(frame_hsv.astype(np.uint8), cv2.COLOR_HSV2BGR).astype(np.float32)

    mask_3ch = cv2.merge([alpha_mask] * 3)
    out = frame_bgr.astype(np.float32) * (1 - mask_3ch) + recolored * mask_3ch
    return out.astype(np.uint8)


def main():
    input_path = sys.argv[1] if len(sys.argv) > 1 else "input.jpg"
    output_path = sys.argv[2] if len(sys.argv) > 2 else "output_hair_color.jpg"

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Model not found at {MODEL_PATH}. Run models/download_models.sh first."
        )

    options = vision.ImageSegmenterOptions(
        base_options=mp_python.BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=vision.RunningMode.IMAGE,
        output_category_mask=False,
        output_confidence_masks=True,  # continuous 0..1, smoother edges than category_mask
    )

    with vision.ImageSegmenter.create_from_options(options) as segmenter:
        image_bgr = cv2.imread(input_path)
        if image_bgr is None:
            raise FileNotFoundError(f"Could not read image: {input_path}")

        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)

        result = segmenter.segment(mp_image)
        hair_mask = result.confidence_masks[1].numpy_view().astype(np.float32)  # index 1 = hair

        # feather the mask edges so the blend isn't jaggy
        hair_mask = cv2.GaussianBlur(hair_mask, (0, 0), sigmaX=2)

        result_bgr = recolor_hair(image_bgr, hair_mask, target_bgr=TARGET_COLOR_BGR)
        cv2.imwrite(output_path, result_bgr)
        print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()

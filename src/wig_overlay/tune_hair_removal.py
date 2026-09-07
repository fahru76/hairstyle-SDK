"""
Interactive tuner for the hair-removal step used in wig_overlay.py
(hair_segmenter mask -> threshold -> dilate -> cv2.inpaint).

The threshold/dilate/inpaint-radius constants in wig_overlay.py were
written as unverified guesses before any real testing. This tool lets
you drag sliders on a live webcam feed and SEE the effect immediately,
instead of editing constants + re-running blind.

Usage:
    python tune_hair_removal.py

Shows three panels side by side:
    [ original | hair mask (white=detected as hair) | inpainted result ]

Sliders:
    Threshold   - confidence above this counts as "hair" (0-100 -> 0.0-1.0)
    Dilate px   - grows the mask so hair edges are fully covered
    Inpaint rad - cv2.inpaint neighborhood radius
    Method      - 0 = Telea, 1 = Navier-Stokes (cv2.INPAINT_NS)

Press 'p' to print the current values (copy these into wig_overlay.py's
HAIR_MASK_THRESHOLD / HAIR_MASK_DILATE_PX / INPAINT_RADIUS constants).
Press 'q' to quit.

Requires: models/hair_segmenter.tflite (run models/download_models.sh first)
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

WINDOW = "Hair Removal Tuner (p=print values, q=quit)"


def nothing(_):
    pass


def main():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Model not found at {MODEL_PATH}. Run models/download_models.sh first."
        )

    cv2.namedWindow(WINDOW)
    cv2.createTrackbar("Threshold x100", WINDOW, 50, 100, nothing)
    cv2.createTrackbar("Dilate px", WINDOW, 6, 30, nothing)
    cv2.createTrackbar("Inpaint radius", WINDOW, 8, 30, nothing)
    cv2.createTrackbar("Method (0=Telea,1=NS)", WINDOW, 0, 1, nothing)

    options = vision.ImageSegmenterOptions(
        base_options=mp_python.BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=vision.RunningMode.IMAGE,
        output_category_mask=False,
        output_confidence_masks=True,
    )

    with vision.ImageSegmenter.create_from_options(options) as segmenter:
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            raise RuntimeError("Could not open webcam (index 0).")

        last_values = None

        while cap.isOpened():
            ok, frame = cap.read()
            if not ok:
                break

            threshold = cv2.getTrackbarPos("Threshold x100", WINDOW) / 100.0
            dilate_px = cv2.getTrackbarPos("Dilate px", WINDOW)
            inpaint_radius = max(1, cv2.getTrackbarPos("Inpaint radius", WINDOW))
            method_idx = cv2.getTrackbarPos("Method (0=Telea,1=NS)", WINDOW)
            inpaint_method = cv2.INPAINT_TELEA if method_idx == 0 else cv2.INPAINT_NS
            last_values = (threshold, dilate_px, inpaint_radius, method_idx)

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            result = segmenter.segment(mp_image)
            hair_confidence = result.confidence_masks[1].numpy_view().astype(np.float32)

            binary_mask = (hair_confidence > threshold).astype(np.uint8) * 255
            if dilate_px > 0:
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilate_px, dilate_px))
                binary_mask = cv2.dilate(binary_mask, kernel)

            if np.any(binary_mask):
                inpainted = cv2.inpaint(frame, binary_mask, inpaint_radius, inpaint_method)
            else:
                inpainted = frame.copy()

            mask_bgr = cv2.cvtColor(binary_mask, cv2.COLOR_GRAY2BGR)

            h, w = frame.shape[:2]
            label_h = 24
            panels = []
            for img, label in [(frame, "original"), (mask_bgr, "mask"), (inpainted, "inpainted")]:
                panel = np.zeros((h + label_h, w, 3), dtype=np.uint8)
                panel[label_h:, :, :] = img
                cv2.putText(panel, label, (8, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
                panels.append(panel)

            combined = np.hstack(panels)
            cv2.imshow(WINDOW, combined)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("p") and last_values is not None:
                t, d, r, m = last_values
                method_name = "Telea" if m == 0 else "NS"
                print(
                    f"HAIR_MASK_THRESHOLD = {t:.2f}\n"
                    f"HAIR_MASK_DILATE_PX = {d}\n"
                    f"INPAINT_RADIUS = {r}\n"
                    f"inpaint method = {method_name} "
                    f"(cv2.INPAINT_{'TELEA' if m == 0 else 'NS'})\n"
                )

        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

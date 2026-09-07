"""
Visually verify the anchor points used for wig/hairstyle overlay:
temple + chin come straight from MediaPipe face landmarks, but the three
"hairline" points (top-center, top-left, top-right) are corrected using
the hair_segmenter mask instead of used raw from face_landmarker.

WHY: MediaPipe's face mesh does not extend into hair-covered area at all
-- its topmost landmarks (e.g. ID 10) are a fixed anatomical proportion
relative to the eyes/eyebrows, not a detection of the actual visible
hairline. For anyone whose forehead height differs from the model's
"average" template (e.g. a higher/receded hairline), landmark 10 will
sit visibly BELOW the real hairline. Confirmed visually during testing
on 2026-09-07 -- see PROGRESS.md.

FIX: keep landmark 10/109/338 for their X position (horizontal placement
is still reasonable geometry), but instead of trusting their Y, scan the
hair_segmenter confidence mask upward at that X to find where hair
pixels actually start. Temple (127, 356) and chin (152) landmarks were
visually confirmed accurate and are used as-is.

Usage:
    python pick_landmarks.py your_face.jpg

Output: landmarks_check.jpg with:
    - green dots: every face landmark (for exploring alternative IDs)
    - red dots + ID number: raw landmark position for CANDIDATE_IDS
    - blue dots: hairline-corrected position (for the 3 hairline IDs only)
      -- compare red vs blue to judge how big the correction is.

Requires: models/face_landmarker.task AND models/hair_segmenter.tflite
(run models/download_models.sh first)
"""

import os
import sys
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

FACE_MODEL_PATH = os.path.join(
    os.path.dirname(__file__), "..", "..", "models", "face_landmarker.task"
)
HAIR_MODEL_PATH = os.path.join(
    os.path.dirname(__file__), "..", "..", "models", "hair_segmenter.tflite"
)

# Starting-point candidates: [hairline-center, hairline-left, hairline-right,
# temple-left, temple-right, chin]. The first three get their Y overridden
# by find_hairline_y() below; 127/356/152 are used as detected.
CANDIDATE_IDS = [10, 109, 338, 127, 356, 152]
HAIRLINE_OVERRIDE_IDS = {10, 109, 338}

HAIR_MASK_THRESHOLD = 0.5
HAIRLINE_SEARCH_BAND_PX = 6      # how wide a column to scan around x
HAIRLINE_MIN_ROW_COVERAGE = 0.6  # fraction of the band that must be "hair" to count as the hairline row


def find_hairline_y(hair_mask_binary, x, band=HAIRLINE_SEARCH_BAND_PX):
    """
    Scans a narrow vertical band around column x, top to bottom, and
    returns the first row where hair coverage crosses
    HAIRLINE_MIN_ROW_COVERAGE. Requiring a coverage fraction (not just any
    single hair pixel) avoids a stray misclassified pixel from higher up
    (e.g. a wisp, or noise) triggering a false hairline.
    """
    h, w = hair_mask_binary.shape[:2]
    x0 = max(0, x - band)
    x1 = min(w, x + band + 1)
    col_band = hair_mask_binary[:, x0:x1]
    row_coverage = col_band.mean(axis=1)
    rows = np.where(row_coverage >= HAIRLINE_MIN_ROW_COVERAGE)[0]
    if len(rows) == 0:
        return None
    return int(rows[0])


def main():
    input_path = sys.argv[1] if len(sys.argv) > 1 else "your_face.jpg"
    output_path = "landmarks_check.jpg"

    if not os.path.exists(FACE_MODEL_PATH):
        raise FileNotFoundError(
            f"Model not found at {FACE_MODEL_PATH}. Run models/download_models.sh first."
        )
    if not os.path.exists(HAIR_MODEL_PATH):
        raise FileNotFoundError(
            f"Model not found at {HAIR_MODEL_PATH}. Run models/download_models.sh first."
        )

    face_options = vision.FaceLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=FACE_MODEL_PATH),
        running_mode=vision.RunningMode.IMAGE,
        num_faces=1,
    )
    hair_options = vision.ImageSegmenterOptions(
        base_options=mp_python.BaseOptions(model_asset_path=HAIR_MODEL_PATH),
        running_mode=vision.RunningMode.IMAGE,
        output_category_mask=False,
        output_confidence_masks=True,
    )

    with vision.FaceLandmarker.create_from_options(face_options) as landmarker, \
         vision.ImageSegmenter.create_from_options(hair_options) as segmenter:

        image_bgr = cv2.imread(input_path)
        if image_bgr is None:
            raise FileNotFoundError(f"Could not read image: {input_path}")

        h, w = image_bgr.shape[:2]
        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        face_result = landmarker.detect(mp_image)
        if not face_result.face_landmarks:
            raise RuntimeError("No face detected in image.")
        face = face_result.face_landmarks[0]

        hair_result = segmenter.segment(mp_image)
        hair_confidence = hair_result.confidence_masks[1].numpy_view().astype(np.float32)
        hair_binary = (hair_confidence > HAIR_MASK_THRESHOLD).astype(np.uint8)

        # draw all landmarks so you can explore for a better index if needed
        for lm in face:
            x, y = int(lm.x * w), int(lm.y * h)
            cv2.circle(image_bgr, (x, y), 1, (0, 255, 0), -1)

        # highlight candidate anchors with their index number (raw landmark position)
        for idx in CANDIDATE_IDS:
            lm = face[idx]
            x, y = int(lm.x * w), int(lm.y * h)
            cv2.circle(image_bgr, (x, y), 4, (0, 0, 255), -1)
            cv2.putText(
                image_bgr, str(idx), (x + 5, y - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1
            )

            # for hairline IDs, also draw the mask-corrected point in blue
            if idx in HAIRLINE_OVERRIDE_IDS:
                corrected_y = find_hairline_y(hair_binary, x)
                if corrected_y is not None:
                    cv2.circle(image_bgr, (x, corrected_y), 4, (255, 0, 0), -1)
                    cv2.line(image_bgr, (x, y), (x, corrected_y), (255, 0, 0), 1)
                    cv2.putText(
                        image_bgr, f"{idx}-fix", (x + 5, corrected_y - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1
                    )
                else:
                    print(f"Warning: no hairline found via mask for ID {idx} at x={x}; "
                          f"falling back to raw landmark position.")

        cv2.imwrite(output_path, image_bgr)
        print(f"Saved: {output_path}")
        print("Red = raw face landmark. Blue = hairline corrected via hair_segmenter (for 10/109/338 only).")
        print("Blue points should sit right at your actual hairline. Green dots are all landmarks, for exploring alternatives.")


if __name__ == "__main__":
    main()

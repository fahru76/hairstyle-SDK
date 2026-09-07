"""
Visually verify which MediaPipe face landmark IDs to use as anchor points
for wig/hairstyle overlay.

IMPORTANT: landmark density around the hairline/forehead is MediaPipe's
sparsest region, so don't trust candidate IDs from the internet blindly.
Run this, open the output image, and confirm the highlighted points
actually land where you want your anchors (e.g. mid-forehead, left/right
temple) before using them in annotate_asset.py / wig_overlay.py.

Usage:
    python pick_landmarks.py your_face.jpg

Requires: models/face_landmarker.task (run models/download_models.sh first)
"""

import os
import sys
import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

MODEL_PATH = os.path.join(
    os.path.dirname(__file__), "..", "..", "models", "face_landmarker.task"
)

# Starting-point candidates for hairline/temple region. UNVERIFIED — confirm
# visually against landmarks_check.jpg before trusting these.
CANDIDATE_IDS = [10, 109, 338, 127, 356, 152]


def main():
    input_path = sys.argv[1] if len(sys.argv) > 1 else "your_face.jpg"
    output_path = "landmarks_check.jpg"

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Model not found at {MODEL_PATH}. Run models/download_models.sh first."
        )

    options = vision.FaceLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=vision.RunningMode.IMAGE,
        num_faces=1,
    )

    with vision.FaceLandmarker.create_from_options(options) as landmarker:
        image_bgr = cv2.imread(input_path)
        if image_bgr is None:
            raise FileNotFoundError(f"Could not read image: {input_path}")

        h, w = image_bgr.shape[:2]
        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        result = landmarker.detect(mp_image)
        if not result.face_landmarks:
            raise RuntimeError("No face detected in image.")

        face = result.face_landmarks[0]

        # draw all landmarks so you can explore for a better index if needed
        for lm in face:
            x, y = int(lm.x * w), int(lm.y * h)
            cv2.circle(image_bgr, (x, y), 1, (0, 255, 0), -1)

        # highlight candidate anchors with their index number
        for idx in CANDIDATE_IDS:
            lm = face[idx]
            x, y = int(lm.x * w), int(lm.y * h)
            cv2.circle(image_bgr, (x, y), 4, (0, 0, 255), -1)
            cv2.putText(
                image_bgr, str(idx), (x + 5, y - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1
            )

        cv2.imwrite(output_path, image_bgr)
        print(f"Saved: {output_path}")
        print("Open it and check whether the red points land where you want your anchors.")
        print("If not, zoom into a green point nearby and update CANDIDATE_IDS.")


if __name__ == "__main__":
    main()

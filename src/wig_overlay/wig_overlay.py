"""
Runtime wig/hairstyle overlay: warps a wig PNG onto a live webcam feed
using homography between annotated asset points (from annotate_asset.py)
and live face landmarks (from face_landmarker), anchored on the IDs
verified in pick_landmarks.py.

Usage:
    python wig_overlay.py

Requires:
    - models/face_landmarker.task (run models/download_models.sh)
    - a wig PNG with alpha channel + its matching CSV (from annotate_asset.py)
      -> edit WIG_PNG_PATH / WIG_CSV_PATH below

KNOWN LIMITATIONS (see AGENTS.md / PROGRESS.md):
    - Homography is a rigid/planar warp, not per-strand deformation.
    - Real hair is NOT masked out yet -> if it pokes out from under the
      wig, it will show. Combine with hair_segmenter (see src/hair_color/)
      to fix this as a follow-up.
"""

import os
import csv
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

FACE_MODEL_PATH = os.path.join(
    os.path.dirname(__file__), "..", "..", "models", "face_landmarker.task"
)

# Point these at a real asset + CSV produced by annotate_asset.py
WIG_PNG_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "data", "wigs", "wig_bob.png")
WIG_CSV_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "data", "wigs", "wig_bob.csv")


def load_asset_points(csv_path):
    ids, pts = [], []
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            ids.append(int(row["landmark_id"]))
            pts.append([float(row["x"]), float(row["y"])])
    return ids, np.array(pts, dtype=np.float32)


def warp_and_blend(frame_bgr, wig_rgba, landmark_ids, src_pts, face_landmarks, w, h):
    dst_pts = np.array(
        [[face_landmarks[i].x * w, face_landmarks[i].y * h] for i in landmark_ids],
        dtype=np.float32,
    )

    # need >=4 non-collinear points for homography
    H, status = cv2.findHomography(src_pts, dst_pts, method=0)
    if H is None:
        return frame_bgr

    warped = cv2.warpPerspective(
        wig_rgba, H, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT
    )

    alpha = warped[:, :, 3:4].astype(np.float32) / 255.0
    fg = warped[:, :, :3].astype(np.float32)
    bg = frame_bgr.astype(np.float32)

    out = fg * alpha + bg * (1 - alpha)
    return out.astype(np.uint8)


def main():
    if not os.path.exists(FACE_MODEL_PATH):
        raise FileNotFoundError(
            f"Model not found at {FACE_MODEL_PATH}. Run models/download_models.sh first."
        )
    if not os.path.exists(WIG_CSV_PATH):
        raise FileNotFoundError(
            f"No CSV at {WIG_CSV_PATH}. Run annotate_asset.py on your wig PNG first."
        )

    landmark_ids, src_pts = load_asset_points(WIG_CSV_PATH)
    wig_rgba = cv2.imread(WIG_PNG_PATH, cv2.IMREAD_UNCHANGED)
    if wig_rgba is None:
        raise FileNotFoundError(f"Could not read wig PNG: {WIG_PNG_PATH}")
    if wig_rgba.shape[2] != 4:
        raise ValueError("Wig PNG must have an alpha channel (transparent background)")

    options = vision.FaceLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=FACE_MODEL_PATH),
        running_mode=vision.RunningMode.IMAGE,
        num_faces=1,
    )

    with vision.FaceLandmarker.create_from_options(options) as landmarker:
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            raise RuntimeError("Could not open webcam (index 0).")

        while cap.isOpened():
            ok, frame = cap.read()
            if not ok:
                break

            h, w = frame.shape[:2]
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            result = landmarker.detect(mp_image)

            if result.face_landmarks:
                face_landmarks = result.face_landmarks[0]
                frame = warp_and_blend(
                    frame, wig_rgba, landmark_ids, src_pts, face_landmarks, w, h
                )

            cv2.imshow("Hairstyle Try-On", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

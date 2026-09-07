"""
Runtime wig/hairstyle overlay: warps a wig PNG onto a live webcam feed
using homography between annotated asset points (from annotate_asset.py)
and live face landmarks (from face_landmarker), anchored on the IDs
verified in pick_landmarks.py.

Before the wig is blended on, the user's real hair is masked out with
the hair_segmenter model (same one used in src/hair_color/) and
inpainted over, so real hair doesn't poke out from under the wig edges.

Usage:
    python wig_overlay.py

Requires:
    - models/face_landmarker.task
    - models/hair_segmenter.tflite
      (run models/download_models.sh for both)
    - a wig PNG with alpha channel + its matching CSV (from annotate_asset.py,
      or the placeholder in data/wigs/wig_bob_test.* from generate_sample_wig.py)
      -> edit WIG_PNG_PATH / WIG_CSV_PATH below

KNOWN LIMITATIONS (see AGENTS.md / PROGRESS.md):
    - Homography is a rigid/planar warp, not per-strand deformation.
    - Hair removal uses cv2.inpaint, a general-purpose fill -- it
      approximates skin/background under the hair, it doesn't know what's
      actually there (e.g. an ear or the collar of a shirt). Works
      reasonably for short/contained hair, degrades for very long hair
      extending past the frame edge or over the shoulders.
    - Running two models (segmenter + landmarker) plus inpainting every
      frame is heavier than either alone -- expect lower FPS than
      src/hair_color/recolor_webcam.py. Consider VIDEO running mode or
      downscaling the frame before segmentation if this is too slow.
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
HAIR_MODEL_PATH = os.path.join(
    os.path.dirname(__file__), "..", "..", "models", "hair_segmenter.tflite"
)

# Point these at a real asset + CSV produced by annotate_asset.py.
# Defaults to the placeholder test asset from generate_sample_wig.py.
WIG_PNG_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "data", "wigs", "wig_bob_test.png")
WIG_CSV_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "data", "wigs", "wig_bob_test.csv")

# Hair-removal tuning
HAIR_MASK_THRESHOLD = 0.5   # confidence above this counts as "hair" to remove
HAIR_MASK_DILATE_PX = 6     # grow the mask a bit so hair edges are fully covered
INPAINT_RADIUS = 8          # cv2.inpaint neighborhood radius


def load_asset_points(csv_path):
    ids, pts = [], []
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            ids.append(int(row["landmark_id"]))
            pts.append([float(row["x"]), float(row["y"])])
    return ids, np.array(pts, dtype=np.float32)


def remove_real_hair(frame_bgr, hair_confidence_mask):
    """
    Masks out the user's real hair and inpaints over it, so it doesn't
    show through/around the wig overlay. hair_confidence_mask is a
    float32 array (0..1), same H/W as frame_bgr.
    """
    binary_mask = (hair_confidence_mask > HAIR_MASK_THRESHOLD).astype(np.uint8) * 255

    if HAIR_MASK_DILATE_PX > 0:
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (HAIR_MASK_DILATE_PX, HAIR_MASK_DILATE_PX)
        )
        binary_mask = cv2.dilate(binary_mask, kernel)

    if not np.any(binary_mask):
        return frame_bgr

    return cv2.inpaint(frame_bgr, binary_mask, INPAINT_RADIUS, cv2.INPAINT_TELEA)


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
    if not os.path.exists(HAIR_MODEL_PATH):
        raise FileNotFoundError(
            f"Model not found at {HAIR_MODEL_PATH}. Run models/download_models.sh first."
        )
    if not os.path.exists(WIG_CSV_PATH):
        raise FileNotFoundError(
            f"No CSV at {WIG_CSV_PATH}. Run annotate_asset.py on your wig PNG first "
            f"(or generate_sample_wig.py for a placeholder test asset)."
        )

    landmark_ids, src_pts = load_asset_points(WIG_CSV_PATH)
    wig_rgba = cv2.imread(WIG_PNG_PATH, cv2.IMREAD_UNCHANGED)
    if wig_rgba is None:
        raise FileNotFoundError(f"Could not read wig PNG: {WIG_PNG_PATH}")
    if wig_rgba.shape[2] != 4:
        raise ValueError("Wig PNG must have an alpha channel (transparent background)")

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

            face_result = landmarker.detect(mp_image)

            if face_result.face_landmarks:
                # 1) remove real hair first
                hair_result = segmenter.segment(mp_image)
                hair_mask = hair_result.confidence_masks[1].numpy_view().astype(np.float32)
                frame = remove_real_hair(frame, hair_mask)

                # 2) then warp + blend the wig on top
                face_landmarks = face_result.face_landmarks[0]
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

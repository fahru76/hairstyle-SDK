"""
Generates a simple procedural placeholder wig/hairstyle PNG asset
(bob-style silhouette, transparent background) for testing the
wig_overlay.py pipeline end-to-end, plus its matching landmark-anchor
CSV so you don't have to click-annotate it by hand for this first test.

This is a PLACEHOLDER for pipeline testing, not a production-quality
hairstyle asset. Replace with a real rendered/photographed wig PNG
once the pipeline itself is verified to work.

Usage:
    python generate_sample_wig.py

Outputs (in this same folder):
    wig_bob_test.png
    wig_bob_test.csv
"""

import csv
import os
from PIL import Image, ImageDraw, ImageFilter

OUT_DIR = os.path.dirname(__file__)
CANVAS = (800, 800)
HAIR_COLOR = (60, 38, 22, 255)  # dark brown, fully opaque where hair is

# Outer silhouette = overall hair boundary (bob cut: covers crown + sides
# down past the ears/jaw). Inner ellipse is cut out so the face underneath
# stays visible -- only the "ring" around it is opaque hair.
OUTER_BBOX = (90, 30, 710, 770)     # (x0, y0, x1, y1)
INNER_BBOX = (230, 160, 570, 660)   # face hole
# removes the part of the ring below the chin so the bob doesn't wrap
# under the jaw (open at the bottom, like a real bob cut)
CHIN_CUT_BBOX = (230, 590, 570, 950)

# Must match CANDIDATE_IDS order in src/wig_overlay/pick_landmarks.py:
# [10, 109, 338, 127, 356, 152]
# These are placed relative to the inner (face) hole so that when this
# asset is homography-warped onto a real face using the same landmark
# IDs, the hairline/temple/chin points roughly line up.
ANCHOR_POINTS = {
    10:  (400, 165),   # top hairline, center of forehead
    109: (320, 175),   # hairline, left of center
    338: (480, 175),   # hairline, right of center
    127: (228, 330),   # left temple
    356: (572, 330),   # right temple
    152: (400, 655),   # chin
}
LANDMARK_ORDER = [10, 109, 338, 127, 356, 152]


def main():
    img = Image.new("RGBA", CANVAS, (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    # outer hair shape
    draw.ellipse(OUTER_BBOX, fill=HAIR_COLOR)

    # cut-outs: face hole (center) + chin area (bottom), so hair doesn't
    # wrap fully around like a closed ring
    cuts = Image.new("RGBA", CANVAS, (0, 0, 0, 0))
    cuts_draw = ImageDraw.Draw(cuts)
    cuts_draw.ellipse(INNER_BBOX, fill=(255, 255, 255, 255))
    cuts_draw.ellipse(CHIN_CUT_BBOX, fill=(255, 255, 255, 255))

    # subtract cuts from hair silhouette via alpha compositing
    import numpy as np
    a_arr = np.array(img.split()[3], dtype=np.int16)
    cuts_a = np.array(cuts.split()[3], dtype=np.int16)
    new_a = np.clip(a_arr - cuts_a, 0, 255).astype("uint8")
    img.putalpha(Image.fromarray(new_a))

    # soften the edge a touch so it doesn't look razor-cut
    alpha = img.split()[3].filter(ImageFilter.GaussianBlur(2))
    img.putalpha(alpha)

    png_path = os.path.join(OUT_DIR, "wig_bob_test.png")
    img.save(png_path)

    csv_path = os.path.join(OUT_DIR, "wig_bob_test.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["landmark_id", "x", "y"])
        for lid in LANDMARK_ORDER:
            x, y = ANCHOR_POINTS[lid]
            writer.writerow([lid, x, y])

    print(f"Saved: {png_path}")
    print(f"Saved: {csv_path}")
    print(
        "\nThis is a placeholder shape for pipeline testing only. "
        "Point WIG_PNG_PATH / WIG_CSV_PATH in wig_overlay.py at these "
        "two files to test the overlay end-to-end."
    )


if __name__ == "__main__":
    main()

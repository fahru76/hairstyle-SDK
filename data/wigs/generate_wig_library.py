"""
Generates a small library of procedural PLACEHOLDER hairstyle assets
(PNG with alpha + matching landmark-anchor CSV) for a men's barbershop
try-on app -- crew cut, undercut, pompadour, plus the original bob from
generate_sample_wig.py.

IMPORTANT: these are simplified geometric silhouettes for exercising the
wig_overlay.py pipeline (variety of shapes, sanity-testing homography
warp on different silhouette geometry), NOT photorealistic hairstyle
renders. Swap in real rendered/photographed assets before shipping
anything user-facing -- see AGENTS.md / PROGRESS.md.

All styles share the same anchor-point scheme as wig_bob_test.png (same
virtual face template), so they're all drop-in compatible with
wig_overlay.py's WIG_PNG_PATH / WIG_CSV_PATH without any other changes.

Usage:
    python generate_wig_library.py

Outputs (in this folder):
    wig_crew_cut.png / .csv
    wig_undercut.png / .csv
    wig_pompadour.png / .csv
"""

import csv
import os
import numpy as np
from PIL import Image, ImageDraw, ImageFilter

OUT_DIR = os.path.dirname(__file__)
CANVAS = (800, 800)

# Shared across all styles -- same virtual face template as
# generate_sample_wig.py's wig_bob_test, so every style lines up with
# the same face_landmarker points in wig_overlay.py.
INNER_BBOX = (230, 160, 570, 660)  # face hole
LANDMARK_ORDER = [10, 109, 338, 127, 356, 152]
ANCHOR_POINTS = {
    10:  (400, 165),
    109: (320, 175),
    338: (480, 175),
    127: (228, 330),
    356: (572, 330),
    152: (400, 655),
}


def save_asset(name, img):
    png_path = os.path.join(OUT_DIR, f"wig_{name}.png")
    csv_path = os.path.join(OUT_DIR, f"wig_{name}.csv")

    img.save(png_path)

    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["landmark_id", "x", "y"])
        for lid in LANDMARK_ORDER:
            x, y = ANCHOR_POINTS[lid]
            writer.writerow([lid, x, y])

    print(f"Saved: {png_path}")
    print(f"Saved: {csv_path}")


def subtract(base_alpha, cut_shapes):
    """base_alpha: PIL 'L' image. cut_shapes: list of (draw_fn) callables
    that draw white (255) onto a fresh same-size mask. Returns new alpha
    with all cuts subtracted."""
    w, h = base_alpha.size
    cuts = Image.new("L", (w, h), 0)
    cuts_draw = ImageDraw.Draw(cuts)
    for draw_fn in cut_shapes:
        draw_fn(cuts_draw)
    base_arr = np.array(base_alpha, dtype=np.int16)
    cuts_arr = np.array(cuts, dtype=np.int16)
    return Image.fromarray(np.clip(base_arr - cuts_arr, 0, 255).astype("uint8"))


def make_ring(outer_bbox, inner_bbox, color, extra_cuts=None, feather=2):
    """Common pattern used by crew_cut/pompadour: a filled outer
    ellipse with the face hole (and optional extra shapes) cut out."""
    img = Image.new("RGBA", CANVAS, (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    draw.ellipse(outer_bbox, fill=color)

    cuts = [lambda d: d.ellipse(inner_bbox, fill=255)]
    if extra_cuts:
        cuts.extend(extra_cuts)

    alpha = subtract(img.split()[3], cuts)
    if feather:
        alpha = alpha.filter(ImageFilter.GaussianBlur(feather))
    img.putalpha(alpha)
    return img


def build_crew_cut():
    """Short, tight, all-over crop -- ring stays close to the head and
    ends above the ears (no chin-level coverage needed; the outer
    ellipse simply doesn't reach that far down)."""
    color = (30, 27, 24, 255)  # near-black
    outer_bbox = (150, 55, 650, 560)
    return make_ring(outer_bbox, INNER_BBOX, color)


def build_pompadour():
    """Crew-cut-like ring at the sides/back, plus a swept-up quiff
    volume at the front-center to suggest classic pompadour shape."""
    color = (45, 30, 18, 255)  # dark brown
    outer_bbox = (150, 55, 650, 560)

    img = make_ring(outer_bbox, INNER_BBOX, color, feather=0)

    # add the quiff bump: a smooth wave polygon at the front-top,
    # peaking above the ring's own top edge
    quiff = Image.new("RGBA", CANVAS, (0, 0, 0, 0))
    quiff_draw = ImageDraw.Draw(quiff)
    quiff_points = [
        (300, 130), (320, 60), (360, 15), (400, 0),
        (440, 15), (480, 60), (500, 130),
        (460, 100), (400, 85), (340, 100),
    ]
    quiff_draw.polygon(quiff_points, fill=color)

    combined = Image.alpha_composite(img, quiff)
    alpha = combined.split()[3].filter(ImageFilter.GaussianBlur(2))
    combined.putalpha(alpha)
    return combined


def build_undercut():
    """Shaved/short sides (no hair drawn near the temples at all --
    anchor points there will map onto real skin, which is correct for
    this style) with a fuller top cap."""
    color = (20, 18, 16, 255)
    cap_bbox = (250, 55, 550, 260)  # top-of-head cap only, doesn't reach temples

    img = Image.new("RGBA", CANVAS, (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    draw.ellipse(cap_bbox, fill=color)

    alpha = img.split()[3].filter(ImageFilter.GaussianBlur(2))
    img.putalpha(alpha)
    return img


def main():
    styles = {
        "crew_cut": build_crew_cut,
        "pompadour": build_pompadour,
        "undercut": build_undercut,
    }
    for name, builder in styles.items():
        save_asset(name, builder())

    print(
        "\nAll styles share the same anchor scheme as wig_bob_test.png "
        "(from generate_sample_wig.py) -- point wig_overlay.py's "
        "WIG_PNG_PATH / WIG_CSV_PATH at any of these to try it, no other "
        "changes needed."
    )
    print(
        "Reminder: these are simplified procedural silhouettes for "
        "pipeline testing, not photorealistic hairstyle assets."
    )


if __name__ == "__main__":
    main()

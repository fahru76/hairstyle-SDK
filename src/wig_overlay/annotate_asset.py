"""
Click-annotate anchor points on a wig/hairstyle PNG asset, matching the
landmark IDs chosen (and verified) in pick_landmarks.py.

Usage:
    python annotate_asset.py path/to/wig_bob.png

Click on the image in the SAME ORDER as landmark_ids below. Output is
saved as a CSV next to the PNG (e.g. wig_bob.csv), consumed by wig_overlay.py.
"""

import sys
import csv
import cv2

# Must match, in the same order, the IDs verified in pick_landmarks.py
landmark_ids = [10, 109, 338, 127, 356, 152]


def main():
    if len(sys.argv) < 2:
        print("Usage: python annotate_asset.py path/to/wig.png")
        sys.exit(1)

    asset_path = sys.argv[1]
    img = cv2.imread(asset_path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {asset_path}")

    clicked_points = []

    def on_click(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and len(clicked_points) < len(landmark_ids):
            clicked_points.append((x, y))
            print(f"Landmark {landmark_ids[len(clicked_points) - 1]} -> ({x},{y})")

    cv2.namedWindow("Annotate")
    cv2.setMouseCallback("Annotate", on_click)

    print("Click on the image in this order:", landmark_ids)
    while True:
        disp = img[:, :, :3].copy() if img.shape[2] == 4 else img.copy()
        for i, (x, y) in enumerate(clicked_points):
            cv2.circle(disp, (x, y), 4, (0, 0, 255), -1)
            cv2.putText(
                disp, str(landmark_ids[i]), (x + 5, y - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1
            )
        cv2.imshow("Annotate", disp)
        if cv2.waitKey(1) & 0xFF == ord("q") or len(clicked_points) == len(landmark_ids):
            break

    out_csv = asset_path.rsplit(".", 1)[0] + ".csv"
    with open(out_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["landmark_id", "x", "y"])
        for lid, (x, y) in zip(landmark_ids, clicked_points):
            writer.writerow([lid, x, y])

    print("Saved:", out_csv)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

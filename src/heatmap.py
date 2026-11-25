"""Build a movement heatmap from tracks.jsonl.

With no homography configured (configs/pitch.json), the heatmap is drawn in
image space. Fill in 4+ point correspondences in pitch.json to warp onto a
top-down 105x68 m pitch, which is the version worth putting in the README.

Usage:
    python src/heatmap.py --tracks tracks.jsonl --out heatmap.png          # all players
    python src/heatmap.py --tracks tracks.jsonl --id 7 --out player7.png   # one track
"""
import argparse
import json

import cv2
import numpy as np


def load_points(path: str, track_id: int | None):
    pts = []
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            if r["cls"] == "ball":
                continue
            if track_id is not None and r["id"] != track_id:
                continue
            pts.append(r["foot"])
    return np.array(pts, dtype=np.float32)


def load_homography(path: str):
    try:
        with open(path) as f:
            cfg = json.load(f)
    except FileNotFoundError:
        return None, (105, 68)
    img = cfg.get("image_points") or []
    pitch = cfg.get("pitch_points") or []
    size = tuple(cfg.get("pitch_size_m", [105, 68]))
    if len(img) >= 4 and len(img) == len(pitch):
        H, _ = cv2.findHomography(np.array(img, np.float32), np.array(pitch, np.float32))
        return H, size
    return None, size


def gaussian_heatmap(pts: np.ndarray, w: int, h: int) -> np.ndarray:
    acc = np.zeros((h, w), dtype=np.float32)
    for x, y in pts:
        xi, yi = int(round(x)), int(round(y))
        if 0 <= xi < w and 0 <= yi < h:
            acc[yi, xi] += 1.0
    acc = cv2.GaussianBlur(acc, (0, 0), sigmaX=max(w, h) * 0.015)
    if acc.max() > 0:
        acc /= acc.max()
    return acc


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tracks", default="tracks.jsonl")
    ap.add_argument("--pitch", default="configs/pitch.json")
    ap.add_argument("--id", type=int, default=None)
    ap.add_argument("--out", default="heatmap.png")
    ap.add_argument("--res", type=int, default=12, help="pixels per metre for pitch mode")
    args = ap.parse_args()

    pts = load_points(args.tracks, args.id)
    if len(pts) == 0:
        raise SystemExit("No points found for that filter.")

    H, (pw, ph) = load_homography(args.pitch)
    if H is not None:
        warped = cv2.perspectiveTransform(pts.reshape(-1, 1, 2), H).reshape(-1, 2)
        w, h = int(pw * args.res), int(ph * args.res)
        warped[:, 0] *= args.res
        warped[:, 1] = (ph - warped[:, 1]) * args.res  # flip y so up is the far goal
        field = draw_pitch(w, h)
        heat = gaussian_heatmap(warped, w, h)
    else:
        w = int(pts[:, 0].max()) + 20
        h = int(pts[:, 1].max()) + 20
        field = np.full((h, w, 3), 40, np.uint8)
        heat = gaussian_heatmap(pts, w, h)

    color = cv2.applyColorMap((heat * 255).astype(np.uint8), cv2.COLORMAP_JET)
    mask = (heat > 0.02)[..., None]
    out = np.where(mask, cv2.addWeighted(field, 0.45, color, 0.55, 0), field)
    cv2.imwrite(args.out, out)
    print(f"Wrote {args.out} ({len(pts)} positions, "
          f"{'pitch' if H is not None else 'image'} space)")


def draw_pitch(w: int, h: int) -> np.ndarray:
    img = np.full((h, w, 3), (60, 140, 60), np.uint8)
    white = (235, 235, 235)
    cv2.rectangle(img, (2, 2), (w - 3, h - 3), white, 2)
    cv2.line(img, (w // 2, 2), (w // 2, h - 3), white, 2)
    cv2.circle(img, (w // 2, h // 2), int(min(w, h) * 0.12), white, 2)
    return img


if __name__ == "__main__":
    main()

"""Make a short synthetic test clip that ACTUALLY contains people, so the whole
pipeline (real YOLO detection + tracking + teams + render) can be smoke-tested
without you having to supply footage yet.

We take Ultralytics' stock `bus.jpg` (which has several pedestrians), upscale it,
and pan/zoom a camera window across it for a few seconds, plus a moving "ball".
Output: assets/sample.mp4

    python scripts/make_test_clip.py
"""
from __future__ import annotations

import os
import urllib.request

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(__file__))
ASSETS = os.path.join(ROOT, "assets")
SRC_URL = "https://ultralytics.com/images/bus.jpg"


def _source_image() -> np.ndarray:
    os.makedirs(ASSETS, exist_ok=True)
    src = os.path.join(ASSETS, "bus.jpg")
    if not os.path.exists(src):
        print(f"downloading {SRC_URL}")
        urllib.request.urlretrieve(SRC_URL, src)
    img = cv2.imread(src)
    return cv2.resize(img, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)


def main() -> int:
    big = _source_image()
    H, W = big.shape[:2]
    out_w, out_h = 960, 540
    fps, seconds = 25, 12
    n = fps * seconds

    dest = os.path.join(ASSETS, "sample.mp4")
    vw = cv2.VideoWriter(dest, cv2.VideoWriter_fourcc(*"mp4v"), fps, (out_w, out_h))

    max_x, max_y = W - out_w, H - out_h
    for i in range(n):
        t = i / (n - 1)
        x = int((0.5 - 0.5 * np.cos(t * np.pi)) * max(0, max_x))
        y = int(0.2 * max(0, max_y) * np.sin(t * 2 * np.pi))
        y = max(0, min(max(0, max_y), y))
        window = big[y:y + out_h, x:x + out_w].copy()
        # a moving "ball" so the ball/involvement path is exercised too
        bx = int(out_w * (0.2 + 0.6 * t))
        by = int(out_h * (0.7 + 0.1 * np.sin(t * 6)))
        cv2.circle(window, (bx, by), 9, (245, 245, 245), -1)
        cv2.circle(window, (bx, by), 9, (40, 40, 40), 1)
        vw.write(window)

    vw.release()
    print(f"wrote {dest}  ({seconds}s @ {fps}fps, {out_w}x{out_h})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

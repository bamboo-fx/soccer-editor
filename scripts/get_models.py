"""Download free, soccer-specific YOLO weights into ./models/ and print the
config snippet to enable them.

    python scripts/get_models.py

Pulls (all free, no Roboflow inference subscription):
  * a dedicated soccer BALL model   (martinjolif/yolo-football-ball-detection, YOLO11n)
  * a soccer PLAYER model           (uisikdag/yolo-v8-football-players-detection)

If a repo's exact filename changes, we just grab the first *.pt it contains.
"""
from __future__ import annotations

import os
import sys

MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")


def grab(repo: str) -> str | None:
    from huggingface_hub import HfApi, hf_hub_download
    files = [f for f in HfApi().list_repo_files(repo) if f.endswith(".pt")]
    if not files:
        print(f"  ! no .pt found in {repo}")
        return None
    fn = files[0]
    print(f"  downloading {repo}/{fn} ...")
    path = hf_hub_download(repo, fn, local_dir=MODELS_DIR)
    return path


def _grab_osnet() -> None:
    """OSNet person-ReID weights (Market1501) for the `consolidate` stitching step."""
    dest = os.path.join(MODELS_DIR, "osnet_x1_0_market.pth")
    if os.path.exists(dest):
        return
    try:
        import gdown
        print("  downloading OSNet (osnet_x1_0_market.pth) ...")
        gdown.download("https://drive.google.com/uc?id=1vduhq5DpN2q1g4fYEZfPI17MJeh9qyrA",
                       dest, quiet=True)
    except Exception as e:
        print(f"  ! OSNet download skipped ({e}); ReID will fall back to resnet18")


def main() -> int:
    try:
        import huggingface_hub  # noqa: F401
    except ImportError:
        print("Install huggingface_hub first:  pip install huggingface_hub")
        return 1

    os.makedirs(MODELS_DIR, exist_ok=True)
    ball = grab("martinjolif/yolo-football-ball-detection")
    player = grab("uisikdag/yolo-v8-football-players-detection")
    _grab_osnet()

    print("\nDone. To use these, edit config.yaml -> model:")
    if player:
        print(f"  weights: {os.path.relpath(player)}")
        print("  person_classes: [1, 2, 3]   # check the model's class names; gk/player/ref")
    if ball:
        print(f"  ball_weights: {os.path.relpath(ball)}")
        print("  ball_class_in_ball_model: 0")
    print("\nTip: verify class ids with:  python -c \""
          "from ultralytics import YOLO; print(YOLO('PATH.pt').names)\"")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

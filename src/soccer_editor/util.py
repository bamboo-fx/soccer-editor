"""Small shared helpers: config loading, video metadata, paths, json io."""
from __future__ import annotations

import json
import os
from dataclasses import dataclass

import cv2
import yaml


def load_config(path: str | None) -> dict:
    """Load config.yaml, falling back to the repo default if path is None."""
    if path is None:
        here = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        path = os.path.join(here, "config.yaml")
    with open(path, "r") as f:
        return yaml.safe_load(f)


@dataclass
class VideoInfo:
    path: str
    fps: float
    width: int
    height: int
    frame_count: int

    @property
    def duration_s(self) -> float:
        return self.frame_count / self.fps if self.fps else 0.0


def probe_video(path: str) -> VideoInfo:
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {path}")
    info = VideoInfo(
        path=path,
        fps=cap.get(cv2.CAP_PROP_FPS) or 30.0,
        width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        frame_count=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
    )
    cap.release()
    return info


def stride_for(fps: float, fps_target: float) -> int:
    """How many source frames to skip to hit ~fps_target. 0/invalid -> every frame."""
    if not fps_target or fps_target <= 0 or fps <= 0:
        return 1
    return max(1, round(fps / fps_target))


def save_json(obj, path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def load_json(path: str):
    with open(path, "r") as f:
        return json.load(f)


def fmt_timestamp(seconds: float) -> str:
    m, s = divmod(seconds, 60)
    return f"{int(m):02d}:{s:05.2f}"

"""Stage 1: detect + track players and the ball through the video.

We drive Ultralytics' built-in tracker frame-by-frame (persist=True keeps tracker
state between calls) so we can sub-sample frames and stay in control of IO.

Outputs into <out>/:
    tracks.json      every sampled frame: players (with track id + box) and balls
    colors.json      per-track torso-color samples (fed to team clustering)
    thumbs/<id>.jpg  one representative crop per track (for the contact sheet)
"""
from __future__ import annotations

import os

import cv2
import numpy as np
from tqdm import tqdm

from .util import VideoInfo, save_json, stride_for


def _torso_color(crop: np.ndarray) -> list[float]:
    """Mean BGR of the shirt region, ignoring grass/shadow/white-line pixels.

    We sample the upper-middle of the box (torso), then in HSV drop low-saturation
    and very dark/bright pixels so background and shadows don't pollute the kit
    colour used for team clustering.
    """
    h, w = crop.shape[:2]
    if h < 4 or w < 4:
        return [0.0, 0.0, 0.0]
    torso = crop[int(h * 0.15):int(h * 0.55), int(w * 0.2):int(w * 0.8)]
    if torso.size == 0:
        torso = crop
    hsv = cv2.cvtColor(torso, cv2.COLOR_BGR2HSV)
    s, v = hsv[..., 1], hsv[..., 2]
    mask = (s > 40) & (v > 40) & (v < 250)
    pix = torso[mask]
    if pix.size < 9:                      # too few kit pixels — fall back to raw mean
        pix = torso.reshape(-1, 3)
    return [float(c) for c in pix.reshape(-1, 3).mean(axis=0)]


def run(info: VideoInfo, cfg: dict, out_dir: str) -> dict:
    from ultralytics import YOLO  # imported lazily so --help works without torch

    m = cfg["model"]
    person_classes = set(m["person_classes"])
    ball_cls = m["ball_class"]
    use_ball_model = bool(m.get("ball_weights"))

    model = YOLO(m["weights"])
    ball_model = YOLO(m["ball_weights"]) if use_ball_model else None
    # Track only the people with the main model; the ball comes from whichever
    # model is best (separate ball model if configured, else main model's ball cls).
    track_classes = sorted(person_classes if use_ball_model else person_classes | {ball_cls})

    stride = stride_for(info.fps, cfg["sampling"]["fps_target"])
    sampled_fps = info.fps / stride

    thumbs_dir = os.path.join(out_dir, "thumbs")
    os.makedirs(thumbs_dir, exist_ok=True)

    cap = cv2.VideoCapture(info.path)
    frames: list[dict] = []
    color_samples: dict[int, list[list[float]]] = {}
    best_thumb_score: dict[int, float] = {}

    total_sampled = (info.frame_count // stride) if info.frame_count else None
    pbar = tqdm(total=total_sampled, desc="detect+track", unit="frame")

    fidx = -1
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        fidx += 1
        if fidx % stride != 0:
            continue

        res = model.track(
            frame,
            persist=True,
            tracker=cfg["tracker"]["name"],
            classes=track_classes,
            conf=m["conf"],
            imgsz=m["imgsz"],
            verbose=False,
        )[0]

        players, balls = [], []

        # Ball from the dedicated model (if configured).
        if ball_model is not None:
            bres = ball_model.predict(
                frame, conf=m["ball_conf"], imgsz=m["imgsz"],
                classes=[m["ball_class_in_ball_model"]], verbose=False,
            )[0]
            if bres.boxes is not None:
                for box, cf in zip(bres.boxes.xyxy.cpu().numpy(),
                                   bres.boxes.conf.cpu().numpy()):
                    x1, y1, x2, y2 = [int(v) for v in box]
                    balls.append({"xyxy": [x1, y1, x2, y2], "conf": float(cf)})

        if res.boxes is not None and len(res.boxes) > 0:
            xyxy = res.boxes.xyxy.cpu().numpy()
            cls = res.boxes.cls.cpu().numpy().astype(int)
            conf = res.boxes.conf.cpu().numpy()
            ids = (
                res.boxes.id.cpu().numpy().astype(int)
                if res.boxes.id is not None
                else np.full(len(cls), -1)
            )
            for box, c, cf, tid in zip(xyxy, cls, conf, ids):
                x1, y1, x2, y2 = [int(v) for v in box]
                if ball_model is None and c == ball_cls:
                    balls.append({"xyxy": [x1, y1, x2, y2], "conf": float(cf)})
                    continue
                if c not in person_classes or tid < 0:
                    continue
                players.append({"id": int(tid), "xyxy": [x1, y1, x2, y2], "conf": float(cf)})

                crop = frame[max(0, y1):max(0, y2), max(0, x1):max(0, x2)]
                if crop.size:
                    color_samples.setdefault(int(tid), []).append(_torso_color(crop))
                    # keep the biggest, most-confident crop as this track's thumbnail
                    score = float(cf) * (x2 - x1) * (y2 - y1)
                    if score > best_thumb_score.get(int(tid), 0):
                        best_thumb_score[int(tid)] = score
                        cv2.imwrite(os.path.join(thumbs_dir, f"{int(tid)}.jpg"), crop)

        frames.append({
            "frame": fidx,
            "time_s": fidx / info.fps,
            "players": players,
            "balls": balls,
        })
        pbar.update(1)

    cap.release()
    pbar.close()

    tracks_meta = {
        tid: {"id": tid, "samples": len(s)} for tid, s in color_samples.items()
    }
    tracks_doc = {
        "video": info.path,
        "fps": info.fps,
        "stride": stride,
        "sampled_fps": sampled_fps,
        "width": info.width,
        "height": info.height,
        "frames": frames,
        "tracks": tracks_meta,
    }
    save_json(tracks_doc, os.path.join(out_dir, "tracks.json"))
    save_json(
        {str(tid): samples for tid, samples in color_samples.items()},
        os.path.join(out_dir, "colors.json"),
    )
    print(f"  tracked {len(tracks_meta)} player identities across {len(frames)} sampled frames")
    return tracks_doc

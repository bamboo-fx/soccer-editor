"""Stage 5/6: render the annotated highlight reel for the chosen player.

For each involvement segment we copy the source frames and, on each, draw a
spotlight (dim everyone else) + a marker ring under the target. Each clip opens
with a brief freeze-frame so the viewer locks onto the player. We also emit a
timestamp list (json + csv) of exactly when the player was involved.
"""
from __future__ import annotations

import csv
import os
import shutil
import subprocess

import cv2
import numpy as np
from tqdm import tqdm

from .geometry import compute_involvement
from .util import fmt_timestamp, load_json, save_json


def _annotate(frame, box, team_bgr, spotlight: bool, circle: bool):
    if box is None:
        return frame
    x1, y1, x2, y2 = box
    cx, cy = (x1 + x2) // 2, y2
    rx, ry = max(12, (x2 - x1) // 2 + 8), max(6, (y2 - y1) // 8 + 4)

    if spotlight:
        dim = (frame * 0.40).astype(np.uint8)
        mask = np.zeros(frame.shape[:2], np.uint8)
        r = int(1.3 * max(x2 - x1, y2 - y1))
        cv2.circle(mask, ((x1 + x2) // 2, (y1 + y2) // 2), r, 255, -1)
        mask3 = cv2.merge([mask, mask, mask]) > 0
        frame = np.where(mask3, frame, dim)

    if circle:
        cv2.ellipse(frame, (cx, cy), (rx, ry), 0, 0, 360, team_bgr, 3, cv2.LINE_AA)
        cv2.ellipse(frame, (cx, cy), (rx, ry), 0, 0, 360, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(frame, "YOU", (x1, max(18, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, team_bgr, 2, cv2.LINE_AA)
    return frame


def _team_color(out_dir: str, target_id: int):
    try:
        teams = load_json(os.path.join(out_dir, "teams.json"))
        label = teams["assignments"].get(str(target_id))
        if label is not None:
            return tuple(int(v) for v in teams["team_colors_bgr"][str(label)])
    except Exception:
        pass
    return (0, 215, 255)  # amber default


def render(out_dir: str, video_path: str, target_id: int, cfg: dict, dest: str) -> dict:
    tracks_doc = load_json(os.path.join(out_dir, "tracks.json"))
    fps = tracks_doc["fps"]
    segments, interp, stats = compute_involvement(tracks_doc, target_id, cfg)
    if not segments:
        raise RuntimeError(f"No involvement segments found for player #{target_id}.")

    team_bgr = _team_color(out_dir, target_id)
    hl = cfg["highlights"]
    spotlight, circle = hl["spotlight"], hl["circle"]
    freeze_n = int(hl["freeze_intro_seconds"] * fps)

    cap = cv2.VideoCapture(video_path)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    tmp = dest + ".tmp.mp4"
    writer = cv2.VideoWriter(tmp, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    rows = []
    for i, seg in enumerate(tqdm(segments, desc="render", unit="clip")):
        f0, f1 = int(seg["start_s"] * fps), int(seg["end_s"] * fps)
        cap.set(cv2.CAP_PROP_POS_FRAMES, f0)

        # freeze-frame intro on the first frame of the clip
        ok, first = cap.read()
        if not ok:
            continue
        intro = _annotate(first.copy(), interp.box_at(f0), team_bgr, spotlight, circle)
        for _ in range(freeze_n):
            writer.write(intro)

        cap.set(cv2.CAP_PROP_POS_FRAMES, f0)
        for f in range(f0, f1):
            ok, frame = cap.read()
            if not ok:
                break
            writer.write(_annotate(frame, interp.box_at(f), team_bgr, spotlight, circle))

        rows.append({
            "clip": i + 1,
            "start_s": round(seg["start_s"], 2),
            "end_s": round(seg["end_s"], 2),
            "start_tc": fmt_timestamp(seg["start_s"]),
            "end_tc": fmt_timestamp(seg["end_s"]),
        })

    cap.release()
    writer.release()
    _finalize(tmp, dest, fps)

    save_json(
        {"player": target_id, "mode": stats["reason"], "clips": rows, "stats": stats},
        os.path.splitext(dest)[0] + ".timestamps.json",
    )
    csv_path = os.path.splitext(dest)[0] + ".timestamps.csv"
    with open(csv_path, "w", newline="") as f:
        wtr = csv.DictWriter(f, fieldnames=["clip", "start_s", "end_s", "start_tc", "end_tc"])
        wtr.writeheader()
        wtr.writerows(rows)

    print(f"  {len(rows)} clips, mode='{stats['reason']}' (ball seen in "
          f"{stats['ball_seen_frames']} frames) -> {dest}")
    return {"segments": rows, "stats": stats, "dest": dest}


def _finalize(tmp: str, dest: str, fps: float) -> None:
    """Re-encode to browser-friendly H.264 if ffmpeg is available; else keep mp4v."""
    if shutil.which("ffmpeg"):
        try:
            subprocess.run(
                ["ffmpeg", "-y", "-i", tmp, "-c:v", "libx264", "-pix_fmt", "yuv420p",
                 "-movflags", "+faststart", "-loglevel", "error", dest],
                check=True,
            )
            os.remove(tmp)
            return
        except subprocess.CalledProcessError:
            pass
    shutil.move(tmp, dest)

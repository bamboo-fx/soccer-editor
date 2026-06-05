"""Stage 4: decide WHEN the chosen player is involved, and WHERE they are.

Involvement = the player is close to the ball (distance measured in multiples of
the player's own box height, so it auto-scales with camera zoom). If the ball is
barely ever detected (common with COCO weights on real footage) we fall back to
"presence" — every moment the player is on screen — so you still get a reel.

Produces:
  segments     [{start_s, end_s, reason}]  merged + padded time ranges
  box_at(f)    interpolated target box for any SOURCE frame (for annotation)
"""
from __future__ import annotations

from bisect import bisect_left


def _center(box):
    x1, y1, x2, y2 = box
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


def _best_ball(balls):
    if not balls:
        return None
    b = max(balls, key=lambda b: b["conf"])
    return _center(b["xyxy"])


def compute_involvement(tracks_doc: dict, target_id: int, cfg: dict):
    inv = cfg["involvement"]
    prox = inv["proximity_player_heights"]
    ball_memory = inv["ball_memory_frames"]
    fps = tracks_doc["fps"]

    frames = tracks_doc["frames"]
    target_boxes: dict[int, list[int]] = {}     # sampled frame -> box
    involved_times: list[float] = []
    ball_seen = 0
    last_ball = None
    last_ball_age = 10**9

    for rec in frames:
        tgt = next((p for p in rec["players"] if p["id"] == target_id), None)
        if tgt is not None:
            target_boxes[rec["frame"]] = tgt["xyxy"]

        ball = _best_ball(rec["balls"])
        if ball is not None:
            ball_seen += 1
            last_ball, last_ball_age = ball, 0
        else:
            last_ball_age += 1

        if tgt is None:
            continue
        usable_ball = last_ball if last_ball_age <= ball_memory else None
        if usable_ball is None:
            continue
        x1, y1, x2, y2 = tgt["xyxy"]
        ph = max(1.0, y2 - y1)
        tcx, tcy = _center(tgt["xyxy"])
        dist = ((tcx - usable_ball[0]) ** 2 + (tcy - usable_ball[1]) ** 2) ** 0.5
        if dist <= prox * ph:
            involved_times.append(rec["frame"] / fps)

    # Fallback: ball almost never detected -> use plain on-screen presence.
    reason = "near-ball"
    if ball_seen < max(3, 0.02 * len(frames)) or not involved_times:
        reason = "on-screen"
        involved_times = [f / fps for f in sorted(target_boxes)]

    segments = _segmentize(sorted(involved_times), cfg, tracks_doc)
    interp = _Interpolator(target_boxes, fps)
    return segments, interp, {"ball_seen_frames": ball_seen, "reason": reason}


def _segmentize(times, cfg, tracks_doc):
    if not times:
        return []
    hl = cfg["highlights"]
    pre, post = hl["pre_seconds"], hl["post_seconds"]
    merge_gap, min_seg = hl["merge_gap_seconds"], hl["min_segment_seconds"]
    sample_dt = tracks_doc["stride"] / tracks_doc["fps"]
    duration = tracks_doc["frames"][-1]["time_s"] + sample_dt

    # group consecutive involved samples (gap <= merge_gap) into raw spans
    spans = []
    s = e = times[0]
    for t in times[1:]:
        if t - e <= merge_gap:
            e = t
        else:
            spans.append((s, e))
            s = e = t
    spans.append((s, e))

    segments = []
    for s, e in spans:
        a, b = max(0.0, s - pre), min(duration, e + post)
        if b - a < min_seg:
            continue
        # merge with previous if padding caused overlap
        if segments and a <= segments[-1]["end_s"]:
            segments[-1]["end_s"] = max(segments[-1]["end_s"], b)
        else:
            segments.append({"start_s": a, "end_s": b, "reason": "involved"})
    return segments


class _Interpolator:
    """Linear-interpolate the target box for any source frame between samples."""

    def __init__(self, target_boxes: dict[int, list[int]], fps: float, max_gap_frames: int = 60):
        self.fps = fps
        self.keys = sorted(target_boxes)
        self.boxes = target_boxes
        self.max_gap = max_gap_frames

    def box_at(self, frame: int):
        if not self.keys:
            return None
        i = bisect_left(self.keys, frame)
        if i < len(self.keys) and self.keys[i] == frame:
            return self.boxes[self.keys[i]]
        lo = self.keys[i - 1] if i > 0 else None
        hi = self.keys[i] if i < len(self.keys) else None
        if lo is None or hi is None:
            return None
        if hi - lo > self.max_gap:
            return None
        t = (frame - lo) / (hi - lo)
        a, b = self.boxes[lo], self.boxes[hi]
        return [int(a[k] + t * (b[k] - a[k])) for k in range(4)]

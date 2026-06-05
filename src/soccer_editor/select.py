"""Stage 3: help the user PICK their player (instead of typing a jersey number).

Two ways to choose, both producing a track id you pass to `highlight`:
  * contact sheet  -> a grid PNG of every tracked player, labelled by id + team
                      colour. Open it, read off the id of the kid you care about.
  * tap (optional) -> opens a frame in a window; click on a player to print their
                      id. Needs a GUI (works on a local Mac, not over SSH).
"""
from __future__ import annotations

import os

import cv2
import numpy as np

from .util import load_json


def build_contact_sheet(out_dir: str, cols: int = 6, cell: int = 160) -> str:
    teams = load_json(os.path.join(out_dir, "teams.json"))
    assign = teams.get("assignments", {})
    team_colors = teams.get("team_colors_bgr", {})

    thumbs_dir = os.path.join(out_dir, "thumbs")
    thumb_ids = {int(f[:-4]) for f in os.listdir(thumbs_dir) if f.endswith(".jpg")}
    # only show ids that still exist in tracks.json (after ReID consolidation)
    tracks_path = os.path.join(out_dir, "tracks.json")
    if os.path.exists(tracks_path):
        live = {int(t) for t in load_json(tracks_path)["tracks"]}
        thumb_ids &= live
    ids = sorted(thumb_ids)
    if not ids:
        raise RuntimeError("No thumbnails found — run `analyze` first.")

    rows = (len(ids) + cols - 1) // cols
    pad, label_h = 8, 26
    cw, ch = cell + pad, cell + label_h + pad
    sheet = np.full((rows * ch + pad, cols * cw + pad, 3), 30, np.uint8)

    for i, tid in enumerate(ids):
        r, c = divmod(i, cols)
        x0, y0 = pad + c * cw, pad + r * ch
        crop = cv2.imread(os.path.join(thumbs_dir, f"{tid}.jpg"))
        if crop is not None:
            h, w = crop.shape[:2]
            s = min(cell / w, cell / h)
            crop = cv2.resize(crop, (int(w * s), int(h * s)))
            yh, xw = crop.shape[:2]
            sheet[y0:y0 + yh, x0:x0 + xw] = crop

        label = assign.get(str(tid))
        tc = tuple(team_colors.get(str(label), [200, 200, 200])) if label is not None else (200, 200, 200)
        cv2.rectangle(sheet, (x0, y0 + cell), (x0 + cell, y0 + cell + label_h), tuple(int(v) for v in tc), -1)
        txt = f"#{tid}" + (f"  T{label}" if label is not None else "")
        cv2.putText(sheet, txt, (x0 + 4, y0 + cell + 19),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

    path = os.path.join(out_dir, "contact_sheet.png")
    cv2.imwrite(path, sheet)
    print(f"  contact sheet -> {path}  ({len(ids)} players)")
    return path


def tap_to_select(out_dir: str, video_path: str, at_seconds: float = 5.0) -> int | None:
    """Open a frame; click a player to get their track id. Returns the id or None."""
    tracks = load_json(os.path.join(out_dir, "tracks.json"))
    fps = tracks["fps"]
    target_frame = int(at_seconds * fps)
    # find the sampled frame nearest the requested time
    rec = min(tracks["frames"], key=lambda fr: abs(fr["frame"] - target_frame))

    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, rec["frame"])
    ok, frame = cap.read()
    cap.release()
    if not ok:
        print("Could not read frame for tap-select.")
        return None

    for p in rec["players"]:
        x1, y1, x2, y2 = p["xyxy"]
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f"#{p['id']}", (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA)

    chosen = {"id": None}

    def on_click(event, x, y, *_):
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        for p in rec["players"]:
            x1, y1, x2, y2 = p["xyxy"]
            if x1 <= x <= x2 and y1 <= y <= y2:
                chosen["id"] = p["id"]
                print(f"selected player #{p['id']}")
                return

    win = "Click your player, then press any key"
    cv2.imshow(win, frame)
    cv2.setMouseCallback(win, on_click)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return chosen["id"]

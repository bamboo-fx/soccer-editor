"""Glue: run the analyze stage, and the highlight stage."""
from __future__ import annotations

import os

from . import detect_track, select, teams
from .util import load_config, probe_video


def analyze(video: str, out_dir: str, config: str | None = None) -> dict:
    cfg = load_config(config)
    info = probe_video(video)
    os.makedirs(out_dir, exist_ok=True)
    print(f"[1/3] detect + track  ({info.width}x{info.height}, {info.fps:.1f}fps, "
          f"{info.duration_s:.0f}s)")
    detect_track.run(info, cfg, out_dir)
    print("[2/3] team clustering")
    teams.run(out_dir, cfg["teams"]["num_teams"])
    print("[3/3] contact sheet")
    sheet = select.build_contact_sheet(out_dir)
    print(f"\nDone. Open {sheet}, find your player's #id, then run `highlight`.")
    return {"out_dir": out_dir, "contact_sheet": sheet}


def consolidate(out_dir: str, video: str, config: str | None = None) -> dict:
    """ReID-stitch fragmented tracklets, then re-cluster teams + rebuild the sheet."""
    from . import reid
    cfg = load_config(config)
    print("[1/3] ReID stitching")
    res = reid.stitch(out_dir, video, cfg)
    print("[2/3] re-cluster teams")
    teams.run(out_dir, cfg["teams"]["num_teams"])
    print("[3/3] rebuild contact sheet")
    select.build_contact_sheet(out_dir)
    print(f"\nConsolidated {res['before']} -> {res['after']} player identities.")
    return res


def highlight(out_dir: str, video: str, track_id: int, dest: str,
              config: str | None = None) -> dict:
    from . import highlight as hl  # avoid name clash with this function
    cfg = load_config(config)
    os.makedirs(os.path.dirname(os.path.abspath(dest)), exist_ok=True)
    return hl.render(out_dir, video, track_id, cfg, dest)

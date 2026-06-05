"""CLI entrypoint:  python -m soccer_editor <command> ...

Commands:
  analyze   VIDEO                 detect+track, cluster teams, build contact sheet
  pick      OUT VIDEO             open a frame, click your player, print their id
  highlight OUT VIDEO --track N   render the annotated reel + timestamps
"""
from __future__ import annotations

import argparse
import sys

from . import pipeline


def main(argv=None) -> int:
    argv = argv if argv is not None else sys.argv[1:]
    ap = argparse.ArgumentParser(prog="soccer_editor")
    ap.add_argument("-c", "--config", default=None, help="path to config.yaml")
    sub = ap.add_subparsers(dest="cmd", required=True)

    a = sub.add_parser("analyze", help="detect+track+teams+contact sheet")
    a.add_argument("video")
    a.add_argument("-o", "--out", default="runs/game", help="output dir")

    p = sub.add_parser("pick", help="click a player in a frame to get their id")
    p.add_argument("out")
    p.add_argument("video")
    p.add_argument("--at", type=float, default=5.0, help="seconds into the video")

    cn = sub.add_parser("consolidate", help="ReID-stitch fragmented tracklets into one id each")
    cn.add_argument("out")
    cn.add_argument("video")

    h = sub.add_parser("highlight", help="render reel for a chosen track id")
    h.add_argument("out")
    h.add_argument("video")
    h.add_argument("--track", type=int, required=True)
    h.add_argument("--dest", default=None, help="output mp4 (default OUT/reel_<id>.mp4)")

    args = ap.parse_args(argv)

    if args.cmd == "analyze":
        pipeline.analyze(args.video, args.out, args.config)
    elif args.cmd == "pick":
        from . import select
        tid = select.tap_to_select(args.out, args.video, args.at)
        print(f"\nchosen track id: {tid}" if tid is not None else "\nno player selected")
    elif args.cmd == "consolidate":
        pipeline.consolidate(args.out, args.video, args.config)
    elif args.cmd == "highlight":
        import os
        dest = args.dest or os.path.join(args.out, f"reel_{args.track}.mp4")
        pipeline.highlight(args.out, args.video, args.track, dest, args.config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

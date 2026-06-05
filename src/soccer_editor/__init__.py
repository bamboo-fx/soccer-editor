"""soccer-editor: pick a player, get their highlight reel from raw match footage.

Phase 0 MVP pipeline:
    analyze   -> detect + track players/ball, cluster teams, build a contact sheet
    highlight -> given a chosen track id, cut an annotated highlight reel + timestamps
"""

__version__ = "0.0.1"

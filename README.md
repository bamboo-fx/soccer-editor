# soccer-editor — Phase 0 MVP

Pick a player from raw match footage → get an **annotated highlight reel** of every
moment they're involved, plus a **timestamp list**. No paid APIs, runs locally.

Instead of typing a jersey number (the single least-reliable part of soccer CV on
amateur video), you **pick the player from a contact sheet** (or click them in a
frame). We then track that identity through the match and cut their reel.

```
raw video ──► detect+track ──► team clustering ──► contact sheet ──► YOU pick #id
                (YOLO+ByteTrack)                                          │
                                                                          ▼
        annotated reel + timestamps  ◄── render (spotlight/ring/freeze) ◄─ involvement
```

## What works today (verified end-to-end on CPU)

- **Detection + tracking** — Ultralytics YOLO + ByteTrack, persistent track ids.
- **Team clustering** — torso-colour KMeans into 2 teams (grass/shadow-gated).
- **Pick-your-player** — `contact_sheet.png` grid, or interactive `pick` (click a player).
- **Involvement** — "near the ball" segments, with automatic fallback to "on-screen"
  when the ball isn't reliably detected (so you always get a reel).
- **Highlight render** — spotlight (dim everyone else), marker ring, freeze-frame
  intro, H.264 output, plus `*.timestamps.json` / `.csv`.

## Quick start

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install -e .

# 0) (optional) make a synthetic test clip with real people in it
python scripts/make_test_clip.py            # -> assets/sample.mp4

# 1) analyze: detect+track, cluster teams, build the contact sheet
python -m soccer_editor analyze assets/sample.mp4 -o runs/game1

# 2) (optional) ReID: merge fragmented tracklets of the same player into one id
python -m soccer_editor consolidate runs/game1 assets/sample.mp4

# 3) open runs/game1/contact_sheet.png, read off your player's #id
#    (or: python -m soccer_editor pick runs/game1 assets/sample.mp4 --at 5)

# 4) render their highlight reel + timestamps
python -m soccer_editor highlight runs/game1 assets/sample.mp4 --track 7
#    -> runs/game1/reel_7.mp4 + reel_7.timestamps.{json,csv}
```

Point step 1 at your own 5-minute clip to try real footage.

## Outputs (in the `-o` run dir)

| file | what |
|---|---|
| `tracks.json` | every sampled frame: players (id+box) and balls |
| `teams.json` | track id → team + each team's colour |
| `thumbs/<id>.jpg` | one crop per tracked player |
| `contact_sheet.png` | the pick-your-player grid |
| `reel_<id>.mp4` | annotated highlight reel |
| `reel_<id>.timestamps.{json,csv}` | when the player was involved |

## Tuning

Everything lives in `config.yaml` — sampling fps, detection confidence/resolution,
tracker (ByteTrack vs BoT-SORT), involvement radius, clip padding/merging, and the
spotlight/ring/freeze toggles. No code changes needed to experiment.

## Upgrading the models (big accuracy win)

Default weights are generic COCO YOLO (`person` + `sports ball`) — fine to prove the
pipeline, weak on real soccer (small players, tiny ball). Swap in free, soccer-specific
weights:

```bash
python scripts/get_models.py        # downloads soccer player + ball models into models/
```

Then in `config.yaml` set `model.weights` to the player model (and adjust
`person_classes`), and `model.ball_weights` to the ball model. Sources:
- Players: [roboflow/sports](https://github.com/roboflow/sports) `football-player-detection.pt`
  (classes: ball/goalkeeper/player/referee) or
  [uisikdag/yolo-v8-football-players-detection](https://huggingface.co/uisikdag/yolo-v8-football-players-detection)
- Ball: [martinjolif/yolo-football-ball-detection](https://huggingface.co/martinjolif/yolo-football-ball-detection)
  (YOLO11n, single `ball` class — use a low confidence)

## Performance reality

~7 fps on a laptop CPU at 1280px. A 90-min game ≈ 162k frames at 30fps — process at
reduced `fps_target` (default 8) and lower `imgsz`, or use a GPU (Colab free tier works).

## ReID — identity persistence (Phase 1, in progress)

Even good trackers fragment one player into many ids over a long match. Two layers
address this:

- **Tracker-level ReID** — `trackers/botsort_reid.yaml` (set `tracker.name` to it)
  associates by appearance through short occlusions. Slower on CPU.
- **Tracklet stitching** — `consolidate` re-merges fragmented ids using person-ReID
  embeddings (**OSNet**, Market1501-trained) gated by no-time-overlap, small gap,
  same team, and plausible movement. Runs on CPU in seconds (no YOLO re-run).
  Configurable under `reid:` (embedder, thresholds). Falls back to a generic
  ResNet embedding if `torchreid`/weights are missing.

Known limit: stitching is conservative by design; tune `reid.appearance_thresh`.
True 90-min validation needs long footage (a 30s clip has little re-entry to stitch).

## Roadmap (Phase 1+)

1. **Better identity persistence** — ✅ OSNet stitching + BoT-SORT ReID in place;
   next: jersey-number as a stitch tie-breaker, validate on full-length footage.
2. **True touch detection** — dedicated ball model + possession logic.
3. **Real "within 20 feet"** — pitch-keypoint homography (`football-pitch-detection.pt`)
   to convert pixels → metres. Notes in `config.yaml`.
4. **Jersey number assist** — OCR to *suggest* "you're probably #7", layered on top of
   pick-your-player, never load-bearing.
5. **Web UI** — upload, click your player, download reel.

## Code map

```
src/soccer_editor/
  detect_track.py  YOLO + ByteTrack, torso-colour sampling, thumbnails
  teams.py         KMeans team clustering
  select.py        contact sheet + click-to-pick
  reid.py          OSNet tracklet stitching (identity consolidation)
  geometry.py      involvement segments + box interpolation
  highlight.py     spotlight/ring/freeze render + timestamps
  pipeline.py      orchestration   __main__.py  CLI
```

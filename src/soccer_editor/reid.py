"""ReID stitching: merge fragmented tracklets that are the same player.

Even BoT-SORT loses a player who leaves frame / is occluded for a while and then
restarts them under a new id. Over 90 minutes that means one kid becomes dozens of
ids. This post-process re-links them.

Approach (cost-free, runs on CPU in seconds — no YOLO re-run):
  1. For each track id, re-extract a few crops from the video (we already have the
     boxes) and embed each with a pretrained CNN (torchvision ResNet18 features).
     Average -> one L2-normalised appearance vector per track.
  2. Consider merging track A (earlier) into B (later) only when ALL hold:
       - their time spans don't overlap (a player can't be two tracks at once)
       - the gap between them is small (reid.max_gap_seconds)
       - same team (from teams.json)
       - appearance cosine similarity >= reid.appearance_thresh
       - the implied movement across the gap is physically plausible
  3. Greedy union-find merge (highest similarity first), keeping spans disjoint.
  4. Rewrite ids in tracks.json and re-cluster teams + contact sheet.

The embedding is pluggable — swap ResNet for a real ReID net (OSNet/torchreid) to
discriminate same-team players better; that's the production upgrade.
"""
from __future__ import annotations

import os

import cv2
import numpy as np

from .util import load_json, save_json


# ───────────────────────── appearance embedding ─────────────────────────
def _embedder(rc: dict):
    """Build the per-crop appearance encoder.

    reid.embedder: "osnet"  -> OSNet trained on person ReID (best individual
                               discrimination; needs torchreid + weights)
                   "resnet18" -> generic ImageNet features (no extra deps, weaker)
    Both return L2-normalised 512-d vectors with the same preprocessing, so the
    rest of the stitcher is identical.
    """
    import torch

    name = rc.get("embedder", "osnet")
    net = None
    if name == "osnet":
        try:
            import torchreid
            net = torchreid.models.build_model("osnet_x1_0", num_classes=751)
            w = rc.get("osnet_weights", "models/osnet_x1_0_market.pth")
            if os.path.exists(w):
                torchreid.utils.load_pretrained_weights(net, w)
        except Exception as e:                       # fall back gracefully
            print(f"  ! OSNet unavailable ({e}); using resnet18 embedding")
            net = None
    if net is None:
        import torchvision
        net = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.DEFAULT)
        net.fc = torch.nn.Identity()
    net.eval()

    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

    @torch.no_grad()
    def embed(crops_bgr):
        if not crops_bgr:
            return np.zeros((0, 512), np.float32)
        batch = []
        for c in crops_bgr:
            c = cv2.resize(c, (128, 256))            # ReID standard W x H
            c = cv2.cvtColor(c, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            batch.append(torch.from_numpy(c).permute(2, 0, 1))
        x = (torch.stack(batch) - mean) / std
        f = net(x).numpy()
        f /= (np.linalg.norm(f, axis=1, keepdims=True) + 1e-9)
        return f

    return embed


def _collect_track_geometry(frames, fps):
    """Per track: sorted (frame, box) plus first/last frame and centres."""
    g: dict[int, list] = {}
    for fr in frames:
        for p in fr["players"]:
            g.setdefault(p["id"], []).append((fr["frame"], p["xyxy"]))
    info = {}
    for tid, seq in g.items():
        seq.sort()
        info[tid] = {
            "seq": seq,
            "first": seq[0][0],
            "last": seq[-1][0],
            "first_c": _center(seq[0][1]),
            "last_c": _center(seq[-1][1]),
        }
    return info


def _center(b):
    return ((b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0)


def _track_embeddings(video, geom, frames_meta, crops_per_track, embed):
    """Re-read the needed frames once, crop each track, embed, average per track."""
    # which (frame -> [(tid, box)]) crops do we need?
    want: dict[int, list] = {}
    for tid, gi in geom.items():
        seq = gi["seq"]
        idxs = np.linspace(0, len(seq) - 1, min(crops_per_track, len(seq))).astype(int)
        for i in idxs:
            f, box = seq[i]
            want.setdefault(f, []).append((tid, box))

    crops: dict[int, list] = {tid: [] for tid in geom}
    cap = cv2.VideoCapture(video)
    for f in sorted(want):
        cap.set(cv2.CAP_PROP_POS_FRAMES, f)
        ok, frame = cap.read()
        if not ok:
            continue
        for tid, box in want[f]:
            x1, y1, x2, y2 = [max(0, int(v)) for v in box]
            crop = frame[y1:y2, x1:x2]
            if crop.size:
                crops[tid].append(crop)
    cap.release()

    emb = {}
    for tid, cl in crops.items():
        f = embed(cl)
        if len(f):
            v = f.mean(axis=0)
            emb[tid] = v / (np.linalg.norm(v) + 1e-9)
    return emb


# ───────────────────────────── stitching ─────────────────────────────
class _Union:
    def __init__(self, ids):
        self.p = {i: i for i in ids}

    def find(self, i):
        while self.p[i] != i:
            self.p[i] = self.p[self.p[i]]
            i = self.p[i]
        return i

    def union(self, a, b):
        self.p[self.find(a)] = self.find(b)


def stitch(out_dir: str, video: str, cfg: dict) -> dict:
    rc = cfg.get("reid", {})
    max_gap = rc.get("max_gap_seconds", 8.0)
    appth = rc.get("appearance_thresh", 0.86)
    max_speed = rc.get("max_speed_px_per_s", 1500.0)
    crops_per = rc.get("crops_per_track", 8)

    doc = load_json(os.path.join(out_dir, "tracks.json"))
    fps = doc["fps"]
    teams = load_json(os.path.join(out_dir, "teams.json"))["assignments"]
    geom = _collect_track_geometry(doc["frames"], fps)

    print(f"  embedding {len(geom)} tracklets ({rc.get('embedder', 'osnet')}) ...")
    emb = _track_embeddings(video, geom, doc["frames"], crops_per, _embedder(rc))

    ids = list(geom)
    uf = _Union(ids)
    # candidate merges: A ends before B starts, small gap, same team, look alike
    cands = []
    for a in ids:
        for b in ids:
            if a == b:
                continue
            ga, gb = geom[a], geom[b]
            gap_f = gb["first"] - ga["last"]
            if gap_f <= 0:
                continue                                  # overlap or wrong order
            gap_s = gap_f / fps
            if gap_s > max_gap:
                continue
            if teams.get(str(a)) != teams.get(str(b)):
                continue
            if a not in emb or b not in emb:
                continue
            sim = float(np.dot(emb[a], emb[b]))
            if sim < appth:
                continue
            dist = ((ga["last_c"][0] - gb["first_c"][0]) ** 2 +
                    (ga["last_c"][1] - gb["first_c"][1]) ** 2) ** 0.5
            if dist / max(gap_s, 1e-3) > max_speed:
                continue                                  # teleport — reject
            cands.append((sim, a, b))

    cands.sort(reverse=True)
    merged = []
    for sim, a, b in cands:
        ra, rb = uf.find(a), uf.find(b)
        if ra == rb:
            continue
        if _spans_overlap(ra, rb, uf, geom):
            continue
        uf.union(a, b)
        merged.append({"from": a, "into": b, "sim": round(sim, 3)})

    # canonical id = earliest-appearing member of each group
    groups: dict[int, list] = {}
    for i in ids:
        groups.setdefault(uf.find(i), []).append(i)
    id_map = {}
    for members in groups.values():
        canon = min(members, key=lambda t: geom[t]["first"])
        for m in members:
            id_map[m] = canon

    _rewrite(doc, id_map, out_dir)
    print(f"  stitched {len(geom)} -> {len(groups)} identities ({len(merged)} merges)")
    save_json({"merges": merged, "id_map": {str(k): v for k, v in id_map.items()},
               "before": len(geom), "after": len(groups)},
              os.path.join(out_dir, "reid.json"))
    return {"before": len(geom), "after": len(groups), "merges": merged}


def _spans_overlap(ra, rb, uf, geom):
    a = [geom[t] for t in geom if uf.find(t) == ra]
    b = [geom[t] for t in geom if uf.find(t) == rb]
    for x in a:
        for y in b:
            if x["first"] <= y["last"] and y["first"] <= x["last"]:
                return True
    return False


def _rewrite(doc, id_map, out_dir):
    for fr in doc["frames"]:
        for p in fr["players"]:
            p["id"] = id_map.get(p["id"], p["id"])
    new_tracks = {}
    for fr in doc["frames"]:
        for p in fr["players"]:
            new_tracks.setdefault(p["id"], {"id": p["id"], "samples": 0})["samples"] += 1
    doc["tracks"] = new_tracks
    save_json(doc, os.path.join(out_dir, "tracks.json"))

    # remap colour samples so team re-clustering uses consolidated ids
    cpath = os.path.join(out_dir, "colors.json")
    if os.path.exists(cpath):
        colors = load_json(cpath)
        merged_colors: dict[str, list] = {}
        for tid, samples in colors.items():
            canon = str(id_map.get(int(tid), int(tid)))
            merged_colors.setdefault(canon, []).extend(samples)
        save_json(merged_colors, cpath)

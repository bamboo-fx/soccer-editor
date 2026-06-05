"""Stage 2: split tracked players into teams by kit colour.

A deliberately simple stand-in for roboflow/sports' SigLIP+UMAP+KMeans approach:
we take each track's median torso colour and KMeans-cluster those into K teams.
Good enough to colour-label the contact sheet and pre-filter "your team".
Swap in appearance embeddings later for tougher kits.
"""
from __future__ import annotations

import os

import numpy as np
from sklearn.cluster import KMeans

from .util import load_json, save_json


def run(out_dir: str, num_teams: int = 2) -> dict:
    colors = load_json(os.path.join(out_dir, "colors.json"))
    track_ids, feats = [], []
    for tid, samples in colors.items():
        if not samples:
            continue
        track_ids.append(int(tid))
        feats.append(np.median(np.asarray(samples, dtype=float), axis=0))  # BGR

    result = {"num_teams": num_teams, "assignments": {}, "team_colors_bgr": {}}
    if len(track_ids) == 0:
        save_json(result, os.path.join(out_dir, "teams.json"))
        return result

    feats = np.asarray(feats)
    k = min(num_teams, len(track_ids))
    km = KMeans(n_clusters=k, n_init=10, random_state=0).fit(feats)

    for tid, label in zip(track_ids, km.labels_):
        result["assignments"][str(tid)] = int(label)
    for label in range(k):
        result["team_colors_bgr"][str(label)] = [
            int(round(v)) for v in km.cluster_centers_[label]
        ]

    save_json(result, os.path.join(out_dir, "teams.json"))
    counts = {l: int((km.labels_ == l).sum()) for l in range(k)}
    print(f"  teams: {counts} (cluster -> #players)")
    return result

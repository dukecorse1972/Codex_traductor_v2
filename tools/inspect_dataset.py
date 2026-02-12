#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import pickle
import random
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from tqdm import tqdm

from src.data.features import FeatureSpec, _extract_landmarks_list


def inspect_sample(pkl_path: Path) -> dict[str, Any]:
    with pkl_path.open("rb") as f:
        seq = pickle.load(f)
    if not isinstance(seq, list):
        raise ValueError(f"PKL {pkl_path} no contiene una lista de frames.")

    t = len(seq)
    frames_without_hands = 0
    pose_ok = 0
    pose_bad = 0

    for frame in seq:
        hands = frame.get("hands") if isinstance(frame, dict) else None
        hand_landmarks = _extract_landmarks_list(hands, "hand_landmarks") if hands is not None else None
        if not hand_landmarks:
            frames_without_hands += 1

        pose = frame.get("pose") if isinstance(frame, dict) else None
        pose_landmarks_batch = _extract_landmarks_list(pose, "pose_landmarks") if pose is not None else None
        pose_landmarks = pose_landmarks_batch[0] if pose_landmarks_batch and len(pose_landmarks_batch) > 0 else []

        if len(pose_landmarks) == 33:
            attrs_ok = all(all(hasattr(lm, a) for a in ["x", "y", "z", "visibility", "presence"]) for lm in pose_landmarks)
            if attrs_ok:
                pose_ok += 1
            else:
                pose_bad += 1
        else:
            pose_bad += 1

    return {
        "T": t,
        "frames_without_hands": frames_without_hands,
        "total_frames": t,
        "pose_ok": pose_ok,
        "pose_bad": pose_bad,
    }


def decide_t_fixed(lengths: list[int], percentile: int = 95, cap: int = 160, floor: int = 32) -> int:
    if not lengths:
        return 128
    p = int(np.percentile(np.asarray(lengths), percentile))
    return max(floor, min(cap, p))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_csv", type=Path, required=True)
    ap.add_argument("--mediapipe_dir", type=Path, required=True)
    ap.add_argument("--output", type=Path, default=Path("artifacts/feature_spec.json"))
    ap.add_argument("--n_samples", type=int, default=20)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    random.seed(args.seed)

    df = pd.read_csv(args.train_csv, header=None, names=["FILENAME", "CLASS_ID"])
    df["FILENAME"] = df["FILENAME"].astype(str)
    print(f"train rows: {len(df)}")

    ids = df["FILENAME"].tolist()
    sample_ids = random.sample(ids, k=min(args.n_samples, len(ids)))

    stats = []
    missing = 0
    corrupted = 0

    for sid in tqdm(sample_ids, desc="Inspecting PKLs"):
        p = args.mediapipe_dir / f"{sid}.pkl"
        if not p.exists():
            missing += 1
            continue
        try:
            stats.append(inspect_sample(p))
        except Exception as e:
            corrupted += 1
            print(f"[WARN] corrupt/invalid {p.name}: {e}")

    if not stats:
        raise RuntimeError("No se pudo inspeccionar ningún PKL válido.")

    lengths = [s["T"] for s in stats]
    total_frames = sum(s["total_frames"] for s in stats)
    no_hands = sum(s["frames_without_hands"] for s in stats)
    pose_ok = sum(s["pose_ok"] for s in stats)
    pose_bad = sum(s["pose_bad"] for s in stats)

    t_fixed = decide_t_fixed(lengths)
    spec = FeatureSpec.default(t_fixed=t_fixed)
    spec.to_json(args.output)

    print("\n=== Inspection summary ===")
    print(f"sampled_pkls: {len(stats)} | missing: {missing} | corrupted: {corrupted}")
    print(
        f"T stats -> min={int(np.min(lengths))}, p50={int(np.percentile(lengths, 50))}, "
        f"p95={int(np.percentile(lengths, 95))}, max={int(np.max(lengths))}"
    )
    pct_no_hands = 100.0 * no_hands / max(total_frames, 1)
    print(f"frames without hands: {no_hands}/{total_frames} ({pct_no_hands:.2f}%)")
    print(f"pose landmark integrity: ok_frames={pose_ok}, bad_frames={pose_bad}")
    print(f"chosen T_fixed: {t_fixed}")
    print(f"feature dim D_frame: {spec.d_frame}")
    print(f"saved feature spec to: {args.output}")


if __name__ == "__main__":
    main()

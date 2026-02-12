from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

POSE_LANDMARKS = 33
HAND_LANDMARKS = 21
POSE_DIM_PER_LM = 5  # x,y,z,visibility,presence
HAND_DIM_PER_LM = 3  # x,y,z
D_FRAME = POSE_LANDMARKS * POSE_DIM_PER_LM + 2 * HAND_LANDMARKS * HAND_DIM_PER_LM

LEFT_SHOULDER = 11
RIGHT_SHOULDER = 12
LEFT_HIP = 23
RIGHT_HIP = 24


@dataclass
class FeatureSpec:
    d_frame: int = D_FRAME
    t_fixed: int = 128
    feature_order: Optional[List[str]] = None
    normalization: Optional[Dict[str, Any]] = None
    padding: str = "right_zero_pad"
    truncation: str = "uniform_subsample"

    @classmethod
    def default(cls, t_fixed: int = 128) -> "FeatureSpec":
        order = []
        for i in range(POSE_LANDMARKS):
            order.extend(
                [
                    f"pose_{i}_x",
                    f"pose_{i}_y",
                    f"pose_{i}_z",
                    f"pose_{i}_visibility",
                    f"pose_{i}_presence",
                ]
            )
        for hand in ["left", "right"]:
            for i in range(HAND_LANDMARKS):
                order.extend([f"{hand}_hand_{i}_x", f"{hand}_hand_{i}_y", f"{hand}_hand_{i}_z"])

        return cls(
            d_frame=D_FRAME,
            t_fixed=t_fixed,
            feature_order=order,
            normalization={
                "center": "mid_hips_if_visible_else_mid_shoulders_else_origin",
                "scale": "shoulder_distance_if_valid_else_1.0",
                "hands_reference": "same_body_center_and_scale",
                "epsilon": 1e-6,
            },
            padding="right_zero_pad",
            truncation="uniform_subsample",
        )

    def to_json(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self), indent=2, ensure_ascii=False), encoding="utf-8")

    @classmethod
    def from_json(cls, path: str | Path) -> "FeatureSpec":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(**data)


def _get_attr(obj: Any, key: str, default: Any = None) -> Any:
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _extract_landmarks_list(container: Any, key: str) -> Any:
    value = _get_attr(container, key, None)
    if value is None and hasattr(container, "__dict__"):
        value = container.__dict__.get(key, None)
    return value


def _landmark_to_pose_vector(lm: Any) -> List[float]:
    return [
        float(_get_attr(lm, "x", 0.0) or 0.0),
        float(_get_attr(lm, "y", 0.0) or 0.0),
        float(_get_attr(lm, "z", 0.0) or 0.0),
        float(_get_attr(lm, "visibility", 0.0) or 0.0),
        float(_get_attr(lm, "presence", 0.0) or 0.0),
    ]


def _landmark_to_xyz(lm: Any) -> List[float]:
    return [
        float(_get_attr(lm, "x", 0.0) or 0.0),
        float(_get_attr(lm, "y", 0.0) or 0.0),
        float(_get_attr(lm, "z", 0.0) or 0.0),
    ]


def _safe_midpoint(a: Sequence[float], b: Sequence[float]) -> np.ndarray:
    return (np.asarray(a, dtype=np.float32) + np.asarray(b, dtype=np.float32)) / 2.0


def _visibility_ok(pose_arr: np.ndarray, idx: int, min_vis: float = 0.1) -> bool:
    if idx >= pose_arr.shape[0]:
        return False
    vis = pose_arr[idx, 3]
    pres = pose_arr[idx, 4]
    return (vis >= min_vis) and (pres >= min_vis)


def extract_frame_features(frame: Dict[str, Any]) -> np.ndarray:
    """Convert one frame dict to a fixed 291-D vector."""
    pose_container = frame.get("pose") if isinstance(frame, dict) else _get_attr(frame, "pose")
    hands_container = frame.get("hands") if isinstance(frame, dict) else _get_attr(frame, "hands")

    # Pose
    pose_landmarks_batch = _extract_landmarks_list(pose_container, "pose_landmarks") if pose_container is not None else None
    pose_landmarks = pose_landmarks_batch[0] if pose_landmarks_batch and len(pose_landmarks_batch) > 0 else []

    pose_array = np.zeros((POSE_LANDMARKS, POSE_DIM_PER_LM), dtype=np.float32)
    for i in range(min(len(pose_landmarks), POSE_LANDMARKS)):
        pose_array[i] = np.asarray(_landmark_to_pose_vector(pose_landmarks[i]), dtype=np.float32)

    # Body center and scale
    if _visibility_ok(pose_array, LEFT_HIP) and _visibility_ok(pose_array, RIGHT_HIP):
        center = _safe_midpoint(pose_array[LEFT_HIP, :3], pose_array[RIGHT_HIP, :3])
    elif _visibility_ok(pose_array, LEFT_SHOULDER) and _visibility_ok(pose_array, RIGHT_SHOULDER):
        center = _safe_midpoint(pose_array[LEFT_SHOULDER, :3], pose_array[RIGHT_SHOULDER, :3])
    else:
        center = np.zeros(3, dtype=np.float32)

    if pose_array.shape[0] > RIGHT_SHOULDER:
        shoulder_dist = np.linalg.norm(pose_array[LEFT_SHOULDER, :3] - pose_array[RIGHT_SHOULDER, :3])
    else:
        shoulder_dist = 0.0
    scale = float(shoulder_dist) if shoulder_dist > 1e-6 and np.isfinite(shoulder_dist) else 1.0

    pose_xyz = (pose_array[:, :3] - center[None, :]) / scale
    pose_norm = np.concatenate([pose_xyz, pose_array[:, 3:5]], axis=1)

    # Hands with fixed left/right slots
    left_slot = np.zeros((HAND_LANDMARKS, HAND_DIM_PER_LM), dtype=np.float32)
    right_slot = np.zeros((HAND_LANDMARKS, HAND_DIM_PER_LM), dtype=np.float32)

    hand_landmarks_list = _extract_landmarks_list(hands_container, "hand_landmarks") if hands_container is not None else None
    handedness_list = _extract_landmarks_list(hands_container, "handedness") if hands_container is not None else None
    hand_landmarks_list = hand_landmarks_list or []
    handedness_list = handedness_list or []

    assigned = {"left": False, "right": False}

    for i, hand_lms in enumerate(hand_landmarks_list):
        slot = None
        if i < len(handedness_list):
            categories = handedness_list[i]
            # Usually list[Category], first item has category_name
            cat = categories[0] if isinstance(categories, (list, tuple)) and len(categories) > 0 else categories
            cname = str(_get_attr(cat, "category_name", "")).lower()
            if "left" in cname:
                slot = "left"
            elif "right" in cname:
                slot = "right"
        if slot is None:
            if not assigned["left"]:
                slot = "left"
            elif not assigned["right"]:
                slot = "right"
            else:
                continue

        coords = np.zeros((HAND_LANDMARKS, HAND_DIM_PER_LM), dtype=np.float32)
        for j in range(min(len(hand_lms), HAND_LANDMARKS)):
            coords[j] = np.asarray(_landmark_to_xyz(hand_lms[j]), dtype=np.float32)
        coords = (coords - center[None, :]) / scale

        if slot == "left":
            left_slot = coords
            assigned["left"] = True
        else:
            right_slot = coords
            assigned["right"] = True

    feat = np.concatenate([pose_norm.reshape(-1), left_slot.reshape(-1), right_slot.reshape(-1)], axis=0)
    if feat.shape[0] != D_FRAME:
        raise ValueError(f"Frame feature dim mismatch: got {feat.shape[0]}, expected {D_FRAME}")
    return feat.astype(np.float32)


def sequence_to_fixed_tensor(
    frames: Sequence[Dict[str, Any]],
    t_fixed: int,
    augment: bool = False,
    temporal_dropout_p: float = 0.05,
    jitter_std: float = 0.005,
    landmark_dropout_p: float = 0.01,
    rng: Optional[random.Random] = None,
) -> np.ndarray:
    rng = rng or random
    seq = [extract_frame_features(frame) for frame in frames]
    if len(seq) == 0:
        return np.zeros((t_fixed, D_FRAME), dtype=np.float32)

    x = np.stack(seq, axis=0).astype(np.float32)  # (T,D)

    if augment:
        # Temporal dropout
        keep_mask = np.ones((x.shape[0],), dtype=bool)
        for t in range(x.shape[0]):
            if rng.random() < temporal_dropout_p:
                keep_mask[t] = False
        if keep_mask.any():
            x = x[keep_mask]
        # Coordinate jitter on xyz channels only
        xyz_indices = np.arange(D_FRAME)
        keep_coord = []
        for idx in xyz_indices:
            mod = idx % 5
            in_pose = idx < (POSE_LANDMARKS * POSE_DIM_PER_LM)
            in_hand = idx >= (POSE_LANDMARKS * POSE_DIM_PER_LM)
            if (in_pose and mod < 3) or (in_hand and (idx - POSE_LANDMARKS * POSE_DIM_PER_LM) % 3 < 3):
                keep_coord.append(idx)
        keep_coord = np.asarray(keep_coord, dtype=np.int64)
        noise = np.random.normal(0.0, jitter_std, size=(x.shape[0], keep_coord.shape[0])).astype(np.float32)
        x[:, keep_coord] += noise

        # Landmark dropout
        if landmark_dropout_p > 0:
            pose_end = POSE_LANDMARKS * POSE_DIM_PER_LM
            for lm in range(POSE_LANDMARKS):
                if rng.random() < landmark_dropout_p:
                    start = lm * POSE_DIM_PER_LM
                    x[:, start : start + POSE_DIM_PER_LM] = 0.0
            for hand_offset in [pose_end, pose_end + HAND_LANDMARKS * 3]:
                for lm in range(HAND_LANDMARKS):
                    if rng.random() < landmark_dropout_p:
                        start = hand_offset + lm * 3
                        x[:, start : start + 3] = 0.0

    # Fix length with uniform subsample + right zero pad
    t = x.shape[0]
    if t > t_fixed:
        idx = np.linspace(0, t - 1, t_fixed).astype(np.int64)
        x = x[idx]
    elif t < t_fixed:
        pad = np.zeros((t_fixed - t, D_FRAME), dtype=np.float32)
        x = np.concatenate([x, pad], axis=0)

    return x.astype(np.float32)

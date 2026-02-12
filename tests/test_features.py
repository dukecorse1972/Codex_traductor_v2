from types import SimpleNamespace

import numpy as np

from src.data.features import D_FRAME, extract_frame_features


def make_pose_landmarks():
    lms = []
    for i in range(33):
        lms.append(SimpleNamespace(x=0.1 * i, y=0.01 * i, z=-0.02 * i, visibility=0.9, presence=0.9))
    return [lms]


def make_hand_landmarks(offset=0.0):
    return [SimpleNamespace(x=offset + 0.01 * i, y=offset + 0.02 * i, z=offset - 0.01 * i) for i in range(21)]


def test_feature_dimension_291():
    frame = {
        "pose": SimpleNamespace(pose_landmarks=make_pose_landmarks()),
        "hands": SimpleNamespace(
            hand_landmarks=[make_hand_landmarks(0.2), make_hand_landmarks(0.4)],
            handedness=[
                [SimpleNamespace(category_name="Left", score=0.99)],
                [SimpleNamespace(category_name="Right", score=0.98)],
            ],
        ),
        "holistic_legacy": None,
    }
    feat = extract_frame_features(frame)
    assert feat.shape == (D_FRAME,)


def test_feature_order_deterministic():
    frame = {
        "pose": SimpleNamespace(pose_landmarks=make_pose_landmarks()),
        "hands": SimpleNamespace(
            hand_landmarks=[make_hand_landmarks(0.2)],
            handedness=[[SimpleNamespace(category_name="Left", score=0.99)]],
        ),
        "holistic_legacy": None,
    }

    f1 = extract_frame_features(frame)
    f2 = extract_frame_features(frame)
    assert np.allclose(f1, f2)

    # Right hand slot missing => must be zero
    right_start = 33 * 5 + 21 * 3
    assert np.allclose(f1[right_start:], 0.0)

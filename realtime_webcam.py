#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import time
from collections import deque
from pathlib import Path
from types import SimpleNamespace

import cv2
import mediapipe as mp
import numpy as np
import torch

from src.data.features import FeatureSpec, extract_frame_features
from src.models.tcn import GRUBaseline, TCNClassifier


def build_model(name: str, input_dim: int, num_classes: int):
    if name == "gru":
        return GRUBaseline(input_dim=input_dim, num_classes=num_classes)
    return TCNClassifier(input_dim=input_dim, num_classes=num_classes)


def lm_to_ns(lm, with_vis=False):
    if with_vis:
        return SimpleNamespace(x=lm.x, y=lm.y, z=lm.z, visibility=getattr(lm, "visibility", 1.0), presence=1.0)
    return SimpleNamespace(x=lm.x, y=lm.y, z=lm.z)


def mp_results_to_frame(pose_res, hands_res):
    pose_landmarks = []
    if pose_res.pose_landmarks:
        pose_landmarks = [[lm_to_ns(lm, with_vis=True) for lm in pose_res.pose_landmarks.landmark]]

    hand_landmarks = []
    handedness = []
    if hands_res.multi_hand_landmarks:
        for lms, hd in zip(hands_res.multi_hand_landmarks, hands_res.multi_handedness):
            hand_landmarks.append([lm_to_ns(lm, with_vis=False) for lm in lms.landmark])
            label = hd.classification[0].label
            score = hd.classification[0].score
            handedness.append([SimpleNamespace(category_name=label, score=score)])

    frame = {
        "pose": SimpleNamespace(pose_landmarks=pose_landmarks),
        "hands": SimpleNamespace(hand_landmarks=hand_landmarks, handedness=handedness),
        "holistic_legacy": None,
    }
    return frame


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=Path, default=Path("checkpoints/best.pt"))
    ap.add_argument("--feature_spec", type=Path, default=Path("artifacts/feature_spec.json"))
    ap.add_argument("--label_map", type=Path, default=Path("artifacts/label_map.json"))
    ap.add_argument("--camera_id", type=int, default=0)
    ap.add_argument("--infer_every", type=int, default=2)
    ap.add_argument("--smooth_k", type=int, default=5)
    ap.add_argument("--threshold", type=float, default=0.5)
    args = ap.parse_args()

    if not args.checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint no encontrado: {args.checkpoint}")
    if not args.feature_spec.exists():
        raise FileNotFoundError(f"Feature spec no encontrado: {args.feature_spec}")
    if not args.label_map.exists():
        raise FileNotFoundError(f"Label map no encontrado: {args.label_map}")

    spec = FeatureSpec.from_json(args.feature_spec)
    label_map = json.loads(args.label_map.read_text(encoding="utf-8"))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = torch.load(args.checkpoint, map_location=device)
    model_name = ckpt.get("model_name", "tcn")
    model = build_model(model_name, input_dim=spec.d_frame, num_classes=300).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    cap = cv2.VideoCapture(args.camera_id)
    if not cap.isOpened():
        raise RuntimeError("No se pudo abrir la webcam.")

    mp_pose = mp.solutions.pose
    mp_hands = mp.solutions.hands
    pose = mp_pose.Pose(static_image_mode=False, model_complexity=0, min_detection_confidence=0.5, min_tracking_confidence=0.5)
    hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5, min_tracking_confidence=0.5)

    buffer = deque(maxlen=spec.t_fixed)
    logits_hist = deque(maxlen=args.smooth_k)
    frame_idx = 0
    last_t = time.time()
    fps = 0.0

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pose_res = pose.process(rgb)
            hands_res = hands.process(rgb)

            feat = extract_frame_features(mp_results_to_frame(pose_res, hands_res))
            buffer.append(feat)
            frame_idx += 1

            pred_text = "Calibrando…"
            conf = 0.0

            if len(buffer) == spec.t_fixed and frame_idx % args.infer_every == 0:
                x = np.stack(buffer, axis=0).astype(np.float32)
                xt = torch.from_numpy(x).unsqueeze(0).to(device)
                with torch.no_grad():
                    logits = model(xt)
                    logits_hist.append(logits.cpu().numpy())

                avg_logits = np.mean(np.concatenate(list(logits_hist), axis=0), axis=0)
                probs = torch.softmax(torch.tensor(avg_logits), dim=0).numpy()
                pred_id = int(np.argmax(probs))
                conf = float(probs[pred_id])
                if conf < args.threshold:
                    pred_text = "Desconocido"
                else:
                    pred_text = label_map.get(str(pred_id), f"CLASS_{pred_id}")

            now = time.time()
            dt = now - last_t
            if dt > 0:
                fps = 0.9 * fps + 0.1 * (1.0 / dt)
            last_t = now

            cv2.putText(frame, f"Pred: {pred_text}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
            cv2.putText(frame, f"Conf: {conf:.2f}", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 0), 2)
            cv2.putText(frame, f"FPS: {fps:.1f}", (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
            cv2.imshow("LSE isolated sign recognition", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == 27 or key == ord("q"):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()
        pose.close()
        hands.close()


if __name__ == "__main__":
    main()

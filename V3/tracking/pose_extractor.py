"""
PoseExtractor — wraps MediaPipe PoseLandmarker to return all 33 landmarks
plus the V2-compatible 8-element signature and ready-state metrics.
"""

import math
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks.python import vision as mp_vision
from mediapipe.tasks.python.vision import PoseLandmarker, PoseLandmarkerOptions
from mediapipe.tasks.python.core.base_options import BaseOptions

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config as C
from shared.io_utils import expand_bbox, resize_keep_aspect


# ── angle helper ─────────────────────────────────────────────────────────────
def angle_deg(a: Tuple[float, float], b: Tuple[float, float], c: Tuple[float, float]) -> float:
    v1 = np.array([a[0] - b[0], a[1] - b[1]])
    v2 = np.array([c[0] - b[0], c[1] - b[1]])
    n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
    if n1 < 1e-6 or n2 < 1e-6:
        return 180.0
    cosang = np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0)
    return float(np.degrees(np.arccos(cosang)))


# ── fake-result wrapper (MediaPipe tasks API quirk) ───────────────────────────
class _FakeLandmarks:
    def __init__(self, lms):
        self.landmark = lms

class _FakePoseResult:
    def __init__(self, lms):
        self.pose_landmarks = _FakeLandmarks(lms)


class PoseExtractor:
    """
    Manages a pool of MediaPipe PoseLandmarker instances and exposes methods
    to extract full 33-landmark data plus V2-compatible features.
    """

    def __init__(self, model_path: str, pool_size: int = 4):
        options = PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=model_path),
            running_mode=mp_vision.RunningMode.IMAGE,
            num_poses=1,
            min_pose_detection_confidence=0.5,
            min_pose_presence_confidence=0.5,
            min_tracking_confidence=0.5,
            output_segmentation_masks=False,
        )
        self._pool = [PoseLandmarker.create_from_options(options) for _ in range(pool_size)]
        self._executor = ThreadPoolExecutor(max_workers=pool_size)
        self._pool_size = pool_size

    def close(self):
        self._executor.shutdown(wait=False)
        for lm in self._pool:
            lm.close()

    # ── per-crop inference ────────────────────────────────────────────────────
    def _run_one(self, pool_idx: int, crop_bgr: np.ndarray):
        """Run pose on one crop; return _FakePoseResult or None."""
        lm = self._pool[pool_idx % self._pool_size]
        mp_img = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB),
        )
        result = lm.detect(mp_img)
        if result.pose_landmarks and len(result.pose_landmarks) > 0:
            return _FakePoseResult(result.pose_landmarks[0])
        return None

    def submit_batch(self, crops_with_idx: List[Tuple[int, np.ndarray]]):
        """
        Submit a list of (det_i, crop_bgr) to the thread pool.
        Returns a list of futures in the same order.
        """
        futures = []
        for det_i, crop in crops_with_idx:
            fut = self._executor.submit(self._run_one, det_i, crop.copy())
            futures.append(fut)
        return futures

    # ── feature extraction ────────────────────────────────────────────────────
    @staticmethod
    def extract_signature(pose_res) -> Optional[np.ndarray]:
        """Return V2-compatible 8-element normalised pose vector or None."""
        if pose_res is None or pose_res.pose_landmarks is None:
            return None
        lm = pose_res.pose_landmarks.landmark
        ids = [11, 12, 23, 24, 25, 26, 27, 28]
        pts = [(float(lm[i].x), float(lm[i].y), float(lm[i].visibility)) for i in ids]
        if min(p[2] for p in pts) < C.POSE_MIN_VIS:
            return None
        ls, rs, lh, rh, lk, rk, la, ra = pts
        shoulder_mid = np.array([(ls[0] + rs[0]) * 0.5, (ls[1] + rs[1]) * 0.5])
        hip_mid      = np.array([(lh[0] + rh[0]) * 0.5, (lh[1] + rh[1]) * 0.5])
        ankle_mid    = np.array([(la[0] + ra[0]) * 0.5, (la[1] + ra[1]) * 0.5])
        torso_len = float(np.linalg.norm(shoulder_mid - hip_mid))
        leg_len   = float(np.linalg.norm(hip_mid - ankle_mid))
        body_len  = max(1e-6, torso_len + leg_len)
        shoulder_w = float(np.linalg.norm(np.array(ls[:2]) - np.array(rs[:2]))) / body_len
        hip_w      = float(np.linalg.norm(np.array(lh[:2]) - np.array(rh[:2]))) / body_len
        knee_l = angle_deg((lh[0], lh[1]), (lk[0], lk[1]), (la[0], la[1]))
        knee_r = angle_deg((rh[0], rh[1]), (rk[0], rk[1]), (ra[0], ra[1]))
        lean        = float(np.clip((hip_mid[0] - shoulder_mid[0]) / body_len, -1.0, 1.0))
        torso_tilt  = float(np.clip((hip_mid[1] - shoulder_mid[1]) / body_len, -1.0, 1.0))
        feat = np.array([shoulder_w, hip_w, torso_len / body_len, leg_len / body_len,
                         knee_l / 180.0, knee_r / 180.0, lean, torso_tilt], dtype=np.float32)
        n = float(np.linalg.norm(feat))
        if n < 1e-8:
            return None
        return (feat / n).astype(np.float32)

    @staticmethod
    def extract_ready_state(pose_res) -> Tuple:
        """Return (knee_L, knee_R, knee_avg, hip_drop, ready_flag) or (None×4, 0)."""
        if pose_res is None or pose_res.pose_landmarks is None:
            return None, None, None, None, 0
        lm = pose_res.pose_landmarks.landmark
        pts = [(lm[i].x, lm[i].y, lm[i].visibility) for i in [23, 24, 25, 26, 27, 28]]
        if any(p[2] <= C.POSE_MIN_VIS for p in pts):
            return None, None, None, None, 0
        Lhip, Rhip, Lknee, Rknee, Lank, Rank = pts
        knee_L = angle_deg((Lhip[0], Lhip[1]), (Lknee[0], Lknee[1]), (Lank[0], Lank[1]))
        knee_R = angle_deg((Rhip[0], Rhip[1]), (Rknee[0], Rknee[1]), (Rank[0], Rank[1]))
        knee_avg = 0.5 * (knee_L + knee_R)
        hip_y   = 0.5 * (Lhip[1] + Rhip[1])
        ank_y   = 0.5 * (Lank[1] + Rank[1])
        hip_drop = float(np.clip(1.0 - np.clip(ank_y - hip_y, 0.0, 1.0), 0.0, 1.0))
        ready = 1 if knee_avg <= C.READY_KNEE_ANGLE_MAX and hip_drop >= C.READY_HIP_DROP_MIN else 0
        return float(knee_L), float(knee_R), float(knee_avg), float(hip_drop), int(ready)

    @staticmethod
    def landmarks_to_row(pose_res, stable_id: int, frame_idx: int,
                         bbox_x1: float, bbox_y1: float, bbox_x2: float, bbox_y2: float) -> dict:
        """
        Serialise all 33 landmarks plus pixel-space joint positions.
        Returns a dict ready to merge into the detection row.
        Normalised coords in [0,1] relative to the crop; pixel coords are absolute frame pixels.
        """
        row: dict = {}
        crop_w = max(1.0, bbox_x2 - bbox_x1)
        crop_h = max(1.0, bbox_y2 - bbox_y1)

        if pose_res is None or pose_res.pose_landmarks is None:
            for i in range(33):
                row[f"lm{i}_x"]   = float("nan")
                row[f"lm{i}_y"]   = float("nan")
                row[f"lm{i}_vis"] = 0.0
            for col in [
                "shoulder_L_px_x", "shoulder_L_px_y",
                "shoulder_R_px_x", "shoulder_R_px_y",
                "elbow_L_px_x",    "elbow_L_px_y",
                "elbow_R_px_x",    "elbow_R_px_y",
                "wrist_L_px_x",    "wrist_L_px_y",
                "wrist_R_px_x",    "wrist_R_px_y",
                "hip_L_px_x",      "hip_L_px_y",
                "hip_R_px_x",      "hip_R_px_y",
            ]:
                row[col] = float("nan")
            return row

        lm = pose_res.pose_landmarks.landmark
        for i in range(33):
            row[f"lm{i}_x"]   = float(lm[i].x)
            row[f"lm{i}_y"]   = float(lm[i].y)
            row[f"lm{i}_vis"] = float(lm[i].visibility)

        # pixel-space positions: lm.x/y are normalised to crop, need to map back to frame
        def lm_px(idx):
            return (
                bbox_x1 + float(lm[idx].x) * crop_w,
                bbox_y1 + float(lm[idx].y) * crop_h,
            )

        named = {
            "shoulder_L": 11, "shoulder_R": 12,
            "elbow_L":    13, "elbow_R":    14,
            "wrist_L":    15, "wrist_R":    16,
            "hip_L":      23, "hip_R":      24,
        }
        for name, idx in named.items():
            px_x, px_y = lm_px(idx)
            row[f"{name}_px_x"] = float(px_x)
            row[f"{name}_px_y"] = float(px_y)
        return row

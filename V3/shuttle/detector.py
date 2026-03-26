"""
ShuttleDetector — Pass 2 of the V3 pipeline.

Ports ShuttleTrackerPx + Kalman2D from Old/ball tracking not working proper.py,
adds DirectionChangeDetector and meter-space coordinates, and runs as a separate
video pass (reads player bboxes from skeleton.parquet by frame_idx).
"""

import math
import os
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config as C
from shared.calibration import px_to_meters, COURT_W, COURT_L
from shared.io_utils import ensure_dir
from shuttle.schemas import SHUTTLE_COLS


# ── Kalman filter ─────────────────────────────────────────────────────────────
class Kalman2D:
    """Constant-velocity Kalman in pixels: state [x,y,vx,vy], measurement [x,y]."""

    def __init__(self):
        self.kf = cv2.KalmanFilter(4, 2)
        self.kf.transitionMatrix  = np.array(
            [[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=np.float32)
        self.kf.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], dtype=np.float32)
        self.kf.processNoiseCov   = np.eye(4, dtype=np.float32) * 1e-2
        self.kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * 10.0
        self.kf.errorCovPost      = np.eye(4, dtype=np.float32) * 800.0
        self.inited = False

    def init(self, x: float, y: float):
        self.kf.statePost = np.array([[x], [y], [0.0], [0.0]], dtype=np.float32)
        self.inited = True

    def predict(self) -> Tuple[float, float]:
        p = self.kf.predict()
        return float(p[0, 0]), float(p[1, 0])

    def update(self, x: float, y: float) -> Tuple[float, float]:
        m = np.array([[x], [y]], dtype=np.float32)
        s = self.kf.correct(m)
        return float(s[0, 0]), float(s[1, 0])


# ── direction change detector ─────────────────────────────────────────────────
class DirectionChangeDetector:
    """
    Detects sharp direction changes in shuttle velocity history.
    When the dot product of consecutive velocity vectors drops below
    SHUTTLE_DIR_DOT_THRESH the shuttle has changed direction (a shot occurred).
    """

    def __init__(self, history_len: int = None, dot_thresh: float = None):
        self._history_len = history_len or C.SHUTTLE_DIRECTION_HISTORY
        self._dot_thresh  = dot_thresh  or C.SHUTTLE_DIR_DOT_THRESH
        self._positions: List[Tuple[float, float]] = []

    def update(self, x: float, y: float) -> int:
        """Add position; return 1 if direction changed, else 0."""
        self._positions.append((x, y))
        if len(self._positions) > self._history_len + 2:
            self._positions = self._positions[-(self._history_len + 2):]
        if len(self._positions) < 3:
            return 0
        # velocity vectors from last two pairs
        x2, y2 = self._positions[-1]
        x1, y1 = self._positions[-2]
        x0, y0 = self._positions[-3]
        vx_a, vy_a = x1 - x0, y1 - y0
        vx_b, vy_b = x2 - x1, y2 - y1
        len_a = math.hypot(vx_a, vy_a)
        len_b = math.hypot(vx_b, vy_b)
        if len_a < 1.0 or len_b < 1.0:
            return 0
        dot = (vx_a * vx_b + vy_a * vy_b) / (len_a * len_b)
        return 1 if dot < self._dot_thresh else 0

    def reset(self):
        self._positions.clear()


# ── candidate detection helpers ───────────────────────────────────────────────
def build_allowed_mask(
    shape_hw: Tuple[int, int],
    player_bboxes: List[Tuple[float, float, float, float]],
) -> np.ndarray:
    """Return 255=allowed mask with player regions zeroed out."""
    h, w = shape_hw
    allowed = np.ones((h, w), dtype=np.uint8) * 255
    if not C.SHUTTLE_MASK_PLAYERS:
        return allowed
    for x1, y1, x2, y2 in player_bboxes:
        x1p = int(max(0, math.floor(x1) - C.SHUTTLE_PLAYER_PAD_PX))
        y1p = int(max(0, math.floor(y1) - C.SHUTTLE_PLAYER_PAD_PX))
        x2p = int(min(w - 1, math.ceil(x2) + C.SHUTTLE_PLAYER_PAD_PX))
        y2p = int(min(h - 1, math.ceil(y2) + C.SHUTTLE_PLAYER_PAD_PX))
        cv2.rectangle(allowed, (x1p, y1p), (x2p, y2p), 0, -1)
    return allowed


def detect_shuttle_candidates(
    prev_gray: np.ndarray,
    gray: np.ndarray,
    allowed_mask: Optional[np.ndarray],
) -> List[Tuple[float, float, float]]:
    """Return list of (cx, cy, score) candidates."""
    diff = cv2.absdiff(gray, prev_gray)
    _, mask = cv2.threshold(diff, C.SHUTTLE_DIFF_THRESH, 255, cv2.THRESH_BINARY)
    if allowed_mask is not None:
        mask = cv2.bitwise_and(mask, allowed_mask)
    if C.SHUTTLE_ERODE_ITERS > 0:
        mask = cv2.erode(mask, None, iterations=C.SHUTTLE_ERODE_ITERS)
    if C.SHUTTLE_DILATE_ITERS > 0:
        mask = cv2.dilate(mask, None, iterations=C.SHUTTLE_DILATE_ITERS)
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cands = []
    for cnt in cnts:
        area = cv2.contourArea(cnt)
        if area < C.SHUTTLE_MIN_AREA or area > C.SHUTTLE_MAX_AREA:
            continue
        x, y, w, h = cv2.boundingRect(cnt)
        if w > C.SHUTTLE_MAX_WH or h > C.SHUTTLE_MAX_WH:
            continue
        aspect = max(w / max(1, h), h / max(1, w))
        if aspect > C.SHUTTLE_MAX_ASPECT:
            continue
        cx = x + w / 2.0
        cy = y + h / 2.0
        patch = gray[max(0, y - 1):min(gray.shape[0], y + h + 1),
                     max(0, x - 1):min(gray.shape[1], x + w + 1)]
        if patch.size == 0:
            continue
        bright = float(np.percentile(patch, 92))
        if bright < C.SHUTTLE_BRIGHT_MIN:
            continue
        score = (bright / 255.0) * (1.0 / (1.0 + 0.02 * area))
        cands.append((cx, cy, score))
    return cands


# ── single-pass tracker ───────────────────────────────────────────────────────
class ShuttleTrackerPx:
    """Per-frame shuttle tracker (pixel space)."""

    def __init__(self):
        self.kf        = Kalman2D()
        self.prev_gray: Optional[np.ndarray] = None
        self.misses    = 0
        self.trail: List[Tuple[int, int]] = []

    def reset(self):
        self.kf        = Kalman2D()
        self.prev_gray = None
        self.misses    = 0
        self.trail.clear()

    def update(
        self,
        frame: np.ndarray,
        player_bboxes: List[Tuple[float, float, float, float]],
    ) -> dict:
        out = {
            "shuttle_px_x": float("nan"),
            "shuttle_px_y": float("nan"),
            "shuttle_visible": 0,
            "shuttle_conf":    0.0,
        }

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if self.prev_gray is None:
            self.prev_gray = gray
            return out

        allowed = build_allowed_mask(gray.shape, player_bboxes)

        pred = None
        if self.kf.inited:
            pred = self.kf.predict()
            px, py = int(round(pred[0])), int(round(pred[1]))
            if 0 <= px < gray.shape[1] and 0 <= py < gray.shape[0]:
                cv2.circle(allowed, (px, py), C.SHUTTLE_UNMASK_PRED_RADIUS, 255, -1)

        cands = detect_shuttle_candidates(self.prev_gray, gray, allowed)

        best = None
        best_score = -1e9
        for cx, cy, base in cands:
            if pred is not None:
                d = float(np.hypot(cx - pred[0], cy - pred[1]))
                if d > C.SHUTTLE_GATE_PX:
                    continue
                score = base * (1.0 / (1.0 + 0.03 * d))
            else:
                score = base
            if score > best_score:
                best_score = score
                best = (cx, cy, score)

        if best is not None:
            cx, cy, _ = best
            conf = float(np.clip(best_score * 2.0, 0.0, 1.0))
            self.misses = 0
            if not self.kf.inited:
                self.kf.init(cx, cy)
                sx, sy = cx, cy
            else:
                sx, sy = self.kf.update(cx, cy)
            out.update({"shuttle_px_x": float(sx), "shuttle_px_y": float(sy),
                        "shuttle_visible": 1, "shuttle_conf": conf})
            self.trail.append((int(round(sx)), int(round(sy))))
            self.trail = self.trail[-C.SHUTTLE_TRAIL_LEN:]
        else:
            if self.kf.inited:
                self.misses += 1
                px, py = self.kf.predict()
                out.update({"shuttle_px_x": float(px), "shuttle_px_y": float(py),
                            "shuttle_visible": 0, "shuttle_conf": 0.0})
                if self.misses > C.SHUTTLE_MAX_MISSES:
                    self.reset()

        self.prev_gray = gray
        return out


# ── orchestrating class ───────────────────────────────────────────────────────
class ShuttleDetector:
    """
    Pass 2: reads skeleton.parquet for player bboxes per frame,
    replays the video to run ShuttleTrackerPx, writes shuttle.parquet.
    """

    def run(self, video_path: str, skeleton_path: str, H: np.ndarray,
            output_dir: str) -> str:
        """Returns path to shuttle.parquet."""
        ensure_dir(output_dir)

        # load player bbox lookup: frame_idx → list of (x1,y1,x2,y2)
        skel_df = pd.read_parquet(skeleton_path,
                                   columns=["frame_idx", "stable_id",
                                            "bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2",
                                            "x_m", "y_m", "foot_px_x", "foot_px_y"])
        bbox_by_frame: Dict[int, List] = {}
        pos_by_frame:  Dict[int, List] = {}
        for row in skel_df.itertuples(index=False):
            fi = int(row.frame_idx)
            if fi not in bbox_by_frame:
                bbox_by_frame[fi] = []
                pos_by_frame[fi]  = []
            bbox_by_frame[fi].append((float(row.bbox_x1), float(row.bbox_y1),
                                       float(row.bbox_x2), float(row.bbox_y2)))
            pos_by_frame[fi].append((int(row.stable_id), float(row.foot_px_x), float(row.foot_px_y),
                                      float(row.x_m), float(row.y_m)))

        cap = cv2.VideoCapture(video_path)
        fps = float(cap.get(cv2.CAP_PROP_FPS)) or C.SAVE_FPS_FALLBACK
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        tracker   = ShuttleTrackerPx()
        dir_det   = DirectionChangeDetector()
        prev_sx = prev_sy = float("nan")
        rows: List[dict] = []
        frame_idx = -1

        print(f"V3 — Pass 2 shuttle detection ({total} frames)...")

        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                break
            frame_idx  += 1
            timestamp_s = frame_idx / fps

            bboxes = bbox_by_frame.get(frame_idx, [])
            result = tracker.update(frame, bboxes)

            sx = result["shuttle_px_x"]
            sy = result["shuttle_px_y"]
            visible = result["shuttle_visible"]

            # metre-space coords
            if math.isfinite(sx) and math.isfinite(sy):
                shuttle_x_m, shuttle_y_m = px_to_meters(H, sx, sy)
                shuttle_x_m = float(np.clip(shuttle_x_m, -1.0, COURT_W + 1.0))
                shuttle_y_m = float(np.clip(shuttle_y_m, -1.0, COURT_L + 1.0))
            else:
                shuttle_x_m = shuttle_y_m = float("nan")

            # direction change
            dir_change = 0
            if visible == 1 and math.isfinite(sx):
                dir_change = dir_det.update(sx, sy)
            elif not tracker.kf.inited:
                dir_det.reset()

            # shuttle speed (pixels → approximate m/s)
            shuttle_speed = float("nan")
            if visible == 1 and math.isfinite(sx) and math.isfinite(prev_sx):
                dpx = math.hypot(sx - prev_sx, sy - prev_sy)
                shuttle_speed = dpx / max(1.0 / fps, 1e-6)  # px/s (rough)
            prev_sx, prev_sy = sx, sy

            # nearest player
            nearest_id   = -1
            nearest_dist = float("nan")
            players = pos_by_frame.get(frame_idx, [])
            if players and math.isfinite(sx):
                best_d = 1e9
                for sid, fpx, fpy, _xm, _ym in players:
                    d = math.hypot(sx - fpx, sy - fpy)
                    if d < best_d:
                        best_d   = d
                        nearest_id   = sid
                        nearest_dist = d

            rows.append({
                "frame_idx":             frame_idx,
                "timestamp_s":           float(timestamp_s),
                "shuttle_px_x":          float(sx),
                "shuttle_px_y":          float(sy),
                "shuttle_x_m":           float(shuttle_x_m),
                "shuttle_y_m":           float(shuttle_y_m),
                "shuttle_visible":       int(visible),
                "shuttle_conf":          float(result["shuttle_conf"]),
                "direction_change_flag": int(dir_change),
                "shuttle_speed_mps":     float(shuttle_speed),
                "nearest_player_id":     int(nearest_id),
                "nearest_player_dist_m": float(nearest_dist),
            })

            if frame_idx % 500 == 0:
                print(f"  shuttle frame {frame_idx}/{total}")

        cap.release()

        shuttle_path = os.path.join(output_dir, "shuttle.parquet")
        df = pd.DataFrame(rows, columns=SHUTTLE_COLS)
        df.to_parquet(shuttle_path, engine="pyarrow", compression="snappy", index=False)
        print(f"V3 — shuttle.parquet written: {len(df)} rows")
        return shuttle_path

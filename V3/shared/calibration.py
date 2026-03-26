"""
Calibration utilities — ported from V2/badminton_analyzer.py.
Handles loading / saving court homography JSON files and coordinate transforms.
"""

import os
import json
from datetime import datetime
from typing import Optional, Tuple, List

import cv2
import numpy as np

import tkinter as tk
from tkinter import filedialog

# ── court geometry (regulation doubles) ──────────────────────────────────────
COURT_W = 6.10   # metres, left→right
COURT_L = 13.40  # metres, near baseline→far baseline

COURT_DST = np.array(
    [[0.0, 0.0], [COURT_W, 0.0], [COURT_W, COURT_L], [0.0, COURT_L]],
    dtype=np.float32,
)

# ── click-state (module-level so callbacks can write to it) ──────────────────
_clicked_points: List[List[int]] = []


# ── coordinate transforms ─────────────────────────────────────────────────────
def px_to_meters(H: np.ndarray, x_px: float, y_px: float) -> Tuple[float, float]:
    pt = np.array([[[x_px, y_px]]], dtype=np.float32)
    out = cv2.perspectiveTransform(pt, H.astype(np.float64))[0, 0]
    return float(out[0]), float(out[1])


def meters_to_px(H: np.ndarray, x_m: float, y_m: float) -> Tuple[int, int]:
    Hinv = np.linalg.inv(H)
    pt = np.array([[[x_m, y_m]]], dtype=np.float32)
    out = cv2.perspectiveTransform(pt, Hinv.astype(np.float64))[0, 0]
    return int(round(out[0])), int(round(out[1]))


# ── persistence ───────────────────────────────────────────────────────────────
def save_calibration(path: str, setup_name: str, points: List[List[int]], H: np.ndarray) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    data = {
        "setup_name": setup_name,
        "court_w_m": COURT_W,
        "court_l_m": COURT_L,
        "dst_points_m": COURT_DST.tolist(),
        "clicked_points_px": points,
        "homography_px_to_m": H.tolist(),
        "saved_at": datetime.now().isoformat(),
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    print(f"Saved calibration: {path}")


def load_calibration(path: str) -> Optional[Tuple[List[List[int]], np.ndarray]]:
    """Return (clicked_points_px, H) or None if file missing/invalid."""
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        pts = data.get("clicked_points_px")
        H_list = data.get("homography_px_to_m")
        if not (isinstance(pts, list) and len(pts) == 4):
            return None
        H = np.array(H_list, dtype=np.float64)
        if H.shape != (3, 3):
            return None
        print(f"Loaded calibration: {path}")
        return pts, H
    except Exception:
        return None


# ── interactive corner clicking ───────────────────────────────────────────────
def _corner_mouse_cb(event, x, y, flags, param):
    global _clicked_points
    if event == cv2.EVENT_LBUTTONDOWN:
        _clicked_points.append([x, y])


def click_corners(frame0: np.ndarray) -> List[List[int]]:
    """Show frame, let user click 4 court corners (TL, TR, BR, BL)."""
    global _clicked_points
    _clicked_points = []
    clone = frame0.copy()
    cv2.namedWindow("Click court corners", cv2.WINDOW_NORMAL)
    cv2.setMouseCallback("Click court corners", _corner_mouse_cb)
    while True:
        vis = clone.copy()
        for i, p in enumerate(_clicked_points):
            cv2.circle(vis, tuple(p), 7, (0, 255, 0), -1)
            cv2.putText(vis, str(i + 1), (p[0] + 10, p[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.putText(vis, "Click 4 corners: TL, TR, BR, BL | R=reset | ESC/Q=quit",
                    (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.imshow("Click court corners", vis)
        key = cv2.waitKey(10) & 0xFF
        if key in (27, ord("q"), ord("Q")):
            cv2.destroyAllWindows()
            raise SystemExit("User quit during calibration.")
        if key in (ord("r"), ord("R")):
            _clicked_points = []
        if len(_clicked_points) == 4:
            break
    cv2.destroyWindow("Click court corners")
    return _clicked_points


# ── player bootstrapping ──────────────────────────────────────────────────────
_clicked_player_points: List[List[int]] = []


def _player_mouse_cb(event, x, y, flags, param):
    global _clicked_player_points
    if event == cv2.EVENT_LBUTTONDOWN:
        _clicked_player_points.append([x, y])


def click_players(frame0: np.ndarray, num_players: int = 4) -> List[List[int]]:
    """Show frame, let user click each player to initialise stable IDs."""
    global _clicked_player_points
    _clicked_player_points = []
    clone = frame0.copy()
    cv2.namedWindow("Click players", cv2.WINDOW_NORMAL)
    cv2.setMouseCallback("Click players", _player_mouse_cb)
    while True:
        vis = clone.copy()
        for i, p in enumerate(_clicked_player_points):
            cv2.circle(vis, tuple(p), 9, (0, 255, 0), -1)
            cv2.putText(vis, f"P{i + 1}", (p[0] + 12, p[1] - 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.putText(vis, f"Click {num_players} players | R=reset | ESC/Q=quit",
                    (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.imshow("Click players", vis)
        key = cv2.waitKey(10) & 0xFF
        if key in (27, ord("q"), ord("Q")):
            cv2.destroyAllWindows()
            raise SystemExit("User quit during player init.")
        if key in (ord("r"), ord("R")):
            _clicked_player_points = []
        if len(_clicked_player_points) == num_players:
            break
    cv2.destroyWindow("Click players")
    return _clicked_player_points


# ── video file picker ─────────────────────────────────────────────────────────
def select_video() -> str:
    root = tk.Tk()
    root.withdraw()
    root.wm_attributes("-topmost", 1)
    file_path = filedialog.askopenfilename(
        title="Select Badminton Video",
        filetypes=[
            ("Video files", "*.mp4 *.avi *.mov *.mkv *.MOV"),
            ("All files", "*.*"),
        ],
    )
    if not file_path:
        raise RuntimeError("No video file selected.")
    print(f"Selected video: {file_path}")
    return file_path

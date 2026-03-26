"""
Drawing helpers — ported from V2/badminton_analyzer.py.
"""

import math
from typing import Optional, Tuple

import cv2
import numpy as np


def draw_rounded_rect(
    img: np.ndarray,
    x1: float, y1: float, x2: float, y2: float,
    color: Tuple[int, int, int],
    thickness: int = -1,
    radius: int = 10,
) -> None:
    x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
    radius = max(0, min(radius, (x2 - x1) // 2, (y2 - y1) // 2))
    if thickness < 0:
        cv2.rectangle(img, (x1 + radius, y1), (x2 - radius, y2), color, -1)
        cv2.rectangle(img, (x1, y1 + radius), (x2, y2 - radius), color, -1)
        cv2.circle(img, (x1 + radius, y1 + radius), radius, color, -1)
        cv2.circle(img, (x2 - radius, y1 + radius), radius, color, -1)
        cv2.circle(img, (x1 + radius, y2 - radius), radius, color, -1)
        cv2.circle(img, (x2 - radius, y2 - radius), radius, color, -1)
    else:
        cv2.rectangle(img, (x1 + radius, y1), (x2 - radius, y2), color, thickness)
        cv2.rectangle(img, (x1, y1 + radius), (x2, y2 - radius), color, thickness)
        cv2.circle(img, (x1 + radius, y1 + radius), radius, color, thickness)
        cv2.circle(img, (x2 - radius, y1 + radius), radius, color, thickness)
        cv2.circle(img, (x1 + radius, y2 - radius), radius, color, thickness)
        cv2.circle(img, (x2 - radius, y2 - radius), radius, color, thickness)


def overlay_alpha(base_img: np.ndarray, overlay_img: np.ndarray, alpha: float) -> None:
    cv2.addWeighted(overlay_img, alpha, base_img, 1 - alpha, 0, base_img)


def draw_open_ellipse_arc(
    img: np.ndarray,
    center: Tuple[int, int],
    axes: Tuple[int, int],
    start_rad: float,
    end_rad: float,
    color: Tuple[int, int, int],
    thickness: int,
    samples: int = 64,
) -> None:
    cx, cy = center
    rx, ry = axes
    if end_rad < start_rad:
        start_rad, end_rad = end_rad, start_rad
    ts = np.linspace(start_rad, end_rad, max(8, samples))
    pts = np.stack([cx + rx * np.cos(ts), cy + ry * np.sin(ts)], axis=1).astype(np.int32)
    cv2.polylines(img, [pts.reshape(-1, 1, 2)], isClosed=False,
                  color=color, thickness=thickness, lineType=cv2.LINE_AA)


def knee_to_bgr(knee_angle_deg: Optional[float]) -> Tuple[int, int, int]:
    if knee_angle_deg is None:
        return (0, 255, 0)
    a = float(np.clip(knee_angle_deg, 80.0, 175.0))
    t = (175.0 - a) / (175.0 - 80.0)
    r = int(round(255 * t))
    g = int(round(255 * (1.0 - 0.25 * t)))
    return (0, g, r)


def draw_player_panel(
    frame: np.ndarray,
    bbox_xyxy: np.ndarray,
    stable_id: int,
    knee_angle_avg: Optional[float],
    ready_pct: Optional[float],
    ring_color: Tuple[int, int, int] = (0, 255, 0),
    ring_alpha: float = 0.45,
    panel_alpha: float = 0.28,
    ring_y_offset: int = 18,
    panel_w: int = 170,
    panel_h: int = 74,
) -> None:
    x1, y1, x2, y2 = bbox_xyxy.astype(int).tolist()
    w = max(1, x2 - x1)
    cx = int((x1 + x2) / 2)
    foot_y = int(y2)
    rx = int(w * 0.38)
    ry = int(w * 0.18)
    ring_y = foot_y - ring_y_offset

    overlay = frame.copy()
    shadow_center = (cx, ring_y + int(ry * 0.55))
    draw_open_ellipse_arc(overlay, shadow_center, (rx, ry), 0.0, math.pi, (0, 0, 0), 8, 64)
    overlay_alpha(frame, overlay, alpha=0.25)

    overlay = frame.copy()
    draw_open_ellipse_arc(overlay, (cx, ring_y), (rx, ry), 0.0, math.pi, ring_color, 3, 64)
    overlay_alpha(frame, overlay, alpha=ring_alpha)

    px1 = int(cx - panel_w / 2)
    py1 = int(foot_y + 10)
    px2 = px1 + panel_w
    py2 = py1 + panel_h
    H_img, W_img = frame.shape[:2]
    if py2 > H_img - 5:
        shift_y = (H_img - 5) - py2
        py1 += shift_y
        py2 += shift_y
    px1 = max(5, px1)
    px2 = min(W_img - 5, px2)

    overlay = frame.copy()
    draw_rounded_rect(overlay, px1 + 2, py1 + 2, px2 + 2, py2 + 2, (0, 0, 0), thickness=-1, radius=12)
    draw_rounded_rect(overlay, px1, py1, px2, py2, (255, 255, 255), thickness=-1, radius=12)
    overlay_alpha(frame, overlay, alpha=panel_alpha)

    cv2.putText(frame, f"{stable_id}", (px1 + 10, py1 + 26),
                cv2.FONT_HERSHEY_SIMPLEX, 0.95, (10, 10, 10), 3, cv2.LINE_AA)
    txt = f"Knee: {knee_angle_avg:.0f} deg" if knee_angle_avg is not None else "Knee: --"
    cv2.putText(frame, txt, (px1 + 10, py1 + 44),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (10, 10, 10), 2, cv2.LINE_AA)
    if ready_pct is not None:
        cv2.putText(frame, f"READY: {ready_pct:.0f}%", (px1 + 10, py1 + 62),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (10, 10, 10), 2, cv2.LINE_AA)

"""
I/O utilities — ported from V2/badminton_analyzer.py.
"""

import os
import math
from datetime import datetime
from typing import Tuple


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def now_stamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def make_output_paths(output_dir: str, session_id: str) -> dict:
    """Return a dict of all output file paths for a session."""
    return {
        "skeleton":     os.path.join(output_dir, "skeleton.parquet"),
        "shuttle":      os.path.join(output_dir, "shuttle.parquet"),
        "positioning":  os.path.join(output_dir, "positioning.parquet"),
        "reaction":     os.path.join(output_dir, "reaction.parquet"),
        "shots":        os.path.join(output_dir, "shots.parquet"),
        "meta":         os.path.join(output_dir, "session_meta.json"),
        "events":       os.path.join(output_dir, "events_v3.json"),
        "report":       os.path.join(output_dir, "report.html"),
        "video":        os.path.join(output_dir, "annotated.mp4"),
    }


def clamp01(v: float) -> float:
    return max(0.0, min(1.0, v))


def expand_bbox(
    x1: float, y1: float, x2: float, y2: float,
    w_img: int, h_img: int,
    frac: float = 0.12,
) -> Tuple[int, int, int, int]:
    w = x2 - x1
    h = y2 - y1
    ex = int(round(w * frac))
    ey = int(round(h * frac))
    return (
        max(0, int(x1) - ex),
        max(0, int(y1) - ey),
        min(w_img - 1, int(x2) + ex),
        min(h_img - 1, int(y2) + ey),
    )


def resize_keep_aspect(img, max_w: int, max_h: int):
    """Return (resized_img, scale) — never upscales."""
    import cv2
    h, w = img.shape[:2]
    scale = min(max_w / float(w), max_h / float(h), 1.0)
    if scale >= 0.999:
        return img, 1.0
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR), scale


def bbox_iou_xyxy(a: Tuple, b: Tuple) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    union = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
    return inter / union if union > 1e-8 else 0.0

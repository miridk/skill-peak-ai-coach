"""
Court geometry helpers — ported from V2/badminton_analyzer.py and metrics.py.
"""

from typing import Tuple

import cv2
import numpy as np

from .calibration import COURT_W, COURT_L, meters_to_px


# ── zone classification ───────────────────────────────────────────────────────
def classify_zone(x_m: float, y_m: float) -> str:
    """Return e.g. 'FAR-FRONT-LEFT', 'NEAR-BACK-RIGHT'."""
    mid = COURT_L / 2.0
    side_group = "FAR" if y_m < mid else "NEAR"
    depth = "FRONT" if abs(y_m - mid) <= 1.98 else "BACK"
    lr = "LEFT" if x_m < COURT_W / 2.0 else "RIGHT"
    return f"{side_group}-{depth}-{lr}"


def side_from_y(y_m: float) -> str:
    return "FAR" if y_m < COURT_L / 2.0 else "NEAR"


def depth_role_from_y(y_m: float) -> str:
    return "FRONT" if abs(y_m - COURT_L / 2.0) <= 1.98 else "BACK"


# ── court guides drawn on video frame ────────────────────────────────────────
def draw_court_guides(frame: np.ndarray, H: np.ndarray) -> None:
    mid_y = COURT_L / 2.0
    short = 1.98

    def lp(xa, ya, xb, yb):
        return meters_to_px(H, xa, ya), meters_to_px(H, xb, yb)

    p1, p2 = lp(0.0, mid_y, COURT_W, mid_y)
    cv2.line(frame, p1, p2, (0, 255, 255), 3, cv2.LINE_AA)

    p1s, p2s = lp(0.0, mid_y - short, COURT_W, mid_y - short)
    cv2.line(frame, p1s, p2s, (255, 255, 0), 2, cv2.LINE_AA)

    p1s2, p2s2 = lp(0.0, mid_y + short, COURT_W, mid_y + short)
    cv2.line(frame, p1s2, p2s2, (255, 255, 0), 2, cv2.LINE_AA)

    p1c, p2c = lp(COURT_W / 2.0, 0.0, COURT_W / 2.0, COURT_L)
    cv2.line(frame, p1c, p2c, (255, 0, 255), 2, cv2.LINE_AA)


# ── matplotlib court for heatmaps / reports ──────────────────────────────────
def draw_badminton_court_mpl(ax) -> None:
    """Draw full doubles court lines on a matplotlib Axes."""
    white = "#ffffff"
    lo, li = 4.0, 2.8
    W, L = COURT_W, COURT_L
    net_y = L / 2.0

    ax.plot([0, W, W, 0, 0], [0, 0, L, L, 0], color=white, linewidth=lo, zorder=6)
    ax.plot([0, W], [net_y, net_y], color=white, linewidth=lo, zorder=6)

    singles_w = 5.18
    margin = (W - singles_w) / 2.0
    ax.plot([margin, margin], [0, L], color=white, linewidth=li, zorder=6)
    ax.plot([W - margin, W - margin], [0, L], color=white, linewidth=li, zorder=6)

    short = 1.98
    ax.plot([0, W], [net_y - short, net_y - short], color=white, linewidth=li, zorder=6)
    ax.plot([0, W], [net_y + short, net_y + short], color=white, linewidth=li, zorder=6)

    cx = W / 2.0
    ax.plot([cx, cx], [net_y - short, net_y], color=white, linewidth=li, zorder=6)
    ax.plot([cx, cx], [net_y, net_y + short], color=white, linewidth=li, zorder=6)

    long_d = 0.76
    ax.plot([0, W], [long_d, long_d], color=white, linewidth=li, zorder=6)
    ax.plot([0, W], [L - long_d, L - long_d], color=white, linewidth=li, zorder=6)


# ── court polygon test ────────────────────────────────────────────────────────
def point_in_court_asymmetric_margin(
    pt: Tuple[float, float],
    court_pts: np.ndarray,
    top_margin: int,
    side_bottom_margin: int,
) -> bool:
    x, y = float(pt[0]), float(pt[1])
    poly = court_pts.reshape(-1, 2).astype(np.float32)
    if cv2.pointPolygonTest(court_pts, (x, y), False) >= 0:
        return True
    signed_dist = float(cv2.pointPolygonTest(court_pts, (x, y), True))
    top_y_min = min(float(p[1]) for p in poly)
    if y >= top_y_min and signed_dist >= -side_bottom_margin:
        return True
    return False


def point_to_polygon_signed_distance(pt: Tuple[float, float], poly: np.ndarray) -> float:
    return float(cv2.pointPolygonTest(poly, pt, True))

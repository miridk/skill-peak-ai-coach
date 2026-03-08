import os
import json
import csv
import math
import cv2
import numpy as np
import torch
import open_clip

from PIL import Image
from scipy.optimize import linear_sum_assignment
from datetime import datetime
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from ultralytics import YOLO
import supervision as sv
import mediapipe as mp


# ----------------------------
# CONFIG
# ----------------------------
VIDEO_PATH = "input.mp4"
SAVE_DIR = "save"
EXPORT_DIR = "exports"

COURT_SETUP_NAME = "almind_viuf_hallen_baseline2"
CALIB_DIR = "calibration"
CALIB_FILE = os.path.join(CALIB_DIR, f"{COURT_SETUP_NAME}.json")

SAVE_FPS_FALLBACK = 30.0

# Badminton court (meter)
COURT_W = 6.10
COURT_L = 13.40

# top-left, top-right, bottom-right, bottom-left
COURT_DST = np.array(
    [[0.0, 0.0], [COURT_W, 0.0], [COURT_W, COURT_L], [0.0, COURT_L]],
    dtype=np.float32,
)

# Detection
COURT_MARGIN_PX = 85         # top margin
SIDE_BOTTOM_MARGIN_PX = 2    # left, right and bottom margin
CONF = 0.45
IOU = 0.6
IMGSZ = 1080

# Logical players for doubles
PLAYER_IDS = [1, 2, 3, 4]
MAX_PLAYERS = 4

# Identity / CLIP
CLIP_MODEL_NAME = "ViT-B-32"
CLIP_PRETRAINED = "laion2b_s34b_b79k"
CLIP_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

MAX_AGE_SECONDS = 3.0
MAX_MATCH_DIST_M = 3.2

# Assignment weights
W_CLIP = 0.50
W_MOTION = 0.22
W_COLOR = 0.08
W_SIZE = 0.05
W_SIDE = 0.17
W_EDGE = 0.08

# Anti-switch
SWITCH_MARGIN = 0.14
SWITCH_CONFIRM_FRAMES = 8
UNMATCHED_COST = 1.15

# Stronger anti-swap pair logic
PAIR_SWAP_MAX_M = 1.10
PAIR_SWAP_IOU_THRESH = 0.02
PAIR_SWAP_COST_MARGIN = 0.18
PAIR_SWAP_HIT_MIN = 18
PAIR_SWAP_AMBIGUOUS_BONUS = 0.12

# Outsider rejection
MAX_ACCEPT_COST = 0.72
OUTSIDER_DIST_M = 1.75
OUTSIDER_CLIP_MIN_SIM = 0.58

# Embedding smoothing / memory
CLIP_EMA_ALPHA = 0.14
COLOR_EMA_ALPHA = 0.16
VEL_ALPHA = 0.40
SIZE_ALPHA = 0.25
MEMORY_BANK_SIZE = 12
MEMORY_MATCH_TOPK = 4

# Occlusion handling
OCCLUSION_IOU_THRESH = 0.18
OCCLUSION_EXTRA_STICKINESS = 0.16
OCCLUSION_MOTION_BOOST = 0.08
OCCLUSION_CLIP_REDUCE = 0.10
OCCLUSION_FREEZE_BANK_UPDATE = True

# Side plausibility
SIDE_PENALTY = 0.22
SIDE_GRACE_FRAMES = 20
HARD_SIDE_PENALTY = 0.38
HARD_SIDE_HITS_MIN = 20

# Line ambiguity / same-row handling
AMBIGUOUS_X_PX = 115
AMBIGUOUS_Y_PX = 265
AMBIGUOUS_IOU_THRESH = 0.03
AMBIGUOUS_EXTRA_SWITCH_MARGIN = 0.40
AMBIGUOUS_EXTRA_CONFIRM_FRAMES = 10
AMBIGUOUS_FREEZE_BANK_UPDATE = True
AMBIGUOUS_MAX_ACCEPT_COST = 0.42

# Live stats overlay settings
SPEED_SMOOTH_ALPHA = 0.35
SPEED_CAP_KMH = 45.0
DIST_JUMP_CAP_M = 2.5

# Acceleration overlay settings
ACCEL_SMOOTH_ALPHA = 0.65
ACCEL_CAP_MPS2 = 11.0
ACCEL_DEADZONE = 0.10

# Ring style
RING_COLOR = (0, 255, 0)
RING_ALPHA = 0.45
RING_THICKNESS = 3
RING_SHADOW_ALPHA = 0.25

# Panel transparency
PANEL_ALPHA = 0.28

# Panel layout
PANEL_WIDTH = 170
PANEL_HEIGHT = 74

# Ring position + open arc settings
RING_Y_OFFSET_PX = 18
RING_ARC_SAMPLES = 64

# Pose / ready
POSE_EVERY_N_FRAMES = 2
POSE_MIN_VIS = 0.25
POSE_BBOX_EXPAND = 0.18
POSE_CROP_MAX_W = 320
POSE_CROP_MAX_H = 320

READY_KNEE_ANGLE_MAX = 160.0
READY_HIP_DROP_MIN = 0.04
READY_MAX_SPEED_KMH = 6.0

READY_SMOOTH_ALPHA = 0.35
READY_HIP_SMOOTH_ALPHA = 0.35

clicked_points: list[list[int]] = []
clicked_player_points: list[list[int]] = []


# ----------------------------
# UTIL
# ----------------------------
def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def now_stamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def make_output_path(save_dir: str, input_path: str) -> str:
    base = os.path.splitext(os.path.basename(input_path))[0]
    return os.path.join(save_dir, f"{base}_tracked_{now_stamp()}.mp4")


def make_export_paths(export_dir: str, input_path: str) -> tuple[str, str]:
    base = os.path.splitext(os.path.basename(input_path))[0]
    stamp = now_stamp()
    return (
        os.path.join(export_dir, f"{base}_{stamp}.csv"),
        os.path.join(export_dir, f"{base}_{stamp}.jsonl"),
    )


def point_in_polygon_margin(pt, poly, margin_px=35) -> bool:
    return cv2.pointPolygonTest(poly, pt, True) >= -margin_px


def point_to_polygon_signed_distance(pt, poly) -> float:
    return float(cv2.pointPolygonTest(poly, pt, True))


def point_in_court_asymmetric_margin(
    pt: Tuple[float, float],
    court_pts: np.ndarray,
    top_margin: float,
    side_bottom_margin: float,
) -> bool:
    """
    Safer asymmetric margin:
    - true inside polygon
    - small margin on left/right/bottom using polygon signed distance
    - larger margin only above the top edge
    """
    x, y = float(pt[0]), float(pt[1])
    poly = court_pts.reshape(-1, 2).astype(np.float32)

    # Order expected: TL, TR, BR, BL
    tl, tr, br, bl = poly

    # 1) strictly inside polygon
    if cv2.pointPolygonTest(court_pts, (x, y), False) >= 0:
        return True

    # 2) small generic margin, but NOT above the top edge
    signed_dist = float(cv2.pointPolygonTest(court_pts, (x, y), True))

    top_y_min = min(float(tl[1]), float(tr[1]))

    # allow small margin only if point is not above the top edge area
    if y >= top_y_min and signed_dist >= -side_bottom_margin:
        return True

    # 3) special larger allowance only near the top edge
    x1, y1 = float(tl[0]), float(tl[1])
    x2, y2 = float(tr[0]), float(tr[1])

    dx = x2 - x1
    dy = y2 - y1
    seg_len2 = dx * dx + dy * dy
    if seg_len2 <= 1e-6:
        return False

    # projection onto top segment
    t = ((x - x1) * dx + (y - y1) * dy) / seg_len2

    # allow a little horizontal slack near the ends
    slack = side_bottom_margin / max(1.0, math.sqrt(seg_len2))
    if t < -slack or t > 1.0 + slack:
        return False

    t_clamped = max(0.0, min(1.0, t))
    proj_x = x1 + t_clamped * dx
    proj_y = y1 + t_clamped * dy

    # perpendicular distance to top edge
    perp_dist = math.hypot(x - proj_x, y - proj_y)

    # only allow points ABOVE the top edge
    line_y_at_x = proj_y
    is_above_top = y < line_y_at_x

    if is_above_top and perp_dist <= top_margin:
        return True

    return False


def px_to_meters(H: np.ndarray, x_px: float, y_px: float) -> tuple[float, float]:
    pt = np.array([[[x_px, y_px]]], dtype=np.float32)
    out = cv2.perspectiveTransform(pt, H)[0, 0]
    return float(out[0]), float(out[1])


def meters_to_px(H: np.ndarray, x_m: float, y_m: float) -> tuple[int, int]:
    Hinv = np.linalg.inv(H)
    pt = np.array([[[x_m, y_m]]], dtype=np.float32)
    out = cv2.perspectiveTransform(pt, Hinv)[0, 0]
    return int(round(out[0])), int(round(out[1]))


def clamp01(v: float) -> float:
    return float(max(0.0, min(1.0, v)))


def cosine_sim(a: Optional[np.ndarray], b: Optional[np.ndarray]) -> float:
    if a is None or b is None:
        return 0.0
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na < 1e-8 or nb < 1e-8:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def safe_hist_corr(a: Optional[np.ndarray], b: Optional[np.ndarray]) -> float:
    if a is None or b is None:
        return 0.0
    try:
        sim = float(cv2.compareHist(a.astype(np.float32), b.astype(np.float32), cv2.HISTCMP_CORREL))
        return clamp01((sim + 1.0) * 0.5)
    except Exception:
        return 0.0


def expand_bbox(x1, y1, x2, y2, w_img, h_img, frac=0.12):
    w = x2 - x1
    h = y2 - y1
    ex = int(round(w * frac))
    ey = int(round(h * frac))
    nx1 = max(0, int(x1) - ex)
    ny1 = max(0, int(y1) - ey)
    nx2 = min(w_img - 1, int(x2) + ex)
    ny2 = min(h_img - 1, int(y2) + ey)
    return nx1, ny1, nx2, ny2


def crop_person(frame: np.ndarray, xyxy: Tuple[float, float, float, float], frac: float = 0.04) -> Optional[np.ndarray]:
    h_img, w_img = frame.shape[:2]
    x1, y1, x2, y2 = expand_bbox(*xyxy, w_img, h_img, frac=frac)
    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        return None
    if crop.shape[0] < 12 or crop.shape[1] < 8:
        return None
    return crop


def extract_color_feature(frame: np.ndarray, box_xyxy: Tuple[float, float, float, float]) -> Optional[np.ndarray]:
    crop = crop_person(frame, box_xyxy, frac=0.02)
    if crop is None:
        return None

    h, w = crop.shape[:2]
    tx1 = int(w * 0.22)
    tx2 = int(w * 0.78)
    ty1 = int(h * 0.16)
    ty2 = int(h * 0.62)
    torso = crop[ty1:ty2, tx1:tx2]
    if torso.size == 0 or torso.shape[0] < 8 or torso.shape[1] < 8:
        return None

    hsv = cv2.cvtColor(torso, cv2.COLOR_BGR2HSV)
    _, s, v = cv2.split(hsv)
    mask = ((s > 28) & (v > 35) & (v < 245)).astype(np.uint8) * 255
    if int(mask.sum()) < 25:
        mask = None

    hist_hs = cv2.calcHist([hsv], [0, 1], mask, [18, 16], [0, 180, 0, 256])
    hist_v = cv2.calcHist([hsv], [2], mask, [16], [0, 256])
    hist_hs = cv2.normalize(hist_hs, hist_hs).flatten()
    hist_v = cv2.normalize(hist_v, hist_v).flatten()
    return np.concatenate([hist_hs, hist_v]).astype(np.float32)


def classify_zone(x_m: float, y_m: float) -> str:
    mid = COURT_L / 2.0
    side_group = "FAR" if y_m < mid else "NEAR"
    dist_to_net = abs(y_m - mid)
    depth = "FRONT" if dist_to_net <= 1.98 else "BACK"
    lr = "LEFT" if x_m < COURT_W / 2.0 else "RIGHT"
    return f"{side_group}-{depth}-{lr}"


def bbox_iou_xyxy(a: Tuple[float, float, float, float], b: Tuple[float, float, float, float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b

    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)

    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih

    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter

    if union <= 1e-8:
        return 0.0
    return float(inter / union)


def side_from_y(y_m: float) -> str:
    return "FAR" if y_m < (COURT_L / 2.0) else "NEAR"


def best_bank_similarity(bank: List[np.ndarray], feat: Optional[np.ndarray], topk: int = 4) -> float:
    if feat is None or not bank:
        return 0.0

    sims = []
    for b in bank:
        s = cosine_sim(b, feat)
        sims.append(clamp01((s + 1.0) * 0.5))

    if not sims:
        return 0.0

    sims.sort(reverse=True)
    sims = sims[:max(1, topk)]
    return float(sum(sims) / len(sims))


# ----------------------------
# DRAW HELPERS
# ----------------------------
def draw_court_guides(frame: np.ndarray, H: np.ndarray) -> None:
    mid_y = COURT_L / 2.0
    short = 1.98

    def line_pts(xa, ya, xb, yb):
        p1 = meters_to_px(H, xa, ya)
        p2 = meters_to_px(H, xb, yb)
        return p1, p2

    p1, p2 = line_pts(0.0, mid_y, COURT_W, mid_y)
    cv2.line(frame, p1, p2, (0, 255, 255), 3, cv2.LINE_AA)

    p1s, p2s = line_pts(0.0, mid_y - short, COURT_W, mid_y - short)
    cv2.line(frame, p1s, p2s, (255, 255, 0), 2, cv2.LINE_AA)

    p1s2, p2s2 = line_pts(0.0, mid_y + short, COURT_W, mid_y + short)
    cv2.line(frame, p1s2, p2s2, (255, 255, 0), 2, cv2.LINE_AA)

    p1c, p2c = line_pts(COURT_W / 2.0, 0.0, COURT_W / 2.0, COURT_L)
    cv2.line(frame, p1c, p2c, (255, 0, 255), 2, cv2.LINE_AA)

    cv2.putText(frame, "NET", (min(p1[0], p2[0]) + 8, min(p1[1], p2[1]) - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, cv2.LINE_AA)


def draw_rounded_rect(img, x1, y1, x2, y2, color, thickness=-1, radius=10):
    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
    radius = int(max(0, min(radius, (x2 - x1) // 2, (y2 - y1) // 2)))
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


def overlay_alpha(base_img, overlay_img, alpha: float):
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
):
    cx, cy = center
    rx, ry = axes
    if end_rad < start_rad:
        start_rad, end_rad = end_rad, start_rad
    ts = np.linspace(start_rad, end_rad, max(8, samples))
    pts = np.stack([cx + rx * np.cos(ts), cy + ry * np.sin(ts)], axis=1).astype(np.int32)
    pts = pts.reshape((-1, 1, 2))
    cv2.polylines(img, [pts], isClosed=False, color=color, thickness=thickness, lineType=cv2.LINE_AA)


def knee_to_bgr(knee_angle_deg: Optional[float]) -> Tuple[int, int, int]:
    if knee_angle_deg is None:
        return (0, 255, 0)
    a = float(np.clip(knee_angle_deg, 80.0, 175.0))
    t = (175.0 - a) / (175.0 - 80.0)
    r = int(round(255 * t))
    g = int(round(255 * (1.0 - 0.25 * t)))
    return (0, g, r)


def draw_player_overlay(
    frame: np.ndarray,
    bbox_xyxy: np.ndarray,
    stable_id: int,
    knee_angle_avg: Optional[float],
    ready_pct: Optional[float],
    ring_color: Tuple[int, int, int] = RING_COLOR,
    ring_alpha: float = RING_ALPHA,
):
    x1, y1, x2, y2 = bbox_xyxy.astype(int).tolist()
    w = max(1, x2 - x1)

    cx = int((x1 + x2) / 2)
    foot_y = int(y2)

    rx = int(w * 0.38)
    ry = int(w * 0.18)
    ring_y = foot_y - int(RING_Y_OFFSET_PX)

    overlay = frame.copy()
    shadow_center = (cx, ring_y + int(ry * 0.55))
    draw_open_ellipse_arc(overlay, shadow_center, (rx, ry), 0.0, math.pi, (0, 0, 0), 8, RING_ARC_SAMPLES)
    overlay_alpha(frame, overlay, alpha=RING_SHADOW_ALPHA)

    overlay = frame.copy()
    draw_open_ellipse_arc(overlay, (cx, ring_y), (rx, ry), 0.0, math.pi, ring_color, RING_THICKNESS, RING_ARC_SAMPLES)
    overlay_alpha(frame, overlay, alpha=ring_alpha)

    panel_w = PANEL_WIDTH
    panel_h = PANEL_HEIGHT
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
    overlay_alpha(frame, overlay, alpha=PANEL_ALPHA)

    cv2.putText(frame, f"{stable_id}", (px1 + 10, py1 + 26),
                cv2.FONT_HERSHEY_SIMPLEX, 0.95, (10, 10, 10), 3, cv2.LINE_AA)

    if knee_angle_avg is not None:
        cv2.putText(frame, f"Knee: {knee_angle_avg:.0f} deg", (px1 + 10, py1 + 44),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (10, 10, 10), 2, cv2.LINE_AA)
    else:
        cv2.putText(frame, "Knee: --", (px1 + 10, py1 + 44),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (10, 10, 10), 2, cv2.LINE_AA)

    if ready_pct is not None:
        cv2.putText(frame, f"READY: {ready_pct:.0f}%", (px1 + 10, py1 + 62),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (10, 10, 10), 2, cv2.LINE_AA)


# ----------------------------
# CALIBRATION
# ----------------------------
def save_calibration(path: str, points: list[list[int]], H: np.ndarray) -> None:
    ensure_dir(os.path.dirname(path))
    data = {
        "setup_name": COURT_SETUP_NAME,
        "court_w_m": COURT_W,
        "court_l_m": COURT_L,
        "dst_points_m": COURT_DST.tolist(),
        "clicked_points_px": points,
        "homography_px_to_m": H.tolist(),
        "saved_at": datetime.now().isoformat(),
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    print(f"✅ Saved calibration: {path}")


def load_calibration(path: str) -> tuple[list[list[int]], np.ndarray] | None:
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
        print(f"✅ Loaded calibration: {path}")
        return pts, H
    except Exception:
        return None


def mouse_callback(event, x, y, flags, param):
    global clicked_points
    if event == cv2.EVENT_LBUTTONDOWN:
        clicked_points.append([x, y])
        print(f"Clicked: ({x}, {y})  -> {len(clicked_points)}/4")


def click_corners(frame0: np.ndarray) -> list[list[int]]:
    global clicked_points
    print("\nKlik 4 court-hjørner i rækkefølge: top-left, top-right, bottom-right, bottom-left\n")
    clicked_points = []
    clone = frame0.copy()

    cv2.namedWindow("Click court corners", cv2.WINDOW_NORMAL)
    cv2.setMouseCallback("Click court corners", mouse_callback)

    while True:
        vis = clone.copy()
        for i, p in enumerate(clicked_points):
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
            clicked_points = []
        if len(clicked_points) == 4:
            break

    cv2.destroyWindow("Click court corners")
    return clicked_points


# ----------------------------
# MANUAL PLAYER INIT
# ----------------------------
def player_click_callback(event, x, y, flags, param):
    global clicked_player_points
    if event == cv2.EVENT_LBUTTONDOWN:
        clicked_player_points.append([x, y])
        print(f"Player click: ({x}, {y}) -> {len(clicked_player_points)}")


def click_players(frame0: np.ndarray, num_players: int = 4) -> list[list[int]]:
    global clicked_player_points
    clicked_player_points = []
    clone = frame0.copy()

    cv2.namedWindow("Click players", cv2.WINDOW_NORMAL)
    cv2.setMouseCallback("Click players", player_click_callback)

    while True:
        vis = clone.copy()
        for i, p in enumerate(clicked_player_points):
            cv2.circle(vis, tuple(p), 9, (0, 255, 0), -1)
            cv2.putText(
                vis,
                f"P{i+1}",
                (p[0] + 12, p[1] - 12),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )

        cv2.putText(
            vis,
            f"Click {num_players} players | R=reset | ESC/Q=quit",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        cv2.imshow("Click players", vis)
        key = cv2.waitKey(10) & 0xFF

        if key in (27, ord("q"), ord("Q")):
            cv2.destroyAllWindows()
            raise SystemExit("User quit during player init.")

        if key in (ord("r"), ord("R")):
            clicked_player_points = []

        if len(clicked_player_points) == num_players:
            break

    cv2.destroyWindow("Click players")
    return clicked_player_points


# ----------------------------
# CLIP EMBEDDER
# ----------------------------
class ClipEmbedder:
    def __init__(self, model_name: str, pretrained: str, device: str):
        self.device = device
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            model_name,
            pretrained=pretrained,
        )
        self.model = self.model.to(device)
        self.model.eval()

    def encode_batch(self, crops: List[Optional[np.ndarray]]) -> List[Optional[np.ndarray]]:
        tensors = []
        idxs = []
        outputs: List[Optional[np.ndarray]] = [None] * len(crops)

        for i, crop in enumerate(crops):
            if crop is None or crop.size == 0:
                continue
            img = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
            tensor = self.preprocess(img)
            tensors.append(tensor)
            idxs.append(i)

        if not tensors:
            return outputs

        batch = torch.stack(tensors).to(self.device)

        with torch.no_grad():
            feat = self.model.encode_image(batch)
            feat = feat / feat.norm(dim=-1, keepdim=True)
            feat = feat.detach().cpu().numpy().astype(np.float32)

        for i, f in zip(idxs, feat):
            outputs[i] = f

        return outputs


# ----------------------------
# IDENTITY MANAGER
# ----------------------------
@dataclass
class IdentityTrack:
    stable_id: int
    last_x: float
    last_y: float
    vx: float
    vy: float
    last_seen_frame: int

    clip_feat: Optional[np.ndarray]
    color_feat: Optional[np.ndarray]

    clip_bank: List[np.ndarray]
    color_bank: List[np.ndarray]

    box_h: float
    box_w: float

    raw_tracker_id: Optional[int]
    hits: int = 1

    pending_raw_tid: Optional[int] = None
    pending_count: int = 0

    preferred_side: Optional[str] = None
    side_stable_frames: int = 0


class IdentityManager:
    def __init__(self, stable_ids: List[int], max_age_frames: int, max_match_dist_m: float):
        self.stable_ids = stable_ids[:]
        self.max_age_frames = int(max_age_frames)
        self.max_match_dist_m = float(max_match_dist_m)
        self.tracks: Dict[int, IdentityTrack] = {}

    def is_bootstrapped(self) -> bool:
        return len(self.tracks) == len(self.stable_ids)

    def bootstrap(self, detections: List[dict], clicked_points: List[List[int]]) -> None:
        used = set()
        for sid, pt in zip(self.stable_ids, clicked_points):
            best_i = None
            best_d = float("inf")
            for i, det in enumerate(detections):
                if i in used:
                    continue
                cx = 0.5 * (det["bbox_x1"] + det["bbox_x2"])
                cy = 0.5 * (det["bbox_y1"] + det["bbox_y2"])
                d = (cx - pt[0]) ** 2 + (cy - pt[1]) ** 2
                if d < best_d:
                    best_d = d
                    best_i = i

            if best_i is None:
                continue

            used.add(best_i)
            det = detections[best_i]

            clip_bank: List[np.ndarray] = []
            color_bank: List[np.ndarray] = []
            if det.get("clip_feat") is not None:
                clip_bank.append(det["clip_feat"])
            if det.get("color_feat") is not None:
                color_bank.append(det["color_feat"])

            pref_side = side_from_y(float(det["y_m"]))

            self.tracks[sid] = IdentityTrack(
                stable_id=sid,
                last_x=float(det["x_m"]),
                last_y=float(det["y_m"]),
                vx=0.0,
                vy=0.0,
                last_seen_frame=int(det["frame_idx"]),
                clip_feat=det.get("clip_feat"),
                color_feat=det.get("color_feat"),
                clip_bank=clip_bank,
                color_bank=color_bank,
                box_h=float(det.get("box_h", 0.0)),
                box_w=float(det.get("box_w", 0.0)),
                raw_tracker_id=int(det["track_id"]) if det.get("track_id") is not None else None,
                hits=1,
                preferred_side=pref_side,
                side_stable_frames=1,
            )
            det["stable_id"] = sid
            det["side_group"] = "ALL"

    def _predict_xy(self, tr: IdentityTrack, frame_idx: int) -> Tuple[float, float]:
        dtf = max(1, frame_idx - tr.last_seen_frame)
        px = tr.last_x + tr.vx * dtf
        py = tr.last_y + tr.vy * dtf
        return px, py

    def _motion_cost(self, tr: IdentityTrack, det: dict, frame_idx: int, occluded: bool) -> float:
        px, py = self._predict_xy(tr, frame_idx)
        d = math.hypot(float(det["x_m"]) - px, float(det["y_m"]) - py)
        base = clamp01(d / max(1e-6, self.max_match_dist_m))

        # Give slightly more trust to motion for established tracks.
        if tr.hits >= 12:
            base *= 0.92

        if occluded:
            base = max(0.0, base - OCCLUSION_MOTION_BOOST)

        return base

    def _clip_cost(self, tr: IdentityTrack, det: dict, occluded: bool) -> float:
        feat = det.get("clip_feat")
        sim_bank = best_bank_similarity(tr.clip_bank, feat, topk=MEMORY_MATCH_TOPK)

        sim_ema = 0.0
        if tr.clip_feat is not None and feat is not None:
            sim_ema = clamp01((cosine_sim(tr.clip_feat, feat) + 1.0) * 0.5)

        if feat is None:
            return 0.45

        sim = max(sim_bank, sim_ema)
        cost = 1.0 - sim

        if occluded:
            cost = min(1.0, cost + OCCLUSION_CLIP_REDUCE)

        return cost

    def _color_cost(self, tr: IdentityTrack, det: dict) -> float:
        feat = det.get("color_feat")
        sim_bank = 0.0
        if feat is not None and tr.color_bank:
            sims = [safe_hist_corr(b, feat) for b in tr.color_bank]
            sims.sort(reverse=True)
            sims = sims[:max(1, min(4, len(sims)))]
            sim_bank = float(sum(sims) / len(sims))

        sim_ema = safe_hist_corr(tr.color_feat, feat) if feat is not None else 0.0
        sim = max(sim_bank, sim_ema)

        if feat is None:
            return 0.35
        return 1.0 - sim

    def _size_cost(self, tr: IdentityTrack, det: dict) -> float:
        dh = float(det.get("box_h", 0.0))
        dw = float(det.get("box_w", 0.0))

        if tr.box_h <= 1e-6 or dh <= 1e-6:
            ch = 0.25
        else:
            ch = clamp01(abs(dh - tr.box_h) / max(dh, tr.box_h))

        if tr.box_w <= 1e-6 or dw <= 1e-6:
            cw = 0.20
        else:
            cw = clamp01(abs(dw - tr.box_w) / max(dw, tr.box_w))

        return 0.65 * ch + 0.35 * cw

    def _side_cost(self, tr: IdentityTrack, det: dict) -> float:
        if tr.preferred_side is None:
            return 0.0

        current_side = side_from_y(float(det["y_m"]))

        if current_side == tr.preferred_side:
            return 0.0

        # very established player -> stronger side lock
        if tr.hits >= HARD_SIDE_HITS_MIN and tr.side_stable_frames >= SIDE_GRACE_FRAMES:
            return HARD_SIDE_PENALTY

        if tr.side_stable_frames < SIDE_GRACE_FRAMES:
            return SIDE_PENALTY

        return SIDE_PENALTY * 0.55

    def _edge_cost(self, det: dict) -> float:
        d = float(det.get("foot_poly_dist", 0.0))
        if d >= 15:
            return 0.0
        if d <= -20:
            return 1.0
        return clamp01((15.0 - d) / 35.0)

    def _total_cost(self, tr: IdentityTrack, det: dict, frame_idx: int, occluded: bool) -> float:
        return (
            W_CLIP * self._clip_cost(tr, det, occluded)
            + W_MOTION * self._motion_cost(tr, det, frame_idx, occluded)
            + W_COLOR * self._color_cost(tr, det)
            + W_SIZE * self._size_cost(tr, det)
            + W_SIDE * self._side_cost(tr, det)
            + W_EDGE * self._edge_cost(det)
        )

    def _push_bank(self, bank: List[np.ndarray], feat: Optional[np.ndarray], max_size: int) -> List[np.ndarray]:
        if feat is None:
            return bank
        bank.append(feat.astype(np.float32))
        if len(bank) > max_size:
            bank = bank[-max_size:]
        return bank

    def _update_side_memory(self, tr: IdentityTrack, det: dict):
        current_side = side_from_y(float(det["y_m"]))
        if tr.preferred_side is None:
            tr.preferred_side = current_side
            tr.side_stable_frames = 1
            return

        if current_side == tr.preferred_side:
            tr.side_stable_frames += 1
        else:
            if tr.side_stable_frames > SIDE_GRACE_FRAMES:
                tr.preferred_side = current_side
                tr.side_stable_frames = 1
            else:
                tr.side_stable_frames = max(0, tr.side_stable_frames - 1)

    def _compute_ambiguity_flags(self, detections: List[dict]) -> List[bool]:
        """
        Marks detections that are visually ambiguous because multiple players
        are almost on the same camera line.
        """
        flags = [False] * len(detections)

        for i in range(len(detections)):
            xi = float(detections[i].get("center_px_x", 0.0))
            yi = float(detections[i].get("center_px_y", 0.0))
            box_i = (
                float(detections[i]["bbox_x1"]),
                float(detections[i]["bbox_y1"]),
                float(detections[i]["bbox_x2"]),
                float(detections[i]["bbox_y2"]),
            )

            for j in range(i + 1, len(detections)):
                xj = float(detections[j].get("center_px_x", 0.0))
                yj = float(detections[j].get("center_px_y", 0.0))
                box_j = (
                    float(detections[j]["bbox_x1"]),
                    float(detections[j]["bbox_y1"]),
                    float(detections[j]["bbox_x2"]),
                    float(detections[j]["bbox_y2"]),
                )

                dx = abs(xi - xj)
                dy = abs(yi - yj)
                iou = bbox_iou_xyxy(box_i, box_j)

                same_line = dx <= AMBIGUOUS_X_PX and dy <= AMBIGUOUS_Y_PX
                slight_overlap = iou >= AMBIGUOUS_IOU_THRESH

                if same_line or slight_overlap:
                    flags[i] = True
                    flags[j] = True

        return flags

    def _compute_occlusion_flags(self, detections: List[dict]) -> List[bool]:
        flags = [False] * len(detections)
        for i in range(len(detections)):
            box_i = (
                float(detections[i]["bbox_x1"]),
                float(detections[i]["bbox_y1"]),
                float(detections[i]["bbox_x2"]),
                float(detections[i]["bbox_y2"]),
            )
            for j in range(i + 1, len(detections)):
                box_j = (
                    float(detections[j]["bbox_x1"]),
                    float(detections[j]["bbox_y1"]),
                    float(detections[j]["bbox_x2"]),
                    float(detections[j]["bbox_y2"]),
                )
                iou = bbox_iou_xyxy(box_i, box_j)
                if iou >= OCCLUSION_IOU_THRESH:
                    flags[i] = True
                    flags[j] = True
        return flags

    def _det_pair_is_close(self, a: dict, b: dict) -> bool:
        dx_m = float(a["x_m"]) - float(b["x_m"])
        dy_m = float(a["y_m"]) - float(b["y_m"])
        dist_m = math.hypot(dx_m, dy_m)

        box_a = (
            float(a["bbox_x1"]),
            float(a["bbox_y1"]),
            float(a["bbox_x2"]),
            float(a["bbox_y2"]),
        )
        box_b = (
            float(b["bbox_x1"]),
            float(b["bbox_y1"]),
            float(b["bbox_x2"]),
            float(b["bbox_y2"]),
        )
        iou = bbox_iou_xyxy(box_a, box_b)

        return dist_m <= PAIR_SWAP_MAX_M or iou >= PAIR_SWAP_IOU_THRESH

    def _anti_swap_pairs(
        self,
        proposed: Dict[int, int],
        detections: List[dict],
        cost_matrix: np.ndarray,
        track_ids: List[int],
        occlusion_flags: List[bool],
        ambiguity_flags: List[bool],
        frame_idx: int,
    ) -> Dict[int, int]:
        """
        Prevents two established tracks from swapping identities in close / ambiguous situations.
        """
        if len(proposed) < 2:
            return proposed

        out = dict(proposed)
        sids = list(proposed.keys())

        for i in range(len(sids)):
            sid_a = sids[i]
            det_a_idx = out.get(sid_a)
            if det_a_idx is None:
                continue

            tr_a = self.tracks[sid_a]
            det_a = detections[det_a_idx]

            for j in range(i + 1, len(sids)):
                sid_b = sids[j]
                det_b_idx = out.get(sid_b)
                if det_b_idx is None or det_b_idx == det_a_idx:
                    continue

                tr_b = self.tracks[sid_b]
                det_b = detections[det_b_idx]

                # only care about established players
                if tr_a.hits < PAIR_SWAP_HIT_MIN or tr_b.hits < PAIR_SWAP_HIT_MIN:
                    continue

                # only care when detections are close / overlapping
                if not self._det_pair_is_close(det_a, det_b):
                    continue

                amb = ambiguity_flags[det_a_idx] or ambiguity_flags[det_b_idx]
                occ = occlusion_flags[det_a_idx] or occlusion_flags[det_b_idx]

                row_a = track_ids.index(sid_a)
                row_b = track_ids.index(sid_b)

                # cost of chosen assignment
                chosen_cost = float(cost_matrix[row_a, det_a_idx] + cost_matrix[row_b, det_b_idx])

                # cost if pair were swapped
                alt_cost = float(cost_matrix[row_a, det_b_idx] + cost_matrix[row_b, det_a_idx])

                margin = PAIR_SWAP_COST_MARGIN
                if amb:
                    margin += PAIR_SWAP_AMBIGUOUS_BONUS
                if occ:
                    margin += 0.06

                # If swapped alternative is only marginally better,
                # prefer stability for established players.
                if alt_cost <= chosen_cost + margin:
                    raw_a = det_a.get("track_id")
                    raw_b = det_b.get("track_id")

                    keep_a = (
                        tr_a.raw_tracker_id is not None
                        and raw_a is not None
                        and int(tr_a.raw_tracker_id) == int(raw_a)
                    )
                    keep_b = (
                        tr_b.raw_tracker_id is not None
                        and raw_b is not None
                        and int(tr_b.raw_tracker_id) == int(raw_b)
                    )

                    # If both align with prior raw ids, keep them as-is.
                    if keep_a and keep_b:
                        continue

                    # If one aligns strongly with prior history and the other does not,
                    # do not let the weak one drag the stable one into a swap.
                    if keep_a and not keep_b:
                        out[sid_a] = det_a_idx
                        continue

                    if keep_b and not keep_a:
                        out[sid_b] = det_b_idx
                        continue

                    # Extra side-lock check: if both tracks have strong preferred sides,
                    # prefer the assignment that preserves side plausibility.
                    det_a_side = side_from_y(float(det_a["y_m"]))
                    det_b_side = side_from_y(float(det_b["y_m"]))

                    chosen_side_penalty = 0.0
                    swapped_side_penalty = 0.0

                    if tr_a.preferred_side is not None and det_a_side != tr_a.preferred_side:
                        chosen_side_penalty += 1.0
                    if tr_b.preferred_side is not None and det_b_side != tr_b.preferred_side:
                        chosen_side_penalty += 1.0

                    if tr_a.preferred_side is not None and det_b_side != tr_a.preferred_side:
                        swapped_side_penalty += 1.0
                    if tr_b.preferred_side is not None and det_a_side != tr_b.preferred_side:
                        swapped_side_penalty += 1.0

                    if chosen_side_penalty <= swapped_side_penalty:
                        continue

        return out

    def _update_track(self, sid: int, det: dict, frame_idx: int, occluded: bool, ambiguous: bool) -> None:
        tr = self.tracks[sid]

        dtf = max(1, frame_idx - tr.last_seen_frame)
        nvx = (float(det["x_m"]) - tr.last_x) / dtf
        nvy = (float(det["y_m"]) - tr.last_y) / dtf

        tr.vx = VEL_ALPHA * nvx + (1.0 - VEL_ALPHA) * tr.vx
        tr.vy = VEL_ALPHA * nvy + (1.0 - VEL_ALPHA) * tr.vy

        tr.last_x = float(det["x_m"])
        tr.last_y = float(det["y_m"])
        tr.last_seen_frame = int(frame_idx)

        new_clip = det.get("clip_feat")
        if new_clip is not None:
            if tr.clip_feat is None:
                tr.clip_feat = new_clip.astype(np.float32)
            else:
                tr.clip_feat = (CLIP_EMA_ALPHA * new_clip + (1.0 - CLIP_EMA_ALPHA) * tr.clip_feat).astype(np.float32)
                n = float(np.linalg.norm(tr.clip_feat))
                if n > 1e-8:
                    tr.clip_feat = (tr.clip_feat / n).astype(np.float32)

            freeze_clip_bank = (occluded and OCCLUSION_FREEZE_BANK_UPDATE) or (ambiguous and AMBIGUOUS_FREEZE_BANK_UPDATE)
            if not freeze_clip_bank:
                tr.clip_bank = self._push_bank(tr.clip_bank, new_clip, MEMORY_BANK_SIZE)

        new_color = det.get("color_feat")
        if new_color is not None:
            if tr.color_feat is None:
                tr.color_feat = new_color.astype(np.float32)
            else:
                tr.color_feat = (COLOR_EMA_ALPHA * new_color + (1.0 - COLOR_EMA_ALPHA) * tr.color_feat).astype(np.float32)

            freeze_color_bank = (occluded and OCCLUSION_FREEZE_BANK_UPDATE) or (ambiguous and AMBIGUOUS_FREEZE_BANK_UPDATE)
            if not freeze_color_bank:
                tr.color_bank = self._push_bank(tr.color_bank, new_color, MEMORY_BANK_SIZE)

        tr.box_h = SIZE_ALPHA * float(det.get("box_h", tr.box_h)) + (1.0 - SIZE_ALPHA) * tr.box_h
        tr.box_w = SIZE_ALPHA * float(det.get("box_w", tr.box_w)) + (1.0 - SIZE_ALPHA) * tr.box_w

        raw_tid = det.get("track_id")
        if raw_tid is not None:
            tr.raw_tracker_id = int(raw_tid)

        tr.hits += 1
        self._update_side_memory(tr, det)

    def assign(self, frame_idx: int, detections: List[dict]) -> None:
        if not detections or not self.is_bootstrapped():
            return

        occlusion_flags = self._compute_occlusion_flags(detections)
        ambiguity_flags = self._compute_ambiguity_flags(detections)

        track_ids = self.stable_ids
        num_tracks = len(track_ids)
        num_dets = len(detections)

        cost_matrix = np.full((num_tracks, num_dets), UNMATCHED_COST, dtype=np.float32)

        for r, sid in enumerate(track_ids):
            tr = self.tracks[sid]
            for c, det in enumerate(detections):
                occluded = occlusion_flags[c]
                ambiguous = ambiguity_flags[c]

                base_cost = self._total_cost(tr, det, frame_idx, occluded)

                if ambiguous:
                    base_cost += 0.08

                raw_tid = det.get("track_id")
                if tr.raw_tracker_id is not None and raw_tid is not None and int(raw_tid) == int(tr.raw_tracker_id):
                    stickiness = OCCLUSION_EXTRA_STICKINESS if occluded else 0.08
                    base_cost = max(0.0, base_cost - stickiness)

                cost_matrix[r, c] = base_cost

        row_ind, col_ind = linear_sum_assignment(cost_matrix)

        proposed: Dict[int, int] = {}
        for r, c in zip(row_ind, col_ind):
            sid = track_ids[r]
            tr = self.tracks[sid]
            det = detections[c]
            occluded = occlusion_flags[c]

            total_cost = float(cost_matrix[r, c])

            motion_cost = self._motion_cost(tr, det, frame_idx, occluded)
            clip_cost = self._clip_cost(tr, det, occluded)

            clip_sim = 1.0 - clip_cost
            motion_dist_ok = motion_cost <= clamp01(OUTSIDER_DIST_M / max(1e-6, self.max_match_dist_m))

            strong_identity = clip_sim >= OUTSIDER_CLIP_MIN_SIM

            ambiguous = ambiguity_flags[c]
            local_max_accept = AMBIGUOUS_MAX_ACCEPT_COST if ambiguous else MAX_ACCEPT_COST
            acceptable_total = total_cost <= local_max_accept

            if acceptable_total and (motion_dist_ok or strong_identity):
                proposed[sid] = c

        proposed = self._anti_swap_pairs(
            proposed=proposed,
            detections=detections,
            cost_matrix=cost_matrix,
            track_ids=track_ids,
            occlusion_flags=occlusion_flags,
            ambiguity_flags=ambiguity_flags,
            frame_idx=frame_idx,
        )

        for sid, det_idx in proposed.items():
            tr = self.tracks[sid]
            det = detections[det_idx]
            occluded = occlusion_flags[det_idx]
            ambiguous = ambiguity_flags[det_idx]

            row_ix = track_ids.index(sid)
            assigned_cost = float(cost_matrix[row_ix, det_idx])

            row_costs = cost_matrix[row_ix]
            others = [float(v) for j, v in enumerate(row_costs) if j != det_idx]
            second_best_cost = min(others) if others else None

            raw_tid = det.get("track_id")
            same_raw = (
                tr.raw_tracker_id is not None and
                raw_tid is not None and
                int(raw_tid) == int(tr.raw_tracker_id)
            )

            accept = True
            local_switch_margin = SWITCH_MARGIN
            local_confirm = SWITCH_CONFIRM_FRAMES

            if occluded:
                local_switch_margin += OCCLUSION_EXTRA_STICKINESS
                local_confirm += 2

            if ambiguous:
                local_switch_margin += AMBIGUOUS_EXTRA_SWITCH_MARGIN
                local_confirm += AMBIGUOUS_EXTRA_CONFIRM_FRAMES

            if not same_raw and second_best_cost is not None:
                improvement = second_best_cost - assigned_cost
                if improvement >= local_switch_margin:
                    candidate = int(raw_tid) if raw_tid is not None else -1
                    if tr.pending_raw_tid == candidate:
                        tr.pending_count += 1
                    else:
                        tr.pending_raw_tid = candidate
                        tr.pending_count = 1

                    if tr.pending_count < local_confirm:
                        accept = False
                    else:
                        tr.pending_raw_tid = None
                        tr.pending_count = 0
                else:
                    tr.pending_raw_tid = None
                    tr.pending_count = 0
            else:
                tr.pending_raw_tid = None
                tr.pending_count = 0

            if accept:
                det["stable_id"] = sid
                det["side_group"] = "ALL"
                self._update_track(sid, det, frame_idx, occluded, ambiguous)


# ----------------------------
# LIVE STATE
# ----------------------------
@dataclass
class LiveState:
    last_t: float
    last_x: float
    last_y: float
    speed_kmh_ema: float
    speed_mps_ema: float
    accel_mps2_ema: float
    prev_speed_mps: float
    total_dist_m: float

    knee_angle_ema: float
    hip_drop_ema: float
    ready_frames: int
    total_frames: int
    last_pose_frame: int
    last_knee_angle: Optional[float]
    last_ready_flag: int


def init_state(t: float, x_m: float, y_m: float) -> LiveState:
    return LiveState(
        last_t=t, last_x=x_m, last_y=y_m,
        speed_kmh_ema=0.0, speed_mps_ema=0.0,
        accel_mps2_ema=0.0, prev_speed_mps=0.0,
        total_dist_m=0.0,
        knee_angle_ema=175.0, hip_drop_ema=0.0,
        ready_frames=0, total_frames=0,
        last_pose_frame=-999999,
        last_knee_angle=None,
        last_ready_flag=0,
    )


def update_motion(states: Dict[int, LiveState], stable_id: int, t: float, x_m: float, y_m: float) -> Tuple[float, float, float]:
    if stable_id <= 0:
        return 0.0, 0.0, 0.0

    st = states.get(stable_id)
    if st is None:
        states[stable_id] = init_state(t, x_m, y_m)
        return 0.0, 0.0, 0.0

    dt = t - st.last_t
    if dt <= 1e-6:
        return st.speed_kmh_ema, st.total_dist_m, st.accel_mps2_ema

    dx = x_m - st.last_x
    dy = y_m - st.last_y
    step = math.sqrt(dx * dx + dy * dy)

    if step <= DIST_JUMP_CAP_M:
        st.total_dist_m += step

        speed_mps_inst = step / dt
        speed_kmh_inst = float(min(max(speed_mps_inst * 3.6, 0.0), SPEED_CAP_KMH))

        st.speed_kmh_ema = SPEED_SMOOTH_ALPHA * speed_kmh_inst + (1.0 - SPEED_SMOOTH_ALPHA) * st.speed_kmh_ema
        st.speed_mps_ema = st.speed_kmh_ema / 3.6

        accel_inst = (st.speed_mps_ema - st.prev_speed_mps) / dt
        accel_inst = float(np.clip(accel_inst, -ACCEL_CAP_MPS2, ACCEL_CAP_MPS2))

        st.accel_mps2_ema = ACCEL_SMOOTH_ALPHA * accel_inst + (1.0 - ACCEL_SMOOTH_ALPHA) * st.accel_mps2_ema
        st.prev_speed_mps = st.speed_mps_ema

    st.last_t = t
    st.last_x = x_m
    st.last_y = y_m

    return st.speed_kmh_ema, st.total_dist_m, st.accel_mps2_ema


def get_ready_pct(states: Dict[int, LiveState], stable_id: int) -> float:
    st = states.get(stable_id)
    if st is None or st.total_frames <= 0:
        return 0.0
    return 100.0 * float(st.ready_frames) / float(st.total_frames)


# ----------------------------
# POSE HELPERS
# ----------------------------
def angle_deg(a: Tuple[float, float], b: Tuple[float, float], c: Tuple[float, float]) -> float:
    ax, ay = a
    bx, by = b
    cx, cy = c
    v1 = np.array([ax - bx, ay - by], dtype=np.float32)
    v2 = np.array([cx - bx, cy - by], dtype=np.float32)
    n1 = float(np.linalg.norm(v1))
    n2 = float(np.linalg.norm(v2))
    if n1 < 1e-6 or n2 < 1e-6:
        return 180.0
    cosang = float(np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0))
    return float(np.degrees(np.arccos(cosang)))


def resize_keep_aspect(img, max_w, max_h):
    if max_w <= 0 or max_h <= 0:
        return img, 1.0
    h, w = img.shape[:2]
    scale = min(max_w / float(w), max_h / float(h), 1.0)
    if scale >= 0.999:
        return img, 1.0
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR), scale


def compute_ready_from_pose(pose_res) -> Tuple[Optional[float], Optional[float], Optional[float], Optional[float], int]:
    if pose_res is None or pose_res.pose_landmarks is None:
        return None, None, None, None, 0

    lm = pose_res.pose_landmarks.landmark

    def get(i):
        return lm[i].x, lm[i].y, lm[i].visibility

    Lhip = get(23); Rhip = get(24)
    Lknee = get(25); Rknee = get(26)
    Lank = get(27); Rank = get(28)

    vis_ok = (Lhip[2] > POSE_MIN_VIS and Lknee[2] > POSE_MIN_VIS and Lank[2] > POSE_MIN_VIS and
              Rhip[2] > POSE_MIN_VIS and Rknee[2] > POSE_MIN_VIS and Rank[2] > POSE_MIN_VIS)
    if not vis_ok:
        return None, None, None, None, 0

    knee_L = angle_deg((Lhip[0], Lhip[1]), (Lknee[0], Lknee[1]), (Lank[0], Lank[1]))
    knee_R = angle_deg((Rhip[0], Rhip[1]), (Rknee[0], Rknee[1]), (Rank[0], Rank[1]))
    knee_avg = 0.5 * (knee_L + knee_R)

    hip_y = 0.5 * (Lhip[1] + Rhip[1])
    ank_y = 0.5 * (Lank[1] + Rank[1])
    hip_to_ank = float(np.clip(ank_y - hip_y, 0.0, 1.0))
    hip_drop = float(np.clip(1.0 - hip_to_ank, 0.0, 1.0))

    ready = 1 if (knee_avg <= READY_KNEE_ANGLE_MAX and hip_drop >= READY_HIP_DROP_MIN) else 0
    return float(knee_L), float(knee_R), float(knee_avg), float(hip_drop), int(ready)


def update_ready_state(states: Dict[int, LiveState], stable_id: int, frame_idx: int,
                      knee_avg: Optional[float], hip_drop: Optional[float], ready_flag: int) -> None:
    st = states.get(stable_id)
    if st is None:
        return

    st.total_frames += 1
    if ready_flag == 1:
        st.ready_frames += 1

    if knee_avg is not None:
        st.knee_angle_ema = READY_SMOOTH_ALPHA * knee_avg + (1.0 - READY_SMOOTH_ALPHA) * st.knee_angle_ema
        st.last_knee_angle = float(st.knee_angle_ema)

    if hip_drop is not None:
        st.hip_drop_ema = READY_HIP_SMOOTH_ALPHA * hip_drop + (1.0 - READY_HIP_SMOOTH_ALPHA) * st.hip_drop_ema

    st.last_ready_flag = int(ready_flag)
    st.last_pose_frame = int(frame_idx)


# ----------------------------
# MAIN
# ----------------------------
def main():
    if not os.path.exists(VIDEO_PATH):
        raise RuntimeError(f"Fil findes ikke: {VIDEO_PATH}")

    print("Loading CLIP...")
    clip_embedder = ClipEmbedder(CLIP_MODEL_NAME, CLIP_PRETRAINED, CLIP_DEVICE)
    print(f"✅ CLIP loaded on {CLIP_DEVICE}")

    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        raise RuntimeError(f"Kunne ikke åbne videoen: {VIDEO_PATH}")

    ok, frame0 = cap.read()
    if not ok or frame0 is None:
        raise RuntimeError("Videoen blev åbnet, men første frame kunne ikke læses.")

    loaded = load_calibration(CALIB_FILE)
    if loaded is None:
        pts = click_corners(frame0)
        src = np.array(pts, dtype=np.float32)
        H, _ = cv2.findHomography(src, COURT_DST)
        if H is None:
            raise RuntimeError("Kunne ikke beregne homography.")
        save_calibration(CALIB_FILE, pts, H)
    else:
        pts, H = loaded

    src = np.array(pts, dtype=np.float32)
    court_poly_px = src.reshape(-1, 2).astype(np.int32)

    print(f"Klik {MAX_PLAYERS} spillere i første frame...")
    player_points = click_players(frame0, MAX_PLAYERS)

    cap.release()
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        raise RuntimeError(f"Kunne ikke åbne videoen (restart): {VIDEO_PATH}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or frame0.shape[1]
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or frame0.shape[0]
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    if not fps or fps != fps or fps <= 1:
        fps = SAVE_FPS_FALLBACK

    ensure_dir(SAVE_DIR)
    out_path = make_output_path(SAVE_DIR, VIDEO_PATH)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_path, fourcc, fps, (width, height))
    if not writer.isOpened():
        fourcc = cv2.VideoWriter_fourcc(*"avc1")
        writer = cv2.VideoWriter(out_path, fourcc, fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError("Kunne ikke åbne VideoWriter.")

    ensure_dir(EXPORT_DIR)
    csv_path, jsonl_path = make_export_paths(EXPORT_DIR, VIDEO_PATH)

    csv_f = open(csv_path, "w", newline="", encoding="utf-8")
    csv_w = csv.DictWriter(
        csv_f,
        fieldnames=[
            "frame_idx", "timestamp_s",
            "track_id",
            "stable_id", "side_group",
            "bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2",
            "foot_px_x", "foot_px_y",
            "x_m", "y_m",
            "zone",
            "speed_kmh", "distance_m",
            "accel_mps2",
            "knee_angle_L", "knee_angle_R", "knee_angle_avg",
            "ready_flag", "ready_pct",
        ],
    )
    csv_w.writeheader()
    jsonl_f = open(jsonl_path, "w", encoding="utf-8")

    print("Loading YOLO...")
    model = YOLO("yolo11m.pt")
    tracker = sv.ByteTrack()

    max_age_frames = int(MAX_AGE_SECONDS * fps)
    identity = IdentityManager(PLAYER_IDS, max_age_frames, MAX_MATCH_DIST_M)
    live_states: Dict[int, LiveState] = {}

    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        enable_segmentation=False,
        min_detection_confidence=0.4,
        min_tracking_confidence=0.4,
    )

    print(f"💾 Saving annotated video to: {out_path}")
    print(f"📦 Exporting detections to:\n   CSV:   {csv_path}\n   JSONL: {jsonl_path}")
    print(f"   size={width}x{height}, fps={fps:.2f}")
    print(f"   YOLO params: conf={CONF}, iou={IOU}, imgsz={IMGSZ}")
    print(f"   CLIP: {CLIP_MODEL_NAME} / {CLIP_PRETRAINED}")
    print(f"   Court margin: top={COURT_MARGIN_PX}px, left/right/bottom={SIDE_BOTTOM_MARGIN_PX}px")
    print(f"   Ready: knee<= {READY_KNEE_ANGLE_MAX:.0f}, hip_drop>= {READY_HIP_DROP_MIN:.2f}, speed<= {READY_MAX_SPEED_KMH:.1f} km/h")

    cv2.namedWindow("Badminton tracking", cv2.WINDOW_NORMAL)

    frame_idx = -1
    bootstrapped = False

    try:
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                print("⚠️ Video ended or could not read next frame.")
                break

            frame_idx += 1
            timestamp_s = frame_idx / float(fps)

            results = model(frame, verbose=False, conf=CONF, iou=IOU, imgsz=IMGSZ)[0]
            detections = sv.Detections.from_ultralytics(results)

            annotated = frame
            det_rows: List[dict] = []

            if detections.class_id is not None and len(detections) > 0:
                detections = detections[detections.class_id == 0]

                if len(detections) > 0:
                    keep = []
                    for xyxy in detections.xyxy:
                        x1, y1, x2, y2 = xyxy
                        foot = (float((x1 + x2) / 2), float(y2))

                        inside = point_in_court_asymmetric_margin(
                            foot,
                            court_poly_px,
                            top_margin=COURT_MARGIN_PX,
                            side_bottom_margin=SIDE_BOTTOM_MARGIN_PX,
                        )
                        keep.append(inside)

                    detections = detections[np.array(keep, dtype=bool)]

                detections = tracker.update_with_detections(detections)

                if detections.tracker_id is not None and len(detections) > 0:
                    crops: List[Optional[np.ndarray]] = []
                    temp_rows: List[dict] = []

                    for xyxy, track_id in zip(detections.xyxy, detections.tracker_id):
                        x1, y1, x2, y2 = xyxy.astype(float)
                        foot_px_x = (x1 + x2) / 2.0
                        foot_px_y = y2
                        center_px_x = float((x1 + x2) / 2.0)
                        center_px_y = float((y1 + y2) / 2.0)

                        x_m, y_m = px_to_meters(H, foot_px_x, foot_px_y)
                        x_m = float(np.clip(x_m, 0.0, COURT_W))
                        y_m = float(np.clip(y_m, 0.0, COURT_L))

                        zone = classify_zone(x_m, y_m)
                        box_w = float(max(1.0, x2 - x1))
                        box_h = float(max(1.0, y2 - y1))
                        foot_poly_dist = point_to_polygon_signed_distance((float(foot_px_x), float(foot_px_y)), court_poly_px)

                        crop = crop_person(frame, (x1, y1, x2, y2), frac=0.04)
                        color_feat = extract_color_feature(frame, (x1, y1, x2, y2))

                        temp_rows.append({
                            "frame_idx": frame_idx,
                            "timestamp_s": timestamp_s,
                            "track_id": int(track_id),
                            "bbox_x1": float(x1), "bbox_y1": float(y1),
                            "bbox_x2": float(x2), "bbox_y2": float(y2),
                            "foot_px_x": float(foot_px_x), "foot_px_y": float(foot_px_y),
                            "center_px_x": center_px_x, "center_px_y": center_px_y,
                            "foot_poly_dist": float(foot_poly_dist),
                            "x_m": x_m, "y_m": y_m,
                            "zone": zone,
                            "box_w": box_w,
                            "box_h": box_h,
                            "clip_feat": None,
                            "color_feat": color_feat,
                            "stable_id": -1,
                            "side_group": "ALL",
                        })
                        crops.append(crop)

                    clip_feats = clip_embedder.encode_batch(crops)
                    for row, feat in zip(temp_rows, clip_feats):
                        row["clip_feat"] = feat

                    det_rows = temp_rows

                    if not bootstrapped:
                        identity.bootstrap(det_rows, player_points)
                        bootstrapped = identity.is_bootstrapped()
                    else:
                        identity.assign(frame_idx, det_rows)

                    det_rows = [r for r in det_rows if int(r.get("stable_id", -1)) in PLAYER_IDS]

                    xyxy_by_track = {int(tid): xy for xy, tid in zip(detections.xyxy, detections.tracker_id)}
                    do_pose_this_frame = (frame_idx % int(max(1, POSE_EVERY_N_FRAMES)) == 0)
                    H_img, W_img = frame.shape[:2]

                    for r in det_rows:
                        sid = int(r["stable_id"])
                        track_id = int(r["track_id"])
                        xyxy = xyxy_by_track.get(track_id)
                        if xyxy is None:
                            continue

                        if sid not in live_states:
                            live_states[sid] = init_state(timestamp_s, r["x_m"], r["y_m"])

                        sp_kmh, dist_m, accel_mps2 = update_motion(live_states, sid, timestamp_s, r["x_m"], r["y_m"])
                        r["speed_kmh"] = float(sp_kmh)
                        r["distance_m"] = float(dist_m)
                        r["accel_mps2"] = float(accel_mps2)

                        r["knee_angle_L"] = None
                        r["knee_angle_R"] = None
                        r["knee_angle_avg"] = None
                        r["ready_flag"] = 0
                        r["ready_pct"] = float(get_ready_pct(live_states, sid))

                        if do_pose_this_frame:
                            x1, y1, x2, y2 = xyxy.astype(float)
                            bx1, by1, bx2, by2 = expand_bbox(x1, y1, x2, y2, W_img, H_img, frac=POSE_BBOX_EXPAND)
                            crop = frame[by1:by2, bx1:bx2]

                            if crop.size > 0:
                                crop_small, _ = resize_keep_aspect(crop, POSE_CROP_MAX_W, POSE_CROP_MAX_H)
                                crop_rgb = cv2.cvtColor(crop_small, cv2.COLOR_BGR2RGB)
                                pose_res = pose.process(crop_rgb)

                                knee_L, knee_R, knee_avg, hip_drop, ready_flag = compute_ready_from_pose(pose_res)

                                if ready_flag == 1 and r["speed_kmh"] > READY_MAX_SPEED_KMH:
                                    ready_flag = 0

                                update_ready_state(
                                    live_states, sid, frame_idx,
                                    knee_avg=knee_avg,
                                    hip_drop=hip_drop,
                                    ready_flag=ready_flag
                                )

                                st = live_states.get(sid)
                                r["knee_angle_L"] = knee_L
                                r["knee_angle_R"] = knee_R
                                r["knee_angle_avg"] = st.last_knee_angle if (st is not None and st.last_knee_angle is not None) else knee_avg
                                r["ready_flag"] = int(ready_flag)
                                r["ready_pct"] = float(get_ready_pct(live_states, sid))

                        st = live_states.get(sid)
                        if st is not None:
                            if r["knee_angle_avg"] is None:
                                r["knee_angle_avg"] = st.last_knee_angle
                            if not do_pose_this_frame:
                                r["ready_flag"] = int(st.last_ready_flag)
                            r["ready_pct"] = float(get_ready_pct(live_states, sid))

                        ring_col = knee_to_bgr(r["knee_angle_avg"])
                        draw_player_overlay(
                            annotated,
                            xyxy,
                            stable_id=sid,
                            knee_angle_avg=r["knee_angle_avg"],
                            ready_pct=r["ready_pct"],
                            ring_color=ring_col,
                        )

                    for r in det_rows:
                        row_out = dict(r)
                        row_out.pop("clip_feat", None)
                        row_out.pop("color_feat", None)
                        row_out.pop("box_w", None)
                        row_out.pop("box_h", None)
                        row_out.pop("foot_poly_dist", None)
                        row_out.pop("center_px_x", None)
                        row_out.pop("center_px_y", None)

                        row_out.setdefault("stable_id", -1)
                        row_out.setdefault("side_group", "ALL")
                        row_out.setdefault("speed_kmh", 0.0)
                        row_out.setdefault("distance_m", 0.0)
                        row_out.setdefault("accel_mps2", 0.0)
                        row_out.setdefault("knee_angle_L", None)
                        row_out.setdefault("knee_angle_R", None)
                        row_out.setdefault("knee_angle_avg", None)
                        row_out.setdefault("ready_flag", 0)
                        row_out.setdefault("ready_pct", 0.0)

                        csv_w.writerow(row_out)
                        jsonl_f.write(json.dumps(row_out, ensure_ascii=False) + "\n")

            cv2.polylines(annotated, [court_poly_px], isClosed=True, color=(255, 255, 0), thickness=2)
            try:
                draw_court_guides(annotated, H)
            except Exception:
                pass

            cv2.putText(
                annotated,
                f"CLIP identity + ambiguity handling + anti-swap | conf={CONF} iou={IOU} imgsz={IMGSZ}",
                (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2, cv2.LINE_AA,
            )
            cv2.putText(
                annotated,
                f"Margins top={COURT_MARGIN_PX}px sides/bottom={SIDE_BOTTOM_MARGIN_PX}px",
                (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA,
            )
            cv2.putText(
                annotated,
                f"Ready: knee<={READY_KNEE_ANGLE_MAX:.0f} hip>={READY_HIP_DROP_MIN:.2f} speed<={READY_MAX_SPEED_KMH:.1f}",
                (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA,
            )

            cv2.imshow("Badminton tracking", annotated)
            writer.write(annotated)

            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q"), ord("Q")):
                print("Stopped by user.")
                break

    finally:
        cap.release()
        writer.release()
        csv_f.close()
        jsonl_f.close()
        cv2.destroyAllWindows()

    print(f"✅ Saved video: {out_path}")
    print(f"✅ Saved CSV:   {csv_path}")
    print(f"✅ Saved JSONL: {jsonl_path}")
    print("Done.")


if __name__ == "__main__":
    try:
        main()
    except SystemExit as e:
        print(str(e))
    except Exception:
        import traceback
        traceback.print_exc()
        input("\nPress Enter to exit...")
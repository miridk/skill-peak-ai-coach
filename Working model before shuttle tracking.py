import os
import json
import csv
import math
import cv2
import numpy as np
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

# Navn for kameravinkel/hal, så du genbruger calibration
COURT_SETUP_NAME = "almind_viuf_hallen_baseline"
CALIB_DIR = "calibration"
CALIB_FILE = os.path.join(CALIB_DIR, f"{COURT_SETUP_NAME}.json")

SAVE_FPS_FALLBACK = 30.0

# Badminton court (meter)
COURT_W = 6.10
COURT_L = 13.40

# Destination points in "top-down court coords" (meter)
# Order: top-left, top-right, bottom-right, bottom-left
COURT_DST = np.array(
    [[0.0, 0.0], [COURT_W, 0.0], [COURT_W, COURT_L], [0.0, COURT_L]],
    dtype=np.float32,
)

# Filter settings
COURT_MARGIN_PX = 75
CONF = 0.45
IOU = 0.6
IMGSZ = 960

# Stable ID settings
USE_TWO_SIDED_SLOTS = True
MAX_AGE_SECONDS = 1.8
MAX_MATCH_DIST_M = 2.0

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

# ----------------------------
# ✅ READY-STANCE / POSE CONFIG (IMPLEMENTED)
# ----------------------------
POSE_EVERY_N_FRAMES = 2          # run pose every N frames (performance)
POSE_MIN_VIS = 0.25              # lowered from 0.35 -> more robust
POSE_BBOX_EXPAND = 0.18          # increased from 0.12 -> include knees/ankles more often
POSE_CROP_MAX_W = 320            # increased from 256 -> better landmarks
POSE_CROP_MAX_H = 320

# READY thresholds (tuned for baseline camera)
READY_KNEE_ANGLE_MAX = 160.0     # was 140
READY_HIP_DROP_MIN = 0.04        # was 0.06

# Optional: only count "ready" when mostly standing still
READY_MAX_SPEED_KMH = 6.0

# smoothing
READY_SMOOTH_ALPHA = 0.35
READY_HIP_SMOOTH_ALPHA = 0.35

clicked_points: list[list[int]] = []


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


def px_to_meters(H: np.ndarray, x_px: float, y_px: float) -> tuple[float, float]:
    pt = np.array([[[x_px, y_px]]], dtype=np.float32)
    out = cv2.perspectiveTransform(pt, H)[0, 0]
    return float(out[0]), float(out[1])


def meters_to_px(H: np.ndarray, x_m: float, y_m: float) -> tuple[int, int]:
    Hinv = np.linalg.inv(H)
    pt = np.array([[[x_m, y_m]]], dtype=np.float32)
    out = cv2.perspectiveTransform(pt, Hinv)[0, 0]
    return int(round(out[0])), int(round(out[1]))


def draw_court_guides(frame: np.ndarray, H: np.ndarray) -> None:
    mid_y = COURT_L / 2.0
    short = 1.98

    def line_pts(xa, ya, xb, yb):
        p1 = meters_to_px(H, xa, ya)
        p2 = meters_to_px(H, xb, yb)
        return p1, p2

    # Net
    p1, p2 = line_pts(0.0, mid_y, COURT_W, mid_y)
    cv2.line(frame, p1, p2, (0, 255, 255), 3, cv2.LINE_AA)

    # Short service lines
    p1s, p2s = line_pts(0.0, mid_y - short, COURT_W, mid_y - short)
    cv2.line(frame, p1s, p2s, (255, 255, 0), 2, cv2.LINE_AA)

    p1s2, p2s2 = line_pts(0.0, mid_y + short, COURT_W, mid_y + short)
    cv2.line(frame, p1s2, p2s2, (255, 255, 0), 2, cv2.LINE_AA)

    # Centerline
    p1c, p2c = line_pts(COURT_W / 2.0, 0.0, COURT_W / 2.0, COURT_L)
    cv2.line(frame, p1c, p2c, (255, 0, 255), 2, cv2.LINE_AA)

    cv2.putText(frame, "NET", (min(p1[0], p2[0]) + 8, min(p1[1], p2[1]) - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, cv2.LINE_AA)


def classify_zone(x_m: float, y_m: float) -> str:
    mid = COURT_L / 2.0
    side_group = "FAR" if y_m < mid else "NEAR"
    dist_to_net = abs(y_m - mid)
    depth = "FRONT" if dist_to_net <= 1.98 else "BACK"
    lr = "LEFT" if x_m < COURT_W / 2.0 else "RIGHT"
    return f"{side_group}-{depth}-{lr}"


# ----------------------------
# DRAW HELPERS
# ----------------------------
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


# stance color (green -> red when knees bend more)
def knee_to_bgr(knee_angle_deg: Optional[float]) -> Tuple[int, int, int]:
    if knee_angle_deg is None:
        return (0, 255, 0)
    a = float(np.clip(knee_angle_deg, 80.0, 175.0))
    t = (175.0 - a) / (175.0 - 80.0)  # 0..1
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

    arc_start = 0.0
    arc_end = math.pi

    overlay = frame.copy()
    shadow_center = (cx, ring_y + int(ry * 0.55))
    draw_open_ellipse_arc(overlay, shadow_center, (rx, ry), arc_start, arc_end, (0, 0, 0), 8, RING_ARC_SAMPLES)
    overlay_alpha(frame, overlay, alpha=RING_SHADOW_ALPHA)

    overlay = frame.copy()
    draw_open_ellipse_arc(overlay, (cx, ring_y), (rx, ry), arc_start, arc_end, ring_color, RING_THICKNESS, RING_ARC_SAMPLES)
    overlay_alpha(frame, overlay, alpha=ring_alpha)

    # Panel
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

    # Text
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
# CALIBRATION (save/load)
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
    print("\nControls:")
    print("  ESC/Q = quit")
    print("  R     = reset corner clicks")
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

        cv2.putText(
            vis,
            "Click 4 corners: TL, TR, BR, BL | R=reset | ESC/Q=quit",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        cv2.imshow("Click court corners", vis)
        key = cv2.waitKey(10) & 0xFF

        if key in (27, ord("q"), ord("Q")):
            cv2.destroyAllWindows()
            raise SystemExit("User quit during calibration.")

        if key in (ord("r"), ord("R")):
            clicked_points = []
            print("Reset clicks (0/4).")

        if len(clicked_points) == 4:
            break

    cv2.destroyWindow("Click court corners")
    return clicked_points


# ----------------------------
# STABLE SLOT ASSIGNER
# ----------------------------
@dataclass
class Slot:
    stable_id: int
    last_x: float
    last_y: float
    last_seen_frame: int


class StableSlotAssigner:
    def __init__(self, stable_ids: List[int], max_age_frames: int, max_match_dist_m: float):
        self.stable_ids = stable_ids[:]
        self.max_age_frames = max_age_frames
        self.max_match_dist_m = max_match_dist_m
        self.slots: Dict[int, Slot] = {}

    def _active_slots(self, frame_idx: int) -> List[Slot]:
        return [s for s in self.slots.values() if (frame_idx - s.last_seen_frame) <= self.max_age_frames]

    def _create_or_reuse_slot(self, x: float, y: float, frame_idx: int) -> int:
        used = set(self.slots.keys())
        for sid in self.stable_ids:
            if sid not in used:
                self.slots[sid] = Slot(sid, x, y, frame_idx)
                return sid
        oldest = min(self.slots.values(), key=lambda s: s.last_seen_frame)
        self.slots[oldest.stable_id] = Slot(oldest.stable_id, x, y, frame_idx)
        return oldest.stable_id

    def assign(self, frame_idx: int, det_points: List[Tuple[float, float]]) -> List[int]:
        if not det_points:
            return []

        active = self._active_slots(frame_idx)
        assignments: List[Optional[int]] = [None] * len(det_points)
        used_slots = set()

        cand = []
        for i, (x, y) in enumerate(det_points):
            best_sid = None
            best_d = 1e9
            for s in active:
                if s.stable_id in used_slots:
                    continue
                d = math.hypot(x - s.last_x, y - s.last_y)
                if d < best_d:
                    best_d = d
                    best_sid = s.stable_id
            cand.append((best_d, i, best_sid))

        cand.sort(key=lambda t: t[0])

        for best_d, i, sid in cand:
            if sid is None or sid in used_slots:
                continue
            if best_d <= self.max_match_dist_m:
                assignments[i] = sid
                used_slots.add(sid)
                self.slots[sid].last_x = det_points[i][0]
                self.slots[sid].last_y = det_points[i][1]
                self.slots[sid].last_seen_frame = frame_idx

        for i, sid in enumerate(assignments):
            if sid is None:
                assignments[i] = self._create_or_reuse_slot(det_points[i][0], det_points[i][1], frame_idx)

        return [int(a) for a in assignments]


def assign_stable_ids_two_sided(frame_idx: int, det_rows: List[dict], near_assigner: StableSlotAssigner, far_assigner: StableSlotAssigner) -> None:
    mid = COURT_L / 2.0
    far_idx = [i for i, r in enumerate(det_rows) if r["y_m"] <= mid]
    near_idx = [i for i, r in enumerate(det_rows) if r["y_m"] > mid]

    far_points = [(det_rows[i]["x_m"], det_rows[i]["y_m"]) for i in far_idx]
    near_points = [(det_rows[i]["x_m"], det_rows[i]["y_m"]) for i in near_idx]

    far_ids = far_assigner.assign(frame_idx, far_points)
    near_ids = near_assigner.assign(frame_idx, near_points)

    for i, sid in zip(far_idx, far_ids):
        det_rows[i]["stable_id"] = sid
        det_rows[i]["side_group"] = "FAR"
    for i, sid in zip(near_idx, near_ids):
        det_rows[i]["stable_id"] = sid
        det_rows[i]["side_group"] = "NEAR"


def assign_stable_ids_single_assigner(frame_idx: int, det_rows: List[dict], assigner: StableSlotAssigner) -> None:
    points = [(r["x_m"], r["y_m"]) for r in det_rows]
    sids = assigner.assign(frame_idx, points)
    for r, sid in zip(det_rows, sids):
        r["stable_id"] = sid
        r["side_group"] = "ALL"


# ----------------------------
# LIVE STATE
# ----------------------------
@dataclass
class LiveState:
    # motion
    last_t: float
    last_x: float
    last_y: float
    speed_kmh_ema: float
    speed_mps_ema: float
    accel_mps2_ema: float
    prev_speed_mps: float
    total_dist_m: float

    # ready stance
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
    """
    Returns:
      knee_L, knee_R, knee_avg, hip_drop, ready_flag
    hip_drop: 0..1 (higher = more "drop" / hips lower)
    """
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
            raise RuntimeError("Kunne ikke beregne homography. Prøv at klikke hjørnerne mere præcist.")
        save_calibration(CALIB_FILE, pts, H)
    else:
        pts, H = loaded

    src = np.array(pts, dtype=np.float32)
    court_poly_px = src.reshape(-1, 2).astype(np.int32)

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
        raise RuntimeError("Kunne ikke åbne VideoWriter. Prøv .avi + XVID eller installér codec.")

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

    print("Loading model…")
    model = YOLO("yolo11n.pt")
    tracker = sv.ByteTrack()

    max_age_frames = int(MAX_AGE_SECONDS * fps)
    if USE_TWO_SIDED_SLOTS:
        far_assigner = StableSlotAssigner([1, 2], max_age_frames, MAX_MATCH_DIST_M)
        near_assigner = StableSlotAssigner([3, 4], max_age_frames, MAX_MATCH_DIST_M)
        all_assigner = None
        print("✅ Stable IDs: two-sided (FAR=P1,P2, NEAR=P3,P4)")
    else:
        all_assigner = StableSlotAssigner([1, 2, 3, 4], max_age_frames, MAX_MATCH_DIST_M)
        far_assigner = None
        near_assigner = None
        print("✅ Stable IDs: single assigner (P1..P4)")

    live_states: Dict[int, LiveState] = {}

    # MediaPipe Pose init
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
    print(f"   Pose: every_n={POSE_EVERY_N_FRAMES}, vis>={POSE_MIN_VIS}, expand={POSE_BBOX_EXPAND}, crop_max={POSE_CROP_MAX_W}x{POSE_CROP_MAX_H}")
    print(f"   Ready: knee<= {READY_KNEE_ANGLE_MAX:.0f}, hip_drop>= {READY_HIP_DROP_MIN:.2f}, speed<= {READY_MAX_SPEED_KMH:.1f} km/h")

    cv2.namedWindow("Badminton tracking", cv2.WINDOW_NORMAL)

    frame_idx = -1

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
                        foot = (int((x1 + x2) / 2), int(y2))
                        midpt = (int((x1 + x2) / 2), int((y1 + y2) / 2))
                        inside = point_in_polygon_margin(foot, court_poly_px, margin_px=COURT_MARGIN_PX) or \
                                 point_in_polygon_margin(midpt, court_poly_px, margin_px=COURT_MARGIN_PX)
                        keep.append(inside)
                    detections = detections[np.array(keep, dtype=bool)]

                detections = tracker.update_with_detections(detections)

                if detections.tracker_id is not None and len(detections) > 0:
                    for xyxy, track_id in zip(detections.xyxy, detections.tracker_id):
                        x1, y1, x2, y2 = xyxy.astype(float)
                        foot_px_x = (x1 + x2) / 2.0
                        foot_px_y = y2

                        x_m, y_m = px_to_meters(H, foot_px_x, foot_px_y)
                        x_m = float(np.clip(x_m, 0.0, COURT_W))
                        y_m = float(np.clip(y_m, 0.0, COURT_L))

                        zone = classify_zone(x_m, y_m)

                        det_rows.append({
                            "frame_idx": frame_idx,
                            "timestamp_s": timestamp_s,
                            "track_id": int(track_id),
                            "bbox_x1": float(x1), "bbox_y1": float(y1),
                            "bbox_x2": float(x2), "bbox_y2": float(y2),
                            "foot_px_x": float(foot_px_x), "foot_px_y": float(foot_px_y),
                            "x_m": x_m, "y_m": y_m,
                            "zone": zone,
                        })

                    if USE_TWO_SIDED_SLOTS:
                        assign_stable_ids_two_sided(frame_idx, det_rows, near_assigner, far_assigner)
                    else:
                        assign_stable_ids_single_assigner(frame_idx, det_rows, all_assigner)

                    H_img, W_img = frame.shape[:2]
                    do_pose_this_frame = (frame_idx % int(max(1, POSE_EVERY_N_FRAMES)) == 0)

                    # per detection
                    for r, xyxy in zip(det_rows, detections.xyxy):
                        sid = int(r.get("stable_id", -1))
                        r.setdefault("side_group", "UNK")

                        # init state if missing
                        if sid > 0 and sid not in live_states:
                            live_states[sid] = init_state(timestamp_s, r["x_m"], r["y_m"])

                        # motion
                        sp_kmh, dist_m, accel_mps2 = update_motion(live_states, sid, timestamp_s, r["x_m"], r["y_m"])
                        r["speed_kmh"] = float(sp_kmh)
                        r["distance_m"] = float(dist_m)
                        r["accel_mps2"] = float(accel_mps2)

                        # defaults
                        r["knee_angle_L"] = None
                        r["knee_angle_R"] = None
                        r["knee_angle_avg"] = None
                        r["ready_flag"] = 0
                        r["ready_pct"] = float(get_ready_pct(live_states, sid)) if sid > 0 else 0.0

                        # pose step
                        if sid > 0 and do_pose_this_frame:
                            x1, y1, x2, y2 = xyxy.astype(float)
                            bx1, by1, bx2, by2 = expand_bbox(x1, y1, x2, y2, W_img, H_img, frac=POSE_BBOX_EXPAND)
                            crop = frame[by1:by2, bx1:bx2]

                            if crop.size > 0:
                                crop_small, _ = resize_keep_aspect(crop, POSE_CROP_MAX_W, POSE_CROP_MAX_H)
                                crop_rgb = cv2.cvtColor(crop_small, cv2.COLOR_BGR2RGB)
                                pose_res = pose.process(crop_rgb)

                                knee_L, knee_R, knee_avg, hip_drop, ready_flag = compute_ready_from_pose(pose_res)

                                # ✅ speed gate (only "ready" if mostly still)
                                if ready_flag == 1 and r["speed_kmh"] > READY_MAX_SPEED_KMH:
                                    ready_flag = 0

                                update_ready_state(
                                    live_states, sid, frame_idx,
                                    knee_avg=knee_avg,
                                    hip_drop=hip_drop,
                                    ready_flag=ready_flag
                                )

                                # store raw + smoothed avg
                                st = live_states.get(sid)
                                r["knee_angle_L"] = knee_L
                                r["knee_angle_R"] = knee_R
                                r["knee_angle_avg"] = st.last_knee_angle if (st is not None and st.last_knee_angle is not None) else knee_avg
                                r["ready_flag"] = int(ready_flag)
                                r["ready_pct"] = float(get_ready_pct(live_states, sid))

                        # ✅ CACHE LAST VALUES when no pose (IMPLEMENTED)
                        if sid > 0:
                            st = live_states.get(sid)
                            if st is not None:
                                if r["knee_angle_avg"] is None:
                                    r["knee_angle_avg"] = st.last_knee_angle
                                # use last ready flag too (prevents flicker to 0 on non-pose frames)
                                if not do_pose_this_frame:
                                    r["ready_flag"] = int(st.last_ready_flag)
                                r["ready_pct"] = float(get_ready_pct(live_states, sid))

                        # Draw
                        ring_col = knee_to_bgr(r["knee_angle_avg"])
                        draw_player_overlay(
                            annotated,
                            xyxy,
                            stable_id=sid,
                            knee_angle_avg=r["knee_angle_avg"],
                            ready_pct=r["ready_pct"],
                            ring_color=ring_col,
                        )

                    # write rows
                    for r in det_rows:
                        row_out = dict(r)
                        row_out.setdefault("stable_id", -1)
                        row_out.setdefault("side_group", "UNK")
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

            # Court + guides
            cv2.polylines(annotated, [court_poly_px], isClosed=True, color=(255, 255, 0), thickness=2)
            try:
                draw_court_guides(annotated, H)
            except Exception:
                pass

            # Overlay info
            cv2.putText(
                annotated,
                f"Stable IDs: {'2+2' if USE_TWO_SIDED_SLOTS else '1..4'} | conf={CONF} iou={IOU} imgsz={IMGSZ}",
                (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2, cv2.LINE_AA,
            )
            cv2.putText(
                annotated,
                f"Ready: knee<={READY_KNEE_ANGLE_MAX:.0f} hip>={READY_HIP_DROP_MIN:.2f} speed<={READY_MAX_SPEED_KMH:.1f} | pose_every={POSE_EVERY_N_FRAMES}",
                (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA,
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
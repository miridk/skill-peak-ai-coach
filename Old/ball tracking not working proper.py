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


# ----------------------------
# CONFIG
# ----------------------------
VIDEO_PATH = "input.mp4"
SAVE_DIR = "save"
EXPORT_DIR = "exports"

# Navn for kameravinkel/hal, så du genbruger calibration
COURT_SETUP_NAME = "test1"
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
COURT_MARGIN_PX = 75  # polygon test margin
CONF = 0.45
IOU = 0.6
IMGSZ = 960

# Stable ID settings
USE_TWO_SIDED_SLOTS = True  # ✅ anbefalet når du filmer bag baseline
MAX_AGE_SECONDS = 1.8       # hvor længe en spiller "holdes i live" ved overlap/occlusion
MAX_MATCH_DIST_M = 2.0      # max distance i meter for match til samme spiller-slot (tune 1.8-2.6)

# Live stats overlay settings
SPEED_SMOOTH_ALPHA = 0.35   # EMA smoothing (0..1). Higher = more reactive, more jitter
SPEED_CAP_KMH = 45.0        # cap to avoid spikes (badminton sprint bursts typisk < 35-40 km/h)
DIST_JUMP_CAP_M = 2.5       # ignore single-frame jumps > this (id swap / noise)

# Acceleration overlay settings
ACCEL_SMOOTH_ALPHA = 0.45   # EMA smoothing for acceleration (0..1)
ACCEL_CAP_MPS2 = 7.0        # cap (m/s^2). Tune 5-9 for badminton
ACCEL_DEADZONE = 0.35       # m/s^2 deadzone around 0 (reduces flicker)

# Ring style (base fallback)
RING_COLOR = (0, 255, 0)      # BGR grøn
RING_ALPHA = 0.45             # 0..1 (lavere = mere gennemsigtig)
RING_THICKNESS = 3
RING_SHADOW_ALPHA = 0.25      # shadow transparency

# Panel transparency
PANEL_ALPHA = 0.20   # 0.20 = meget gennemsigtig, 0.35 = medium

# Panel layout
PANEL_WIDTH = 160
PANEL_HEIGHT = 62
PANEL_ALPHA = 0.28

# Ring position + open arc settings
RING_Y_OFFSET_PX = 18
RING_ARC_SAMPLES = 64

# ----------------------------
# ✅ SHUTTLE TRACKING (PIXEL ONLY)
# ----------------------------
ENABLE_SHUTTLE_TRACKING = True

SHUTTLE_DIFF_THRESH = 16
SHUTTLE_BRIGHT_MIN = 200
SHUTTLE_MIN_AREA = 2
SHUTTLE_MAX_AREA = 160

SHUTTLE_ERODE_ITERS = 1
SHUTTLE_DILATE_ITERS = 1

SHUTTLE_GATE_PX = 160
SHUTTLE_MAX_MISSES = 25

# Player masking (vigtigt!)
SHUTTLE_MASK_PLAYERS = True
SHUTTLE_PLAYER_PAD_PX = 26        # ✅ lidt større for at undgå arm/racket highlights
SHUTTLE_MASK_FOOT_ONLY = False    # ✅ IMPORTANT: mask whole player bbox
SHUTTLE_FOOT_ZONE_FRAC = 0.55     # (kun relevant hvis MASK_FOOT_ONLY=True)

# Draw
SHUTTLE_DRAW = True
SHUTTLE_TRAIL_LEN = 45
SHUTTLE_POINT_RADIUS = 6

# ✅ Allow detection even if shuttle overlaps a player (unmask near prediction)
SHUTTLE_UNMASK_PRED_RADIUS = 90   # px (tune 70-140)

# ✅ Candidate shape limits (filters false positives from arms/rackets)
SHUTTLE_MAX_WH = 26               # max width/height of blob bbox
SHUTTLE_MAX_ASPECT = 2.6          # max(w/h, h/w)

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
    out = cv2.perspectiveTransform(pt, H.astype(np.float32))[0, 0]
    return float(out[0]), float(out[1])


def meters_to_px(H: np.ndarray, x_m: float, y_m: float) -> tuple[int, int]:
    Hinv = np.linalg.inv(H)
    pt = np.array([[[x_m, y_m]]], dtype=np.float32)
    out = cv2.perspectiveTransform(pt, Hinv.astype(np.float32))[0, 0]
    return int(round(out[0])), int(round(out[1]))


def draw_court_guides(frame: np.ndarray, H: np.ndarray) -> None:
    mid_y = COURT_L / 2.0
    short = 1.98

    y_net = mid_y
    y_far_short = float(np.clip(mid_y - short, 0.0, COURT_L))
    y_near_short = float(np.clip(mid_y + short, 0.0, COURT_L))

    def line_pts(xa, ya, xb, yb):
        p1 = meters_to_px(H, xa, ya)
        p2 = meters_to_px(H, xb, yb)
        return p1, p2

    p1, p2 = line_pts(0.0, y_net, COURT_W, y_net)
    cv2.line(frame, p1, p2, (0, 255, 255), 3, cv2.LINE_AA)

    p1s, p2s = line_pts(0.0, y_far_short, COURT_W, y_far_short)
    cv2.line(frame, p1s, p2s, (255, 255, 0), 2, cv2.LINE_AA)
    p1s2, p2s2 = line_pts(0.0, y_near_short, COURT_W, y_near_short)
    cv2.line(frame, p1s2, p2s2, (255, 255, 0), 2, cv2.LINE_AA)

    p1c, p2c = line_pts(COURT_W / 2.0, 0.0, COURT_W / 2.0, COURT_L)
    cv2.line(frame, p1c, p2c, (255, 0, 255), 2, cv2.LINE_AA)

    cv2.putText(frame, "NET", (min(p1[0], p2[0]) + 8, min(p1[1], p2[1]) - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(frame, "SHORT", (min(p1s[0], p2s[0]) + 8, min(p1s[1], p2s[1]) - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 0), 2, cv2.LINE_AA)


def classify_zone(x_m: float, y_m: float) -> str:
    mid = COURT_L / 2.0
    side_group = "FAR" if y_m < mid else "NEAR"
    dist_to_net = abs(y_m - mid)
    depth = "FRONT" if dist_to_net <= 1.98 else "BACK"
    lr = "LEFT" if x_m < COURT_W / 2.0 else "RIGHT"
    return f"{side_group}-{depth}-{lr}"


# ----------------------------
# PRO-STYLE DRAWING HELPERS
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


def accel_to_bgr(accel_mps2: float) -> Tuple[int, int, int]:
    a = float(np.clip(accel_mps2, -ACCEL_CAP_MPS2, ACCEL_CAP_MPS2))

    if abs(a) < ACCEL_DEADZONE:
        return (0, 255, 0)

    if a > 0:
        t = float(np.clip(a / ACCEL_CAP_MPS2, 0.0, 1.0))
        g = int(round(255 * (1.0 - t)))
        r = int(round(255 * t))
        return (0, g, r)
    else:
        t = float(np.clip((-a) / ACCEL_CAP_MPS2, 0.0, 1.0))
        g = int(round(255 * (1.0 - t)))
        b = int(round(255 * t))
        return (b, g, 0)


def draw_player_overlay(
    frame: np.ndarray,
    bbox_xyxy: np.ndarray,
    stable_id: int,
    speed_kmh: Optional[float],
    distance_m: Optional[float],
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
    draw_open_ellipse_arc(
        overlay,
        center=shadow_center,
        axes=(rx, ry),
        start_rad=arc_start,
        end_rad=arc_end,
        color=(0, 0, 0),
        thickness=8,
        samples=RING_ARC_SAMPLES,
    )
    overlay_alpha(frame, overlay, alpha=RING_SHADOW_ALPHA)

    overlay = frame.copy()
    draw_open_ellipse_arc(
        overlay,
        center=(cx, ring_y),
        axes=(rx, ry),
        start_rad=arc_start,
        end_rad=arc_end,
        color=ring_color,
        thickness=RING_THICKNESS,
        samples=RING_ARC_SAMPLES,
    )
    overlay_alpha(frame, overlay, alpha=ring_alpha)

    panel_w = PANEL_WIDTH
    panel_h = PANEL_HEIGHT

    px1 = int(cx - panel_w / 2)
    py1 = int(foot_y + 10)
    px2 = px1 + panel_w
    py2 = py1 + panel_h

    H_img, W_img = frame.shape[:2]
    shift_y = 0
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

    if speed_kmh is not None:
        cv2.putText(frame, f"{speed_kmh:.2f} km/h", (px1 + 10, py1 + 44),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (10, 10, 10), 2, cv2.LINE_AA)

    if distance_m is not None:
        cv2.putText(frame, f"{distance_m:.2f} m", (px1 + 10, py1 + 60),
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
# LIVE STATS (speed + distance + acceleration) per stable_id
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


def update_live_stats(states: Dict[int, LiveState], stable_id: int, t: float, x_m: float, y_m: float) -> Tuple[float, float, float]:
    if stable_id <= 0:
        return 0.0, 0.0, 0.0

    st = states.get(stable_id)
    if st is None:
        states[stable_id] = LiveState(
            last_t=t, last_x=x_m, last_y=y_m,
            speed_kmh_ema=0.0,
            speed_mps_ema=0.0,
            accel_mps2_ema=0.0,
            prev_speed_mps=0.0,
            total_dist_m=0.0
        )
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


# ----------------------------
# ✅ SHUTTLE TRACKING (PIXEL ONLY) - IMPLEMENTATION
# ----------------------------
class Kalman2D:
    """Constant-velocity Kalman in pixels: state [x,y,vx,vy], meas [x,y]."""
    def __init__(self):
        self.kf = cv2.KalmanFilter(4, 2)
        self.kf.transitionMatrix = np.array(
            [[1, 0, 1, 0],
             [0, 1, 0, 1],
             [0, 0, 1, 0],
             [0, 0, 0, 1]], dtype=np.float32
        )
        self.kf.measurementMatrix = np.array(
            [[1, 0, 0, 0],
             [0, 1, 0, 0]], dtype=np.float32
        )
        self.kf.processNoiseCov = np.eye(4, dtype=np.float32) * 1e-2
        self.kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * 10.0
        self.kf.errorCovPost = np.eye(4, dtype=np.float32) * 800.0
        self.inited = False

    def init(self, x: float, y: float):
        self.kf.statePost = np.array([[x], [y], [0], [0]], dtype=np.float32)
        self.inited = True

    def predict(self) -> Tuple[float, float]:
        p = self.kf.predict()
        return float(p[0, 0]), float(p[1, 0])

    def update(self, x: float, y: float) -> Tuple[float, float]:
        m = np.array([[x], [y]], dtype=np.float32)
        s = self.kf.correct(m)
        return float(s[0, 0]), float(s[1, 0])


def build_allowed_mask(shape_hw: Tuple[int, int],
                       player_bboxes_xyxy: List[Tuple[float, float, float, float]]) -> np.ndarray:
    """255=allowed, 0=masked."""
    h, w = shape_hw
    allowed = np.ones((h, w), dtype=np.uint8) * 255

    if not SHUTTLE_MASK_PLAYERS or not player_bboxes_xyxy:
        return allowed

    for x1, y1, x2, y2 in player_bboxes_xyxy:
        x1p = int(max(0, math.floor(x1) - SHUTTLE_PLAYER_PAD_PX))
        y1p = int(max(0, math.floor(y1) - SHUTTLE_PLAYER_PAD_PX))
        x2p = int(min(w - 1, math.ceil(x2) + SHUTTLE_PLAYER_PAD_PX))
        y2p = int(min(h - 1, math.ceil(y2) + SHUTTLE_PLAYER_PAD_PX))

        if SHUTTLE_MASK_FOOT_ONLY:
            foot_y1 = int(y1p + (1.0 - SHUTTLE_FOOT_ZONE_FRAC) * (y2p - y1p))
            cv2.rectangle(allowed, (x1p, foot_y1), (x2p, y2p), 0, -1)
        else:
            # ✅ mask whole player bbox
            cv2.rectangle(allowed, (x1p, y1p), (x2p, y2p), 0, -1)

    return allowed


def detect_shuttle_candidates(prev_gray: np.ndarray,
                             gray: np.ndarray,
                             allowed_mask: Optional[np.ndarray]) -> List[Tuple[float, float, float]]:
    """Return [(cx,cy,score)] based on motion + brightness + small blob."""
    diff = cv2.absdiff(gray, prev_gray)
    _, mask = cv2.threshold(diff, SHUTTLE_DIFF_THRESH, 255, cv2.THRESH_BINARY)

    if allowed_mask is not None:
        mask = cv2.bitwise_and(mask, allowed_mask)

    if SHUTTLE_ERODE_ITERS > 0:
        mask = cv2.erode(mask, None, iterations=SHUTTLE_ERODE_ITERS)
    if SHUTTLE_DILATE_ITERS > 0:
        mask = cv2.dilate(mask, None, iterations=SHUTTLE_DILATE_ITERS)

    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    cands = []
    for c in cnts:
        area = cv2.contourArea(c)
        if area < SHUTTLE_MIN_AREA or area > SHUTTLE_MAX_AREA:
            continue

        x, y, w, h = cv2.boundingRect(c)

        # ✅ NEW: size + aspect filters (kills arms/racket highlights)
        if w > SHUTTLE_MAX_WH or h > SHUTTLE_MAX_WH:
            continue
        aspect = max(w / max(1, h), h / max(1, w))
        if aspect > SHUTTLE_MAX_ASPECT:
            continue

        cx = x + w / 2.0
        cy = y + h / 2.0

        patch = gray[max(0, y - 1):min(gray.shape[0], y + h + 1),
                     max(0, x - 1):min(gray.shape[1], x + w + 1)]
        if patch.size == 0:
            continue

        bright = float(np.percentile(patch, 92))
        if bright < SHUTTLE_BRIGHT_MIN:
            continue

        score = (bright / 255.0) * (1.0 / (1.0 + 0.02 * area))
        cands.append((cx, cy, score))

    return cands


class ShuttleTrackerPx:
    def __init__(self):
        self.kf = Kalman2D()
        self.prev_gray: Optional[np.ndarray] = None
        self.misses = 0
        self.trail: List[Tuple[int, int]] = []

    def reset(self):
        self.kf = Kalman2D()
        self.prev_gray = None
        self.misses = 0
        self.trail.clear()

    def update(self, frame: np.ndarray, player_bboxes_xyxy: List[Tuple[float, float, float, float]]):
        out = {
            "shuttle_px_x": float("nan"),
            "shuttle_px_y": float("nan"),
            "shuttle_visible": 0,
            "shuttle_conf": 0.0,
        }

        if not ENABLE_SHUTTLE_TRACKING:
            return out

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if self.prev_gray is None:
            self.prev_gray = gray
            return out

        allowed = build_allowed_mask(gray.shape, player_bboxes_xyxy)

        pred = None
        if self.kf.inited:
            pred = self.kf.predict()

            # ✅ NEW: unmask around prediction so we can detect shuttle even if it overlaps players
            px, py = int(round(pred[0])), int(round(pred[1]))
            if 0 <= px < gray.shape[1] and 0 <= py < gray.shape[0]:
                cv2.circle(allowed, (px, py), SHUTTLE_UNMASK_PRED_RADIUS, 255, -1)

        cands = detect_shuttle_candidates(self.prev_gray, gray, allowed)

        best = None
        best_score = -1e9
        for cx, cy, base in cands:
            if pred is not None:
                d = float(np.hypot(cx - pred[0], cy - pred[1]))
                if d > SHUTTLE_GATE_PX:
                    continue
                score = base * (1.0 / (1.0 + 0.03 * d))
            else:
                score = base

            if score > best_score:
                best_score = score
                best = (cx, cy, score)

        if best is not None:
            cx, cy, score = best
            conf = float(np.clip(score * 2.0, 0.0, 1.0))
            self.misses = 0

            if not self.kf.inited:
                self.kf.init(cx, cy)
                sx, sy = cx, cy
            else:
                sx, sy = self.kf.update(cx, cy)

            out["shuttle_px_x"] = float(sx)
            out["shuttle_px_y"] = float(sy)
            out["shuttle_visible"] = 1
            out["shuttle_conf"] = conf

            self.trail.append((int(round(sx)), int(round(sy))))
            self.trail = self.trail[-SHUTTLE_TRAIL_LEN:]
        else:
            if self.kf.inited:
                self.misses += 1
                px, py = self.kf.predict()
                out["shuttle_px_x"] = float(px)
                out["shuttle_px_y"] = float(py)
                out["shuttle_visible"] = 0
                out["shuttle_conf"] = 0.0

                if self.misses > SHUTTLE_MAX_MISSES:
                    self.reset()

        self.prev_gray = gray
        return out


def draw_shuttle_overlay_px(frame: np.ndarray, trail: List[Tuple[int, int]], visible: int, conf: float):
    if not SHUTTLE_DRAW:
        return

    for i in range(1, len(trail)):
        cv2.line(frame, trail[i - 1], trail[i], (0, 0, 255), 2, cv2.LINE_AA)

    if trail:
        cv2.circle(frame, trail[-1], SHUTTLE_POINT_RADIUS, (0, 0, 255), -1, cv2.LINE_AA)

    cv2.putText(frame, f"Shuttle(px): vis={visible} conf={conf:.2f}",
                (20, 98), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)


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

            "shuttle_px_x", "shuttle_px_y",
            "shuttle_visible", "shuttle_conf",
        ],
    )
    csv_w.writeheader()
    jsonl_f = open(jsonl_path, "w", encoding="utf-8")

    print("Loading model…")
    model = YOLO("yolo11m.pt")
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
    shuttle_px = ShuttleTrackerPx()

    print(f"💾 Saving annotated video to: {out_path}")
    print(f"📦 Exporting detections to:\n   CSV:   {csv_path}\n   JSONL: {jsonl_path}")
    print(f"   size={width}x{height}, fps={fps:.2f}")
    print(f"   YOLO params: conf={CONF}, iou={IOU}, imgsz={IMGSZ}")

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
            player_bboxes_xyxy: List[Tuple[float, float, float, float]] = []

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
                    for xyxy in detections.xyxy:
                        x1, y1, x2, y2 = xyxy.astype(float).tolist()
                        player_bboxes_xyxy.append((x1, y1, x2, y2))

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

                    for r, xyxy in zip(det_rows, detections.xyxy):
                        sid = int(r.get("stable_id", -1))
                        r.setdefault("side_group", "UNK")

                        sp_kmh, dist_m, accel_mps2 = update_live_stats(
                            live_states, sid, timestamp_s, r["x_m"], r["y_m"]
                        )
                        r["speed_kmh"] = float(sp_kmh)
                        r["distance_m"] = float(dist_m)
                        r["accel_mps2"] = float(accel_mps2)

                        ring_col = accel_to_bgr(accel_mps2)

                        draw_player_overlay(
                            annotated,
                            xyxy,
                            stable_id=sid,
                            speed_kmh=r["speed_kmh"],
                            distance_m=r["distance_m"],
                            ring_color=ring_col,
                        )

            # ✅ Shuttle tracking (pixels only)
            sh = shuttle_px.update(frame, player_bboxes_xyxy)
            draw_shuttle_overlay_px(annotated, shuttle_px.trail, sh["shuttle_visible"], sh["shuttle_conf"])

            cv2.polylines(annotated, [court_poly_px], isClosed=True, color=(255, 255, 0), thickness=2)
            try:
                draw_court_guides(annotated, H)
            except Exception:
                pass

            cv2.putText(
                annotated,
                f"Stable IDs: {'2+2' if USE_TWO_SIDED_SLOTS else '1..4'} | conf={CONF} iou={IOU} imgsz={IMGSZ}",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.75,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                annotated,
                f"Ring: red=accel blue=decel | accel_cap={ACCEL_CAP_MPS2:.1f} deadzone={ACCEL_DEADZONE:.2f}",
                (20, 70),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

            # Export
            if det_rows:
                for r in det_rows:
                    row_out = dict(r)
                    row_out.setdefault("stable_id", -1)
                    row_out.setdefault("side_group", "UNK")
                    row_out.setdefault("speed_kmh", 0.0)
                    row_out.setdefault("distance_m", 0.0)
                    row_out.setdefault("accel_mps2", 0.0)

                    row_out["shuttle_px_x"] = sh["shuttle_px_x"]
                    row_out["shuttle_px_y"] = sh["shuttle_px_y"]
                    row_out["shuttle_visible"] = sh["shuttle_visible"]
                    row_out["shuttle_conf"] = sh["shuttle_conf"]

                    csv_w.writerow(row_out)
                    jsonl_f.write(json.dumps(row_out, ensure_ascii=False) + "\n")
            else:
                row_out = {
                    "frame_idx": frame_idx,
                    "timestamp_s": timestamp_s,
                    "track_id": -1,
                    "stable_id": -1,
                    "side_group": "NONE",
                    "bbox_x1": float("nan"), "bbox_y1": float("nan"),
                    "bbox_x2": float("nan"), "bbox_y2": float("nan"),
                    "foot_px_x": float("nan"), "foot_px_y": float("nan"),
                    "x_m": float("nan"), "y_m": float("nan"),
                    "zone": "N/A",
                    "speed_kmh": 0.0, "distance_m": 0.0, "accel_mps2": 0.0,

                    "shuttle_px_x": sh["shuttle_px_x"],
                    "shuttle_px_y": sh["shuttle_px_y"],
                    "shuttle_visible": sh["shuttle_visible"],
                    "shuttle_conf": sh["shuttle_conf"],
                }
                csv_w.writerow(row_out)
                jsonl_f.write(json.dumps(row_out, ensure_ascii=False) + "\n")

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
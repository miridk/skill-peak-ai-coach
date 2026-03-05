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

COURT_SETUP_NAME = "almind_viuf_hallen_baseline"
CALIB_DIR = "calibration"
CALIB_FILE = os.path.join(CALIB_DIR, f"{COURT_SETUP_NAME}.json")

SAVE_FPS_FALLBACK = 30.0

COURT_W = 6.10
COURT_L = 13.40

COURT_DST = np.array(
    [[0.0, 0.0], [COURT_W, 0.0], [COURT_W, COURT_L], [0.0, COURT_L]],
    dtype=np.float32,
)

COURT_MARGIN_PX = 75
CONF = 0.45
IOU = 0.6
IMGSZ = 960

USE_TWO_SIDED_SLOTS = True
MAX_AGE_SECONDS = 1.8
MAX_MATCH_DIST_M = 2.0

SPEED_SMOOTH_ALPHA = 0.35
SPEED_CAP_KMH = 45.0
DIST_JUMP_CAP_M = 2.5

# Ring style
RING_COLOR = (0, 255, 0)      # BGR grøn
RING_ALPHA = 0.45
RING_THICKNESS = 3
RING_SHADOW_ALPHA = 0.25

# Ring sizing
RING_MARGIN_PX = 12
MIN_RING_DIAMETER_PX = 20

# MediaPipe Pose
POSE_MODEL_COMPLEXITY = 1
POSE_MIN_DET_CONF = 0.5
POSE_MIN_TRK_CONF = 0.5
POSE_VIS_TH = 0.5
POSE_CROP_PAD = 0.15
POSE_EVERY_N_FRAMES = 1  # sæt 2 hvis det bliver langsomt

# ---- NEW: Pose stabilization + gating ----
POSE_CENTER_ALPHA = 0.18      # lavere = mere stabil (0.15-0.35)
POSE_SPAN_ALPHA = 0.22
POSE_MAX_JUMP_PX = 35         # ignorér pose-opdateringer der hopper > N px
POSE_MIN_SPAN_PX = 12
POSE_MAX_SPAN_PX = 260

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


def classify_zone(x_m: float, y_m: float) -> str:
    side = "LEFT" if x_m < COURT_W / 2 else "RIGHT"
    half = "FRONT" if y_m < COURT_L / 2 else "BACK"
    return f"{half}-{side}"


def clamp(v, lo, hi):
    return max(lo, min(hi, v))


# ----------------------------
# NEW: Boundary clamp helpers
# ----------------------------
def inside_poly(pt_xy: Tuple[float, float], poly: np.ndarray, margin_px: int) -> bool:
    return cv2.pointPolygonTest(poly, (float(pt_xy[0]), float(pt_xy[1])), True) >= -margin_px


def nearest_point_on_segment(px, py, ax, ay, bx, by):
    abx, aby = bx - ax, by - ay
    apx, apy = px - ax, py - ay
    denom = abx * abx + aby * aby
    if denom <= 1e-9:
        return ax, ay
    t = (apx * abx + apy * aby) / denom
    t = max(0.0, min(1.0, t))
    return ax + t * abx, ay + t * aby


def clamp_point_to_polygon(pt_xy: Tuple[float, float], poly: np.ndarray) -> Tuple[float, float]:
    px, py = float(pt_xy[0]), float(pt_xy[1])
    best = (px, py)
    best_d2 = 1e18
    n = len(poly)
    if n < 2:
        return px, py
    for i in range(n):
        ax, ay = float(poly[i][0]), float(poly[i][1])
        bx, by = float(poly[(i + 1) % n][0]), float(poly[(i + 1) % n][1])
        qx, qy = nearest_point_on_segment(px, py, ax, ay, bx, by)
        d2 = (qx - px) ** 2 + (qy - py) ** 2
        if d2 < best_d2:
            best_d2 = d2
            best = (qx, qy)
    return best


# ----------------------------
# NEW: Pose stabilizer per stable_id
# ----------------------------
@dataclass
class PoseStable:
    has_value: bool
    cx: float
    cy: float
    span: float


def ema(a: float, b: float, alpha: float) -> float:
    return alpha * b + (1.0 - alpha) * a


def update_pose_stable(
    store: Dict[int, PoseStable],
    stable_id: int,
    raw_cx: float,
    raw_cy: float,
    raw_span: float,
    alpha_center: float,
    alpha_span: float,
    max_jump_px: float,
) -> PoseStable:
    if stable_id <= 0:
        return PoseStable(False, raw_cx, raw_cy, raw_span)

    st = store.get(stable_id)
    if st is None or not st.has_value:
        st = PoseStable(True, raw_cx, raw_cy, raw_span)
        store[stable_id] = st
        return st

    jump = math.hypot(raw_cx - st.cx, raw_cy - st.cy)
    if jump > max_jump_px:
        # ignore raw update
        return st

    st.cx = ema(st.cx, raw_cx, alpha_center)
    st.cy = ema(st.cy, raw_cy, alpha_center)
    st.span = ema(st.span, raw_span, alpha_span)
    st.has_value = True
    store[stable_id] = st
    return st


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


def draw_player_overlay(
    frame: np.ndarray,
    bbox_xyxy: np.ndarray,
    stable_id: int,
    speed_kmh: Optional[float],
    distance_m: Optional[float],
    ring_span_px: Optional[float] = None,
    foot_center_px: Optional[Tuple[float, float]] = None,
):
    x1, y1, x2, y2 = bbox_xyxy.astype(int).tolist()
    w = max(1, x2 - x1)

    if foot_center_px is not None:
        cx = int(foot_center_px[0])
        foot_y = int(foot_center_px[1])
    else:
        cx = int((x1 + x2) / 2)
        foot_y = int(y2)

    if ring_span_px is None or not np.isfinite(ring_span_px):
        ring_span_px = float(w) * 0.55

    diameter = float(ring_span_px) + float(RING_MARGIN_PX * 2)
    diameter = max(diameter, float(MIN_RING_DIAMETER_PX))

    rx = int(diameter / 2)
    ry = int(diameter * 0.35)

    # Shadow
    overlay = frame.copy()
    shadow_center = (cx, foot_y + int(ry * 0.45))
    cv2.ellipse(overlay, shadow_center, (rx, ry), 0, 0, 360, (0, 0, 0), 8)
    overlay_alpha(frame, overlay, alpha=RING_SHADOW_ALPHA)

    # Ring
    overlay = frame.copy()
    cv2.ellipse(overlay, (cx, foot_y), (rx, ry), 0, 0, 360, RING_COLOR, RING_THICKNESS)
    overlay_alpha(frame, overlay, alpha=RING_ALPHA)

    # Panel
    panel_w = int(max(110, min(190, w * 1.25)))
    panel_h = 62

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
    draw_rounded_rect(overlay, px1, py1, px2, py2, (235, 235, 235), thickness=-1, radius=12)
    overlay_alpha(frame, overlay, alpha=0.65)

    cv2.putText(frame, f"{stable_id}", (px1 + 10, py1 + 26),
                cv2.FONT_HERSHEY_SIMPLEX, 0.95, (10, 10, 10), 3, cv2.LINE_AA)

    if speed_kmh is not None:
        cv2.putText(frame, f"{speed_kmh:.2f} km/h", (px1 + 10, py1 + 44),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (10, 10, 10), 2, cv2.LINE_AA)

    if distance_m is not None:
        cv2.putText(frame, f"{distance_m:.2f} m", (px1 + 10, py1 + 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (10, 10, 10), 2, cv2.LINE_AA)


# ----------------------------
# MediaPipe Pose helpers
# ----------------------------
def _get_mp_solutions():
    """
    MediaPipe packaging can differ. Try multiple import paths for 'solutions'.
    Returns module with .pose and .pose.PoseLandmark.
    """
    try:
        s = mp.solutions
        if hasattr(s, "pose"):
            return s
    except Exception:
        pass

    try:
        import mediapipe.solutions as s  # type: ignore
        if hasattr(s, "pose"):
            return s
    except Exception:
        pass

    raise RuntimeError(
        "MediaPipe er installeret, men 'solutions.pose' kan ikke importeres i denne build.\n"
        "Prøv at installere: pip uninstall mediapipe && pip install mediapipe==0.10.14\n"
        f"mp.__file__={getattr(mp,'__file__',None)} version={getattr(mp,'__version__','?')}"
    )


def run_pose_on_crop(pose, mp_solutions, frame_bgr: np.ndarray, xyxy: np.ndarray, pad: float = 0.15):
    H, W = frame_bgr.shape[:2]
    x1, y1, x2, y2 = [float(v) for v in xyxy]
    w = x2 - x1
    h = y2 - y1

    px = w * pad
    py = h * pad

    cx1 = int(clamp(x1 - px, 0, W - 1))
    cy1 = int(clamp(y1 - py, 0, H - 1))
    cx2 = int(clamp(x2 + px, 0, W - 1))
    cy2 = int(clamp(y2 + py, 0, H - 1))

    crop = frame_bgr[cy1:cy2, cx1:cx2]
    if crop.size == 0:
        return None

    crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    res = pose.process(crop_rgb)
    if not res.pose_landmarks:
        return None

    lm = res.pose_landmarks.landmark
    ch, cw = crop.shape[:2]

    def to_frame_xy(idx):
        x = lm[idx].x * cw + cx1
        y = lm[idx].y * ch + cy1
        vis = lm[idx].visibility
        return (float(x), float(y), float(vis))

    PL = mp_solutions.pose.PoseLandmark
    la = to_frame_xy(PL.LEFT_ANKLE.value)
    ra = to_frame_xy(PL.RIGHT_ANKLE.value)
    lh = to_frame_xy(PL.LEFT_HEEL.value)
    rh = to_frame_xy(PL.RIGHT_HEEL.value)
    lhip = to_frame_xy(PL.LEFT_HIP.value)
    rhip = to_frame_xy(PL.RIGHT_HIP.value)
    mid_hip = ((lhip[0] + rhip[0]) / 2.0, (lhip[1] + rhip[1]) / 2.0, min(lhip[2], rhip[2]))

    return {
        "left_ankle": la,
        "right_ankle": ra,
        "left_heel": lh,
        "right_heel": rh,
        "mid_hip": mid_hip,
    }


def best_foot_point(pose_pts, fallback_cx, fallback_y2, vis_th=0.5):
    if pose_pts:
        lh = pose_pts["left_heel"]
        rh = pose_pts["right_heel"]
        la = pose_pts["left_ankle"]
        ra = pose_pts["right_ankle"]

        if lh[2] >= vis_th and rh[2] >= vis_th:
            return ((lh[0] + rh[0]) / 2.0, (lh[1] + rh[1]) / 2.0, "heels")
        if la[2] >= vis_th and ra[2] >= vis_th:
            return ((la[0] + ra[0]) / 2.0, (la[1] + ra[1]) / 2.0, "ankles")

    return (float(fallback_cx), float(fallback_y2), "bbox")


def foot_span_pixels(pose_pts, fallback_w, vis_th=0.5):
    if pose_pts:
        la = pose_pts["left_ankle"]
        ra = pose_pts["right_ankle"]
        if la[2] >= vis_th and ra[2] >= vis_th:
            return float(math.hypot(la[0] - ra[0], la[1] - ra[1]))
    return float(fallback_w) * 0.55


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
# LIVE STATS
# ----------------------------
@dataclass
class LiveState:
    last_t: float
    last_x: float
    last_y: float
    speed_kmh_ema: float
    total_dist_m: float


def update_live_stats(states: Dict[int, LiveState], stable_id: int, t: float, x_m: float, y_m: float) -> Tuple[float, float]:
    if stable_id <= 0:
        return 0.0, 0.0

    st = states.get(stable_id)
    if st is None:
        states[stable_id] = LiveState(last_t=t, last_x=x_m, last_y=y_m, speed_kmh_ema=0.0, total_dist_m=0.0)
        return 0.0, 0.0

    dt = t - st.last_t
    if dt <= 1e-6:
        return st.speed_kmh_ema, st.total_dist_m

    dx = x_m - st.last_x
    dy = y_m - st.last_y
    step = math.sqrt(dx * dx + dy * dy)

    if step <= DIST_JUMP_CAP_M:
        st.total_dist_m += step
        speed_mps = step / dt
        speed_kmh = float(min(max(speed_mps * 3.6, 0.0), SPEED_CAP_KMH))
        st.speed_kmh_ema = SPEED_SMOOTH_ALPHA * speed_kmh + (1.0 - SPEED_SMOOTH_ALPHA) * st.speed_kmh_ema

    st.last_t = t
    st.last_x = x_m
    st.last_y = y_m

    return st.speed_kmh_ema, st.total_dist_m


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
            "foot_src",
            "foot_span_px",
            "pose_ok",
            "x_m", "y_m",
            "zone",
            "speed_kmh", "distance_m",
        ],
    )
    csv_w.writeheader()
    jsonl_f = open(jsonl_path, "w", encoding="utf-8")

    print("Loading model…")
    model = YOLO("yolo11m.pt")
    tracker = sv.ByteTrack()

    mp_solutions = _get_mp_solutions()
    mp_pose = mp_solutions.pose
    pose = mp_pose.Pose(
        static_image_mode=False,
        model_complexity=POSE_MODEL_COMPLEXITY,
        enable_segmentation=False,
        smooth_landmarks=True,
        min_detection_confidence=POSE_MIN_DET_CONF,
        min_tracking_confidence=POSE_MIN_TRK_CONF,
    )

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
    pose_stable_store: Dict[int, PoseStable] = {}

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

            if detections.class_id is not None and len(detections) > 0:
                detections = detections[detections.class_id == 0]  # persons

                # Pose-assisted in-bounds filtering (anchors) - keep conservative
                if len(detections) > 0:
                    keep = []
                    use_pose_this_frame = (frame_idx % POSE_EVERY_N_FRAMES == 0)

                    for xyxy in detections.xyxy:
                        x1, y1, x2, y2 = [float(v) for v in xyxy]
                        cx = int((x1 + x2) / 2)
                        cy = int((y1 + y2) / 2)

                        candidates = [(cx, int(y2)), (cx, cy)]
                        pose_pts = None

                        if use_pose_this_frame:
                            pose_pts = run_pose_on_crop(pose, mp_solutions, frame, xyxy, pad=POSE_CROP_PAD)

                        # prefer mid-hip if available (more stable than feet)
                        if pose_pts:
                            mh = pose_pts["mid_hip"]
                            if mh[2] >= 0.4:
                                candidates.insert(0, (int(mh[0]), int(mh[1])))

                        inside = any(point_in_polygon_margin(pt, court_poly_px, margin_px=COURT_MARGIN_PX) for pt in candidates)
                        keep.append(inside)

                    detections = detections[np.array(keep, dtype=bool)]

                # Track
                detections = tracker.update_with_detections(detections)

                if detections.tracker_id is not None and len(detections) > 0:
                    use_pose_this_frame = (frame_idx % POSE_EVERY_N_FRAMES == 0)

                    # First pass: collect RAW anchors + meters (temporary)
                    for xyxy, track_id in zip(detections.xyxy, detections.tracker_id):
                        x1, y1, x2, y2 = xyxy.astype(float)

                        pose_pts = None
                        if use_pose_this_frame:
                            pose_pts = run_pose_on_crop(pose, mp_solutions, frame, xyxy, pad=POSE_CROP_PAD)

                        fallback_cx = (x1 + x2) / 2.0
                        fallback_y2 = y2
                        bbox_span = float(x2 - x1) * 0.55

                        raw_foot_x, raw_foot_y, foot_src = best_foot_point(
                            pose_pts, fallback_cx, fallback_y2, vis_th=POSE_VIS_TH
                        )
                        raw_span = foot_span_pixels(pose_pts, fallback_w=(x2 - x1), vis_th=POSE_VIS_TH)

                        pose_ok = bool(pose_pts)
                        # validate span; otherwise fallback to bbox
                        if (not pose_ok) or (raw_span < POSE_MIN_SPAN_PX) or (raw_span > POSE_MAX_SPAN_PX):
                            raw_foot_x = float(fallback_cx)
                            raw_foot_y = float(fallback_y2)
                            raw_span = float(bbox_span)
                            foot_src = "bbox"
                            pose_ok = False

                        # temp meters (before stabilization); used for slot assignment split
                        x_m, y_m = px_to_meters(H, raw_foot_x, raw_foot_y)
                        x_m = float(np.clip(x_m, 0.0, COURT_W))
                        y_m = float(np.clip(y_m, 0.0, COURT_L))

                        det_rows.append({
                            "frame_idx": frame_idx,
                            "timestamp_s": timestamp_s,
                            "track_id": int(track_id),
                            "bbox_x1": float(x1), "bbox_y1": float(y1),
                            "bbox_x2": float(x2), "bbox_y2": float(y2),

                            # RAW (pre-stable)
                            "raw_foot_px_x": float(raw_foot_x),
                            "raw_foot_px_y": float(raw_foot_y),
                            "raw_span_px": float(raw_span),

                            # final fields (will be overwritten after stabilization + boundary)
                            "foot_px_x": float(raw_foot_x),
                            "foot_px_y": float(raw_foot_y),
                            "foot_span_px": float(raw_span),
                            "foot_src": str(foot_src),
                            "pose_ok": int(1 if pose_ok else 0),

                            "x_m": x_m, "y_m": y_m,
                            "zone": classify_zone(x_m, y_m),
                        })

                    # Assign stable ids
                    if USE_TWO_SIDED_SLOTS:
                        assign_stable_ids_two_sided(frame_idx, det_rows, near_assigner, far_assigner)
                    else:
                        assign_stable_ids_single_assigner(frame_idx, det_rows, all_assigner)

                    # Second pass: stabilize per stable_id + HARD boundary enforcement + recompute meters/zone
                    for r in det_rows:
                        sid = int(r.get("stable_id", -1))
                        r.setdefault("side_group", "UNK")

                        st = update_pose_stable(
                            pose_stable_store,
                            sid,
                            r["raw_foot_px_x"],
                            r["raw_foot_px_y"],
                            r["raw_span_px"],
                            alpha_center=POSE_CENTER_ALPHA,
                            alpha_span=POSE_SPAN_ALPHA,
                            max_jump_px=POSE_MAX_JUMP_PX,
                        )

                        # stabilized values
                        foot_x = float(st.cx)
                        foot_y = float(st.cy)
                        span_px = float(st.span)

                        # HARD boundary in pixel space
                        bbox_foot = (float((r["bbox_x1"] + r["bbox_x2"]) / 2.0), float(r["bbox_y2"]))
                        pose_foot = (foot_x, foot_y)

                        pose_inside = inside_poly(pose_foot, court_poly_px, margin_px=COURT_MARGIN_PX)
                        bbox_inside = inside_poly(bbox_foot, court_poly_px, margin_px=COURT_MARGIN_PX)

                        if not pose_inside:
                            if bbox_inside:
                                foot_x, foot_y = bbox_foot[0], bbox_foot[1]
                                r["foot_src"] = (r.get("foot_src", "") + "|snap_bbox").strip("|")
                                r["pose_ok"] = 0
                            else:
                                cx, cy = clamp_point_to_polygon(pose_foot, court_poly_px)
                                foot_x, foot_y = float(cx), float(cy)
                                r["foot_src"] = (r.get("foot_src", "") + "|clamp_poly").strip("|")
                                r["pose_ok"] = 0

                        # write final anchors
                        r["foot_px_x"] = float(foot_x)
                        r["foot_px_y"] = float(foot_y)
                        r["foot_span_px"] = float(span_px)

                        # recompute meters using final anchor
                        x_m, y_m = px_to_meters(H, r["foot_px_x"], r["foot_px_y"])
                        r["x_m"] = float(np.clip(x_m, 0.0, COURT_W))
                        r["y_m"] = float(np.clip(y_m, 0.0, COURT_L))
                        r["zone"] = classify_zone(r["x_m"], r["y_m"])

                    # Draw + stats + export (aligned with detections.xyxy order)
                    for r, xyxy in zip(det_rows, detections.xyxy):
                        sid = int(r.get("stable_id", -1))

                        sp_kmh, dist_m = update_live_stats(live_states, sid, timestamp_s, r["x_m"], r["y_m"])
                        r["speed_kmh"] = float(sp_kmh)
                        r["distance_m"] = float(dist_m)

                        draw_player_overlay(
                            annotated,
                            xyxy,
                            stable_id=sid,
                            speed_kmh=r["speed_kmh"],
                            distance_m=r["distance_m"],
                            ring_span_px=r.get("foot_span_px"),
                            foot_center_px=(r["foot_px_x"], r["foot_px_y"]),
                        )

                    # Export (drop raw keys from CSV/JSONL if you want; we keep them out)
                    for r in det_rows:
                        out_row = {k: v for k, v in r.items() if not k.startswith("raw_")}
                        csv_w.writerow(out_row)
                        jsonl_f.write(json.dumps(out_row, ensure_ascii=False) + "\n")

            # Debug court polygon overlay
            cv2.polylines(annotated, [court_poly_px], isClosed=True, color=(255, 255, 0), thickness=2)

            cv2.putText(
                annotated,
                f"Pose:{'ON' if (frame_idx % POSE_EVERY_N_FRAMES == 0) else 'SKIP'} | Stable:{'2+2' if USE_TWO_SIDED_SLOTS else '1..4'} | conf={CONF} iou={IOU} imgsz={IMGSZ}",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.72,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
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
        pose.close()
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
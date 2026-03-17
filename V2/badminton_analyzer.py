# =====================================================
# BADMINTON TRACKER - ULTIMATE VERSION (med lang memory direction)
# =====================================================
# Ændring: Velocity + direction prediction over 12 frames + 0.45 straf ved modsat retning
# Fixes: tilføjet manglende _direction_cost
# Alle definitioner før brug → ingen NameError
# Elipser, overlay, speed/dist/accel, ready %, csv/jsonl bevaret

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
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from ultralytics import YOLO
import supervision as sv
import mediapipe as mp
from mediapipe.tasks.python import vision as mp_vision
from mediapipe.tasks.python.vision import PoseLandmarker, PoseLandmarkerOptions
from mediapipe.tasks.python.core.base_options import BaseOptions

import tkinter as tk
from tkinter import filedialog


# ----------------------------
# CONFIG
# ----------------------------
# VIDEO_PATH = "input.mp4"
SAVE_DIR = "save"
EXPORT_DIR = "exports"

COURT_SETUP_NAME = "almind_viuf_hallen_baseline2"
CALIB_DIR = "calibration"
CALIB_FILE = os.path.join(CALIB_DIR, f"{COURT_SETUP_NAME}.json")

SAVE_FPS_FALLBACK = 30.0

COURT_W = 6.10
COURT_L = 13.40

COURT_DST = np.array(
    [[0.0, 0.0], [COURT_W, 0.0], [COURT_W, COURT_L], [0.0, COURT_L]],
    dtype=np.float32,
)

COURT_MARGIN_PX = 85
SIDE_BOTTOM_MARGIN_PX = 2
CONF = 0.45
IOU = 0.6
IMGSZ = 1088

PLAYER_IDS = [1, 2, 3, 4]
MAX_PLAYERS = 4

CLIP_MODEL_NAME = "ViT-B-32"
CLIP_PRETRAINED = "laion2b_s34b_b79k"
CLIP_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

MAX_AGE_SECONDS = 3.0
MAX_MATCH_DIST_M = 3.2

W_CLIP = 0.34
W_MOTION = 0.18
W_COLOR = 0.06
W_SIZE = 0.04
W_SIDE = 0.12
W_EDGE = 0.06
W_POSE = 0.20
W_DIRECTION = 0.28

SWITCH_MARGIN = 0.14
SWITCH_CONFIRM_FRAMES = 8
UNMATCHED_COST = 1.15

PAIR_SWAP_MAX_M = 1.45
PAIR_SWAP_IOU_THRESH = 0.015
PAIR_SWAP_COST_MARGIN = 0.24
PAIR_SWAP_HIT_MIN = 18
PAIR_SWAP_AMBIGUOUS_BONUS = 0.12

MAX_ACCEPT_COST = 0.72 #TODO PRØV AT SÆNGE DENNE
OUTSIDER_DIST_M = 1.75
OUTSIDER_CLIP_MIN_SIM = 0.58

CLIP_EMA_ALPHA = 0.14
COLOR_EMA_ALPHA = 0.16
VEL_ALPHA = 0.40
SIZE_ALPHA = 0.25
MEMORY_BANK_SIZE = 18
MEMORY_MATCH_TOPK = 5
POSE_EMA_ALPHA = 0.22
POSE_BANK_SIZE = 14

#TODO - I NEED TO PLAY WITH THESE VALUES
VELOCITY_HISTORY_FRAMES = 300
LONG_TERM_DIRECTION_FRAMES = 0.15
LINESPACE_FRAMES = -3 #how much weight to give to the history frames Less number = more weight to the old frame
DIRECTION_COST = 0.9 + 0.45 #Higer number = higher penalty

OCCLUSION_IOU_THRESH = 0.18
OCCLUSION_EXTRA_STICKINESS = 0.16
OCCLUSION_MOTION_BOOST = 0.08
OCCLUSION_CLIP_REDUCE = 0.10
OCCLUSION_FREEZE_BANK_UPDATE = True

SIDE_PENALTY = 0.22
SIDE_GRACE_FRAMES = 20
HARD_SIDE_PENALTY = 0.38
HARD_SIDE_HITS_MIN = 20

AMBIGUOUS_X_PX = 115
AMBIGUOUS_Y_PX = 265
AMBIGUOUS_IOU_THRESH = 0.03
AMBIGUOUS_EXTRA_SWITCH_MARGIN = 0.40
AMBIGUOUS_EXTRA_CONFIRM_FRAMES = 10
AMBIGUOUS_FREEZE_BANK_UPDATE = True
AMBIGUOUS_MAX_ACCEPT_COST = 0.42

ROLE_FRONT_BACK_MARGIN_M = 1.10
ROLE_SWAP_CONFIRM_FRAMES = 14
ROLE_HARD_PENALTY = 0.22
TEAMMATE_MIN_SEP_M = 0.45

POSE_NET_BOOST_FACTOR = 1.65
LATERAL_CROSS_WEIGHT = 0.95
NET_ROLE_EXTRA_PENALTY = 1.25
NET_DISTANCE_THRESHOLD_M = 2.80

RESCUE_MODE_ENABLED = True

SPEED_SMOOTH_ALPHA = 0.35
SPEED_CAP_KMH = 45.0
DIST_JUMP_CAP_M = 2.5
ACCEL_SMOOTH_ALPHA = 0.65
ACCEL_CAP_MPS2 = 11.0

RING_COLOR = (0, 255, 0)
RING_ALPHA = 0.45
RING_THICKNESS = 3
RING_SHADOW_ALPHA = 0.25
PANEL_ALPHA = 0.28
PANEL_WIDTH = 170
PANEL_HEIGHT = 74
RING_Y_OFFSET_PX = 18
RING_ARC_SAMPLES = 64

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

print("Torch CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))

# ============================
# CONFIG - VIDEO SELECTION
# ============================

def select_video() -> str:
    root = tk.Tk()
    root.withdraw()          # Hide the main tkinter window
    root.wm_attributes('-topmost', 1)  # Bring dialog to front

    file_path = filedialog.askopenfilename(
        title="Select Badminton Video",
        filetypes=[
            ("Video files", "*.mp4 *.avi *.mov *.mkv *.MOV"),
            ("All files", "*.*")
        ]
    )
    
    if not file_path:
        raise RuntimeError("No video file selected. Exiting.")
    
    print(f"✅ Selected video: {file_path}")
    return file_path

clicked_points = []
clicked_player_points = []


# ----------------------------
# CLIP EMBEDDER
# ----------------------------
class ClipEmbedder:
    def __init__(self, model_name: str, pretrained: str, device: str):
        self.device = device
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(model_name, pretrained=pretrained)
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
# UTIL + CALIBRATION + CLICK
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


def clamp01(v: float) -> float:
    return max(0.0, min(1.0, v))


def point_to_polygon_signed_distance(pt, poly) -> float:
    return float(cv2.pointPolygonTest(poly, pt, True))


def point_in_court_asymmetric_margin(pt, court_pts, top_margin, side_bottom_margin):
    x, y = float(pt[0]), float(pt[1])
    poly = court_pts.reshape(-1, 2).astype(np.float32)
    if cv2.pointPolygonTest(court_pts, (x, y), False) >= 0:
        return True
    signed_dist = float(cv2.pointPolygonTest(court_pts, (x, y), True))
    top_y_min = min(float(p[1]) for p in poly)
    if y >= top_y_min and signed_dist >= -side_bottom_margin:
        return True
    return False


def px_to_meters(H, x_px, y_px):
    pt = np.array([[[x_px, y_px]]], dtype=np.float32)
    out = cv2.perspectiveTransform(pt, H)[0, 0]
    return float(out[0]), float(out[1])


def meters_to_px(H, x_m, y_m):
    Hinv = np.linalg.inv(H)
    pt = np.array([[[x_m, y_m]]], dtype=np.float32)
    out = cv2.perspectiveTransform(pt, Hinv)[0, 0]
    return int(round(out[0])), int(round(out[1]))


def cosine_sim(a, b):
    if a is None or b is None:
        return 0.0
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na < 1e-8 or nb < 1e-8:
        return 0.0
    return np.dot(a, b) / (na * nb)


def safe_hist_corr(a, b):
    if a is None or b is None:
        return 0.0
    try:
        sim = cv2.compareHist(a.astype(np.float32), b.astype(np.float32), cv2.HISTCMP_CORREL)
        return clamp01((sim + 1.0) * 0.5)
    except:
        return 0.0


def bbox_iou_xyxy(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    union = (ax2-ax1)*(ay2-ay1) + (bx2-bx1)*(by2-by1) - inter
    return inter / union if union > 1e-8 else 0.0


def expand_bbox(x1, y1, x2, y2, w_img, h_img, frac=0.12):
    w = x2 - x1
    h = y2 - y1
    ex = int(round(w * frac))
    ey = int(round(h * frac))
    return max(0, int(x1)-ex), max(0, int(y1)-ey), min(w_img-1, int(x2)+ex), min(h_img-1, int(y2)+ey)


def crop_person(frame, xyxy, frac=0.04):
    h_img, w_img = frame.shape[:2]
    x1, y1, x2, y2 = expand_bbox(*xyxy, w_img, h_img, frac)
    crop = frame[y1:y2, x1:x2]
    return crop if crop.size > 0 and crop.shape[0] >= 12 and crop.shape[1] >= 8 else None


def extract_color_feature(frame, box_xyxy):
    crop = crop_person(frame, box_xyxy, 0.02)
    if crop is None:
        return None
    h, w = crop.shape[:2]
    tx1, tx2 = int(w*0.22), int(w*0.78)
    ty1, ty2 = int(h*0.16), int(h*0.62)
    torso = crop[ty1:ty2, tx1:tx2]
    if torso.size == 0:
        return None
    hsv = cv2.cvtColor(torso, cv2.COLOR_BGR2HSV)
    _, s, v = cv2.split(hsv)
    mask = ((s > 28) & (v > 35) & (v < 245)).astype(np.uint8) * 255
    hist_hs = cv2.calcHist([hsv], [0, 1], mask, [18, 16], [0, 180, 0, 256])
    hist_v = cv2.calcHist([hsv], [2], mask, [16], [0, 256])
    hist_hs = cv2.normalize(hist_hs, hist_hs).flatten()
    hist_v = cv2.normalize(hist_v, hist_v).flatten()
    return np.concatenate([hist_hs, hist_v]).astype(np.float32)


def classify_zone(x_m, y_m):
    mid = COURT_L / 2.0
    side_group = "FAR" if y_m < mid else "NEAR"
    depth = "FRONT" if abs(y_m - mid) <= 1.98 else "BACK"
    lr = "LEFT" if x_m < COURT_W / 2.0 else "RIGHT"
    return f"{side_group}-{depth}-{lr}"


def side_from_y(y_m):
    return "FAR" if y_m < COURT_L / 2.0 else "NEAR"


def depth_role_from_y(y_m):
    return "FRONT" if abs(y_m - COURT_L/2.0) <= 1.98 else "BACK"


def best_bank_similarity(bank: List[np.ndarray], feat: Optional[np.ndarray], topk: int = 5) -> float:
    if feat is None or not bank:
        return 0.0
    sims = []
    weights = np.linspace(0.4, 1.0, len(bank))
    weights /= weights.sum()
    for i, b in enumerate(bank):
        s = cosine_sim(b, feat)
        sims.append(clamp01((s + 1.0) * 0.5) * weights[i])
    sims.sort(reverse=True)
    return float(sum(sims[:topk]) / max(1, len(sims[:topk])))


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
        return pts, H
    except Exception:
        return None


def mouse_callback(event, x, y, flags, param):
    global clicked_points
    if event == cv2.EVENT_LBUTTONDOWN:
        clicked_points.append([x, y])


def click_corners(frame0: np.ndarray) -> list[list[int]]:
    global clicked_points
    clicked_points = []
    clone = frame0.copy()
    cv2.namedWindow("Click court corners", cv2.WINDOW_NORMAL)
    cv2.setMouseCallback("Click court corners", mouse_callback)
    while True:
        vis = clone.copy()
        for i, p in enumerate(clicked_points):
            cv2.circle(vis, tuple(p), 7, (0, 255, 0), -1)
            cv2.putText(vis, str(i + 1), (p[0] + 10, p[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.putText(vis, "Click 4 corners: TL, TR, BR, BL | R=reset | ESC/Q=quit", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
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


def player_click_callback(event, x, y, flags, param):
    global clicked_player_points
    if event == cv2.EVENT_LBUTTONDOWN:
        clicked_player_points.append([x, y])


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
            cv2.putText(vis, f"P{i+1}", (p[0] + 12, p[1] - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.putText(vis, f"Click {num_players} players | R=reset | ESC/Q=quit", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
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
# POSE FUNCTIONS
# ----------------------------
def angle_deg(a, b, c):
    ax, ay = a
    bx, by = b
    cx, cy = c
    v1 = np.array([ax - bx, ay - by])
    v2 = np.array([cx - bx, cy - by])
    n1 = np.linalg.norm(v1)
    n2 = np.linalg.norm(v2)
    if n1 < 1e-6 or n2 < 1e-6:
        return 180.0
    cosang = np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0)
    return float(np.degrees(np.arccos(cosang)))


def resize_keep_aspect(img, max_w, max_h):
    h, w = img.shape[:2]
    scale = min(max_w / float(w), max_h / float(h), 1.0)
    if scale >= 0.999:
        return img, 1.0
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR), scale


def _pose_pt(lm, i):
    p = lm[i]
    return float(p.x), float(p.y), float(p.visibility)


def extract_pose_signature(pose_res):
    if pose_res is None or pose_res.pose_landmarks is None:
        return None
    lm = pose_res.pose_landmarks.landmark
    ids = [11, 12, 23, 24, 25, 26, 27, 28]
    pts = [_pose_pt(lm, i) for i in ids]
    if min(p[2] for p in pts) < POSE_MIN_VIS:
        return None

    ls, rs, lh, rh, lk, rk, la, ra = pts
    shoulder_mid = np.array([(ls[0] + rs[0])*0.5, (ls[1] + rs[1])*0.5])
    hip_mid = np.array([(lh[0] + rh[0])*0.5, (lh[1] + rh[1])*0.5])
    ankle_mid = np.array([(la[0] + ra[0])*0.5, (la[1] + ra[1])*0.5])

    torso_len = float(np.linalg.norm(shoulder_mid - hip_mid))
    leg_len = float(np.linalg.norm(hip_mid - ankle_mid))
    body_len = max(1e-6, torso_len + leg_len)
    shoulder_w = float(np.linalg.norm(np.array(ls[:2]) - np.array(rs[:2]))) / body_len
    hip_w = float(np.linalg.norm(np.array(lh[:2]) - np.array(rh[:2]))) / body_len

    knee_l = angle_deg((lh[0], lh[1]), (lk[0], lk[1]), (la[0], la[1]))
    knee_r = angle_deg((rh[0], rh[1]), (rk[0], rk[1]), (ra[0], ra[1]))
    lean = float(np.clip((hip_mid[0] - shoulder_mid[0]) / body_len, -1.0, 1.0))
    torso_tilt = float(np.clip((hip_mid[1] - shoulder_mid[1]) / body_len, -1.0, 1.0))

    feat = np.array([
        shoulder_w,
        hip_w,
        torso_len / body_len,
        leg_len / body_len,
        knee_l / 180.0,
        knee_r / 180.0,
        lean,
        torso_tilt,
    ], dtype=np.float32)
    n = float(np.linalg.norm(feat))
    if n < 1e-8:
        return None
    return (feat / n).astype(np.float32)


def compute_ready_from_pose(pose_res):
    if pose_res is None or pose_res.pose_landmarks is None:
        return None, None, None, None, 0
    lm = pose_res.pose_landmarks.landmark

    def get(i):
        return lm[i].x, lm[i].y, lm[i].visibility

    Lhip = get(23); Rhip = get(24)
    Lknee = get(25); Rknee = get(26)
    Lank = get(27); Rank = get(28)

    vis_ok = all(v[2] > POSE_MIN_VIS for v in [Lhip, Rhip, Lknee, Rknee, Lank, Rank])
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


# ----------------------------
# RESCUE MODE HOOK
# ----------------------------
def maybe_rescue_assignment(det_rows: List[dict], ambiguous_flags: List[bool], occlusion_flags: List[bool]) -> None:
    if not RESCUE_MODE_ENABLED:
        return
    for i, det in enumerate(det_rows):
        det["rescue_flag"] = int(bool(ambiguous_flags[i] or occlusion_flags[i]))


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
        last_t=t,
        last_x=x_m,
        last_y=y_m,
        speed_kmh_ema=0.0,
        speed_mps_ema=0.0,
        accel_mps2_ema=0.0,
        prev_speed_mps=0.0,
        total_dist_m=0.0,
        knee_angle_ema=175.0,
        hip_drop_ema=0.0,
        ready_frames=0,
        total_frames=0,
        last_pose_frame=-999999,
        last_knee_angle=None,
        last_ready_flag=0
    )


def update_motion(states: Dict[int, LiveState], stable_id: int, t: float, x_m: float, y_m: float):
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


def update_ready_state(states: Dict[int, LiveState], stable_id: int, frame_idx: int, knee_avg, hip_drop, ready_flag):
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


def draw_rounded_rect(img, x1, y1, x2, y2, color, thickness=-1, radius=10):
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


def overlay_alpha(base_img, overlay_img, alpha: float):
    cv2.addWeighted(overlay_img, alpha, base_img, 1 - alpha, 0, base_img)


def draw_open_ellipse_arc(img, center, axes, start_rad, end_rad, color, thickness, samples=64):
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


def draw_player_overlay(frame, bbox_xyxy, stable_id, knee_angle_avg, ready_pct, ring_color=RING_COLOR, ring_alpha=RING_ALPHA):
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

    cv2.putText(frame, f"{stable_id}", (px1 + 10, py1 + 26), cv2.FONT_HERSHEY_SIMPLEX, 0.95, (10, 10, 10), 3, cv2.LINE_AA)
    txt = f"Knee: {knee_angle_avg:.0f} deg" if knee_angle_avg is not None else "Knee: --"
    cv2.putText(frame, txt, (px1 + 10, py1 + 44), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (10, 10, 10), 2, cv2.LINE_AA)
    if ready_pct is not None:
        cv2.putText(frame, f"READY: {ready_pct:.0f}%", (px1 + 10, py1 + 62), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (10, 10, 10), 2, cv2.LINE_AA)


# ----------------------------
# IDENTITY MANAGER (med lang memory direction prediction + rettet _direction_cost)
# ----------------------------
@dataclass
class IdentityTrack:
    stable_id: int
    last_x: float
    last_y: float
    last_seen_frame: int
    clip_feat: Optional[np.ndarray]
    color_feat: Optional[np.ndarray]
    pose_feat: Optional[np.ndarray]
    clip_bank: List[np.ndarray]
    color_bank: List[np.ndarray]
    pose_bank: List[np.ndarray]
    box_h: float
    box_w: float
    raw_tracker_id: Optional[int]

    # Fields with defaults come LAST
    vx_history: List[float] = field(default_factory=list)
    vy_history: List[float] = field(default_factory=list)
    hits: int = 1
    pending_raw_tid: Optional[int] = None
    pending_count: int = 0
    preferred_side: Optional[str] = None
    side_stable_frames: int = 0
    preferred_role: Optional[str] = None
    role_stable_frames: int = 0


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
        for sid, (click_x, click_y) in zip(self.stable_ids, clicked_points):
            best_i = None
            best_score = -float('inf')
            for i, det in enumerate(detections):
                if i in used:
                    continue
                cx = 0.5 * (det["bbox_x1"] + det["bbox_x2"])
                cy = 0.5 * (det["bbox_y1"] + det["bbox_y2"])
                dist = math.hypot(cx - click_x, cy - click_y)

                inside_bonus = 400.0 if (
                    det["bbox_x1"] <= click_x <= det["bbox_x2"] and
                    det["bbox_y1"] <= click_y <= det["bbox_y2"]
                ) else 0.0

                score = inside_bonus - dist * 10.0
                if score > best_score:
                    best_score = score
                    best_i = i

            if best_i is not None:
                used.add(best_i)
                det = detections[best_i]
                clip_bank = [det.get("clip_feat")] if det.get("clip_feat") is not None else []
                color_bank = [det.get("color_feat")] if det.get("color_feat") is not None else []
                pose_bank = [det.get("pose_feat")] if det.get("pose_feat") is not None else []
                pref_side = side_from_y(float(det["y_m"]))
                pref_role = depth_role_from_y(float(det["y_m"]))
                track = IdentityTrack(
                    stable_id=sid,
                    last_x=float(det["x_m"]),
                    last_y=float(det["y_m"]),
                    last_seen_frame=int(det["frame_idx"]),
                    clip_feat=det.get("clip_feat"),
                    color_feat=det.get("color_feat"),
                    pose_feat=det.get("pose_feat"),
                    clip_bank=clip_bank,
                    color_bank=color_bank,
                    pose_bank=pose_bank,
                    box_h=float(det.get("box_h", 0.0)),
                    box_w=float(det.get("box_w", 0.0)),
                    raw_tracker_id=int(det.get("track_id", -1)),
                    vx_history=[],
                    vy_history=[],
                    hits=1,
                    preferred_side=pref_side,
                    side_stable_frames=1,
                    preferred_role=pref_role,
                    role_stable_frames=1,
                )
                self.tracks[sid] = track
                det["stable_id"] = sid
                det["side_group"] = "ALL"

    def _push_bank(self, bank: List[np.ndarray], feat: Optional[np.ndarray], max_size: int) -> List[np.ndarray]:
        if feat is None:
            return bank
        bank.append(feat.astype(np.float32))
        if len(bank) > max_size:
            bank = bank[-max_size:]
        return bank

    def _predict_xy(self, tr: IdentityTrack, frame_idx: int) -> Tuple[float, float]:
        if tr.hits < 2:
            return tr.last_x, tr.last_y

        # Lang memory – vægtet gennemsnit over op til 12 frames
        history_len = min(len(tr.vx_history), VELOCITY_HISTORY_FRAMES)
        if history_len == 0:
            return tr.last_x, tr.last_y

        # Eksponentielt faldende vægt: nyeste ~1.0, ældste ~0.05
        weights = np.exp(np.linspace(0, LINESPACE_FRAMES, history_len))
        weights /= weights.sum()

        weighted_vx = np.sum(np.array(tr.vx_history[-history_len:]) * weights)
        weighted_vy = np.sum(np.array(tr.vy_history[-history_len:]) * weights)

        dtf = max(1, frame_idx - tr.last_seen_frame)
        pred_x = tr.last_x + weighted_vx * dtf
        pred_y = tr.last_y + weighted_vy * dtf
        return pred_x, pred_y

    def _direction_cost(self, tr: IdentityTrack, det: dict, frame_idx: int) -> float:
        if tr.hits < 6:
            return 0.0
        pred_x, pred_y = self._predict_xy(tr, frame_idx)
        dx_pred = pred_x - tr.last_x
        dy_pred = pred_y - tr.last_y
        len_pred = math.hypot(dx_pred, dy_pred)
        if len_pred < 0.05:
            return 0.0

        dx_new = float(det["x_m"]) - tr.last_x
        dy_new = float(det["y_m"]) - tr.last_y
        len_new = math.hypot(dx_new, dy_new)
        if len_new < 0.05:
            return 0.15

        cos_sim = (dx_pred * dx_new + dy_pred * dy_new) / (len_pred * len_new + 1e-8)
        cos_sim = max(-1.0, min(1.0, cos_sim))
        direction_cost = (1.0 - cos_sim) / 2.0

        return direction_cost * (0.48 if len_new > 0.35 else 0.22)

    def _long_term_direction_cost(self, tr: IdentityTrack, det: dict, frame_idx: int) -> float:
        if len(tr.vx_history) < 3:
            return 0.0

        pred_x, pred_y = self._predict_xy(tr, frame_idx)
        dx_pred = pred_x - tr.last_x
        dy_pred = pred_y - tr.last_y
        len_pred = math.hypot(dx_pred, dy_pred)
        if len_pred < 0.08:
            return 0.0

        dx_new = float(det["x_m"]) - tr.last_x
        dy_new = float(det["y_m"]) - tr.last_y
        len_new = math.hypot(dx_new, dy_new)
        if len_new < 0.05:
            return 0.45

        cos_sim = (dx_pred * dx_new + dy_pred * dy_new) / (len_pred * len_new + 1e-8)
        cos_sim = max(-1.0, min(1.0, cos_sim))
        direction_cost = (1.0 - cos_sim) / 2.0

        if cos_sim < -0.2:
            return direction_cost * DIRECTION_COST
        return direction_cost * 0.35

    def _clip_cost(self, tr: IdentityTrack, det: dict, occluded: bool) -> float:
        feat = det.get("clip_feat")
        sim_bank = best_bank_similarity(tr.clip_bank, feat, topk=MEMORY_MATCH_TOPK)
        sim_ema = cosine_sim(tr.clip_feat, feat) if tr.clip_feat is not None and feat is not None else 0.0
        sim_ema = clamp01((sim_ema + 1.0) * 0.5)
        if feat is None:
            return 0.45
        sim = max(sim_bank, sim_ema)
        cost = 1.0 - sim
        if occluded:
            cost = min(1.0, cost + OCCLUSION_CLIP_REDUCE)
        return cost

    def _motion_cost(self, tr: IdentityTrack, det: dict, frame_idx: int, occluded: bool) -> float:
        px, py = self._predict_xy(tr, frame_idx)
        d = math.hypot(float(det["x_m"]) - px, float(det["y_m"]) - py)
        base = clamp01(d / max(1e-6, self.max_match_dist_m))
        if tr.hits >= 12:
            base *= 0.92
        if occluded:
            base = max(0.0, base - OCCLUSION_MOTION_BOOST)
        return base

    def _color_cost(self, tr: IdentityTrack, det: dict) -> float:
        feat = det.get("color_feat")
        sim_bank = 0.0
        if feat is not None and tr.color_bank:
            sims = [safe_hist_corr(b, feat) for b in tr.color_bank]
            sims.sort(reverse=True)
            sim_bank = float(sum(sims[:4]) / max(1, len(sims[:4])))
        sim_ema = safe_hist_corr(tr.color_feat, feat) if feat is not None and tr.color_feat is not None else 0.0
        sim = max(sim_bank, sim_ema)
        if feat is None:
            return 0.35
        return 1.0 - sim

    def _pose_cost(self, tr: IdentityTrack, det: dict, ambiguous: bool) -> float:
        feat = det.get("pose_feat")
        sim_bank = best_bank_similarity(tr.pose_bank, feat, topk=3)
        sim_ema = cosine_sim(tr.pose_feat, feat) if tr.pose_feat is not None and feat is not None else 0.0
        sim_ema = clamp01((sim_ema + 1.0) * 0.5)
        if feat is None:
            return 0.42 if ambiguous else 0.32
        sim = max(sim_bank, sim_ema)
        return 1.0 - sim

    def _size_cost(self, tr: IdentityTrack, det: dict) -> float:
        dh = float(det.get("box_h", 0.0))
        dw = float(det.get("box_w", 0.0))
        ch = 0.25 if tr.box_h <= 1e-6 or dh <= 1e-6 else clamp01(abs(dh - tr.box_h) / max(dh, tr.box_h))
        cw = 0.20 if tr.box_w <= 1e-6 or dw <= 1e-6 else clamp01(abs(dw - tr.box_w) / max(dw, tr.box_w))
        return 0.65 * ch + 0.35 * cw

    def _side_cost(self, tr: IdentityTrack, det: dict) -> float:
        if tr.preferred_side is None:
            return 0.0
        current_side = side_from_y(float(det["y_m"]))
        if current_side == tr.preferred_side:
            return 0.0
        if tr.hits >= HARD_SIDE_HITS_MIN and tr.side_stable_frames >= SIDE_GRACE_FRAMES:
            return HARD_SIDE_PENALTY
        return SIDE_PENALTY if tr.side_stable_frames < SIDE_GRACE_FRAMES else SIDE_PENALTY * 0.55

    def _role_cost(self, tr: IdentityTrack, det: dict) -> float:
        if tr.preferred_role is None:
            return 0.0
        role_now = depth_role_from_y(float(det["y_m"]))
        if role_now == tr.preferred_role:
            return 0.0
        return ROLE_HARD_PENALTY if tr.role_stable_frames >= ROLE_SWAP_CONFIRM_FRAMES else ROLE_HARD_PENALTY * 0.55

    def _lateral_cross_penalty(self, tr: IdentityTrack, det: dict) -> float:
        if tr.hits < 15:
            return 0.0
        prev_lr = "LEFT" if tr.last_x < COURT_W / 2.0 else "RIGHT"
        now_lr = "LEFT" if float(det["x_m"]) < COURT_W / 2.0 else "RIGHT"
        if prev_lr != now_lr:
            dist_to_net = abs(float(det["y_m"]) - COURT_L / 2.0)
            if dist_to_net < 3.5:
                factor = max(0.0, 1.0 - dist_to_net / 3.5)
                return 0.28 + 0.22 * factor
        return 0.0

    def _edge_cost(self, det: dict) -> float:
        d = float(det.get("foot_poly_dist", 0.0))
        if d >= 15:
            return 0.0
        if d <= -20:
            return 1.0
        return clamp01((15.0 - d) / 35.0)

    def _total_cost(self, tr: IdentityTrack, det: dict, frame_idx: int, occluded: bool, ambiguous: bool) -> float:
        clip_c = self._clip_cost(tr, det, occluded)
        motion_c = self._motion_cost(tr, det, frame_idx, occluded)
        color_c = self._color_cost(tr, det)
        size_c = self._size_cost(tr, det)
        side_c = self._side_cost(tr, det)
        edge_c = self._edge_cost(det)
        pose_c = self._pose_cost(tr, det, ambiguous)
        role_c = self._role_cost(tr, det)
        lateral_c = self._lateral_cross_penalty(tr, det)
        direction_c = self._direction_cost(tr, det, frame_idx)
        long_term_direction_c = self._long_term_direction_cost(tr, det, frame_idx)

        dist_to_net = abs(float(det["y_m"]) - COURT_L / 2.0)
        near_net = dist_to_net < NET_DISTANCE_THRESHOLD_M

        w_pose = W_POSE * (POSE_NET_BOOST_FACTOR if near_net else 1.0)
        role_final = role_c * (NET_ROLE_EXTRA_PENALTY if near_net else 1.0)

        return (
            W_CLIP * clip_c +
            W_MOTION * motion_c +
            W_COLOR * color_c +
            W_SIZE * size_c +
            W_SIDE * side_c +
            W_EDGE * edge_c +
            w_pose * pose_c +
            role_final +
            LATERAL_CROSS_WEIGHT * lateral_c +
            W_DIRECTION * direction_c +
            LONG_TERM_DIRECTION_FRAMES * long_term_direction_c
        )

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

    def _update_role_memory(self, tr: IdentityTrack, det: dict):
        role_now = depth_role_from_y(float(det["y_m"]))
        if tr.preferred_role is None:
            tr.preferred_role = role_now
            tr.role_stable_frames = 1
            return
        if role_now == tr.preferred_role:
            tr.role_stable_frames += 1
        else:
            if tr.role_stable_frames > ROLE_SWAP_CONFIRM_FRAMES:
                tr.preferred_role = role_now
                tr.role_stable_frames = 1
            else:
                tr.role_stable_frames = max(0, tr.role_stable_frames - 1)

    def _compute_ambiguity_flags(self, detections: List[dict]) -> List[bool]:
        flags = [False] * len(detections)
        for i in range(len(detections)):
            xi = float(detections[i].get("center_px_x", 0.0))
            yi = float(detections[i].get("center_px_y", 0.0))
            box_i = (float(detections[i]["bbox_x1"]), float(detections[i]["bbox_y1"]), float(detections[i]["bbox_x2"]), float(detections[i]["bbox_y2"]))
            for j in range(i + 1, len(detections)):
                xj = float(detections[j].get("center_px_x", 0.0))
                yj = float(detections[j].get("center_px_y", 0.0))
                box_j = (float(detections[j]["bbox_x1"]), float(detections[j]["bbox_y1"]), float(detections[j]["bbox_x2"]), float(detections[j]["bbox_y2"]))
                dx = abs(xi - xj)
                dy = abs(yi - yj)
                iou = bbox_iou_xyxy(box_i, box_j)
                if (dx <= AMBIGUOUS_X_PX and dy <= AMBIGUOUS_Y_PX) or iou >= AMBIGUOUS_IOU_THRESH:
                    flags[i] = True
                    flags[j] = True
        return flags

    def _compute_occlusion_flags(self, detections: List[dict]) -> List[bool]:
        flags = [False] * len(detections)
        for i in range(len(detections)):
            box_i = (float(detections[i]["bbox_x1"]), float(detections[i]["bbox_y1"]), float(detections[i]["bbox_x2"]), float(detections[i]["bbox_y2"]))
            for j in range(i + 1, len(detections)):
                box_j = (float(detections[j]["bbox_x1"]), float(detections[j]["bbox_y1"]), float(detections[j]["bbox_x2"]), float(detections[j]["bbox_y2"]))
                if bbox_iou_xyxy(box_i, box_j) >= OCCLUSION_IOU_THRESH:
                    flags[i] = True
                    flags[j] = True
        return flags

    def _det_pair_is_close(self, a: dict, b: dict) -> bool:
        dist_m = math.hypot(float(a["x_m"]) - float(b["x_m"]), float(a["y_m"]) - float(b["y_m"]))
        box_a = (float(a["bbox_x1"]), float(a["bbox_y1"]), float(a["bbox_x2"]), float(a["bbox_y2"]))
        box_b = (float(b["bbox_x1"]), float(b["bbox_y1"]), float(b["bbox_x2"]), float(b["bbox_y2"]))
        iou = bbox_iou_xyxy(box_a, box_b)
        return dist_m <= PAIR_SWAP_MAX_M or iou >= PAIR_SWAP_IOU_THRESH

    def _apply_teammate_constraints(self, proposed: Dict[int, int], detections: List[dict]) -> Dict[int, int]:
        if len(proposed) < 4:
            return proposed
        out = dict(proposed)
        teams = [(1, 2), (3, 4)]
        for a, b in teams:
            ia = out.get(a)
            ib = out.get(b)
            if ia is None or ib is None:
                continue
            da = detections[ia]
            db = detections[ib]
            sep = math.hypot(float(da["x_m"]) - float(db["x_m"]), float(da["y_m"]) - float(db["y_m"]))
            if sep >= TEAMMATE_MIN_SEP_M:
                continue
            out[a] = ia
            out[b] = ib
        return out

    def _anti_swap_pairs(self, proposed, detections, cost_matrix, track_ids, occlusion_flags, ambiguity_flags, frame_idx):
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
                if tr_a.hits < PAIR_SWAP_HIT_MIN or tr_b.hits < PAIR_SWAP_HIT_MIN:
                    continue
                if not self._det_pair_is_close(det_a, det_b):
                    continue
                amb = ambiguity_flags[det_a_idx] or ambiguity_flags[det_b_idx]
                occ = occlusion_flags[det_a_idx] or occlusion_flags[det_b_idx]
                row_a = track_ids.index(sid_a)
                row_b = track_ids.index(sid_b)
                chosen = float(cost_matrix[row_a, det_a_idx] + cost_matrix[row_b, det_b_idx])
                alt = float(cost_matrix[row_a, det_b_idx] + cost_matrix[row_b, det_a_idx])
                margin = PAIR_SWAP_COST_MARGIN + (PAIR_SWAP_AMBIGUOUS_BONUS if amb else 0.0) + (0.06 if occ else 0.0)
                if alt <= chosen + margin:
                    raw_a = det_a.get("track_id")
                    raw_b = det_b.get("track_id")
                    keep_a = tr_a.raw_tracker_id is not None and raw_a is not None and int(tr_a.raw_tracker_id) == int(raw_a)
                    keep_b = tr_b.raw_tracker_id is not None and raw_b is not None and int(tr_b.raw_tracker_id) == int(raw_b)
                    if keep_a and keep_b:
                        continue
                    if keep_a and not keep_b:
                        out[sid_a] = det_a_idx
                        continue
                    if keep_b and not keep_a:
                        out[sid_b] = det_b_idx
                        continue
        return out

    def _update_track(self, sid: int, det: dict, frame_idx: int, occluded: bool, ambiguous: bool) -> None:
        tr = self.tracks[sid]
        dtf = max(1, frame_idx - tr.last_seen_frame)
        nvx = (float(det["x_m"]) - tr.last_x) / dtf
        nvy = (float(det["y_m"]) - tr.last_y) / dtf

    # Opdater velocity-historik (begræns til 12 frames)
        tr.vx_history.append(nvx)
        tr.vy_history.append(nvy)
        if len(tr.vx_history) > VELOCITY_HISTORY_FRAMES:
            tr.vx_history = tr.vx_history[-VELOCITY_HISTORY_FRAMES:]
            tr.vy_history = tr.vy_history[-VELOCITY_HISTORY_FRAMES:]

    # Ingen tr.vx / tr.vy mere – vi bruger kun historikken fremover

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
                    tr.clip_feat /= n
            freeze = (occluded and OCCLUSION_FREEZE_BANK_UPDATE) or (ambiguous and AMBIGUOUS_FREEZE_BANK_UPDATE)
            if not freeze:
                tr.clip_bank = self._push_bank(tr.clip_bank, new_clip, MEMORY_BANK_SIZE)

        new_color = det.get("color_feat")
        if new_color is not None:
            if tr.color_feat is None:
                tr.color_feat = new_color.astype(np.float32)
            else:
                tr.color_feat = (COLOR_EMA_ALPHA * new_color + (1.0 - COLOR_EMA_ALPHA) * tr.color_feat).astype(np.float32)
            freeze = (occluded and OCCLUSION_FREEZE_BANK_UPDATE) or (ambiguous and AMBIGUOUS_FREEZE_BANK_UPDATE)
            if not freeze:
                tr.color_bank = self._push_bank(tr.color_bank, new_color, MEMORY_BANK_SIZE)

        new_pose = det.get("pose_feat")
        if new_pose is not None:
            if tr.pose_feat is None:
                tr.pose_feat = new_pose.astype(np.float32)
            else:
                tr.pose_feat = (POSE_EMA_ALPHA * new_pose + (1.0 - POSE_EMA_ALPHA) * tr.pose_feat).astype(np.float32)
                n = float(np.linalg.norm(tr.pose_feat))
                if n > 1e-8:
                    tr.pose_feat /= n
            freeze = (occluded and OCCLUSION_FREEZE_BANK_UPDATE) or (ambiguous and AMBIGUOUS_FREEZE_BANK_UPDATE)
            if not freeze:
                tr.pose_bank = self._push_bank(tr.pose_bank, new_pose, POSE_BANK_SIZE)

        tr.box_h = SIZE_ALPHA * float(det.get("box_h", tr.box_h)) + (1.0 - SIZE_ALPHA) * tr.box_h
        tr.box_w = SIZE_ALPHA * float(det.get("box_w", tr.box_w)) + (1.0 - SIZE_ALPHA) * tr.box_w
        raw_tid = det.get("track_id")
        if raw_tid is not None:
            tr.raw_tracker_id = int(raw_tid)
        tr.hits += 1
        self._update_side_memory(tr, det)
        self._update_role_memory(tr, det)

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
                base_cost = self._total_cost(tr, det, frame_idx, occluded, ambiguous)
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
            ambiguous = ambiguity_flags[c]
            total_cost = float(cost_matrix[r, c])
            motion_cost = self._motion_cost(tr, det, frame_idx, occluded)
            clip_cost = self._clip_cost(tr, det, occluded)
            pose_cost = self._pose_cost(tr, det, ambiguous)
            clip_sim = 1.0 - clip_cost
            pose_sim = 1.0 - pose_cost
            motion_dist_ok = motion_cost <= clamp01(OUTSIDER_DIST_M / max(1e-6, self.max_match_dist_m))
            strong_identity = clip_sim >= OUTSIDER_CLIP_MIN_SIM or pose_sim >= 0.66
            local_max_accept = AMBIGUOUS_MAX_ACCEPT_COST if ambiguous else MAX_ACCEPT_COST
            if total_cost <= local_max_accept and (motion_dist_ok or strong_identity):
                proposed[sid] = c

        proposed = self._anti_swap_pairs(proposed, detections, cost_matrix, track_ids, occlusion_flags, ambiguity_flags, frame_idx)
        proposed = self._apply_teammate_constraints(proposed, detections)

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
            same_raw = tr.raw_tracker_id is not None and raw_tid is not None and int(raw_tid) == int(tr.raw_tracker_id)
            accept = True
            local_switch_margin = SWITCH_MARGIN + (OCCLUSION_EXTRA_STICKINESS if occluded else 0.0) + (AMBIGUOUS_EXTRA_SWITCH_MARGIN if ambiguous else 0.0)
            local_confirm = SWITCH_CONFIRM_FRAMES + (2 if occluded else 0) + (AMBIGUOUS_EXTRA_CONFIRM_FRAMES if ambiguous else 0)
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
# MAIN
# ----------------------------
def main():

    VIDEO_PATH = select_video()

    if not os.path.exists(VIDEO_PATH):
        raise RuntimeError(f"Fil findes ikke: {VIDEO_PATH}")

    print("Loading CLIP...")
    clip_embedder = ClipEmbedder(CLIP_MODEL_NAME, CLIP_PRETRAINED, CLIP_DEVICE)
    print(f"CLIP loaded on {CLIP_DEVICE}")

    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        raise RuntimeError(f"Kunne ikke åbne videoen: {VIDEO_PATH}")
    ok, frame0 = cap.read()
    if not ok or frame0 is None:
        raise RuntimeError("Første frame kunne ikke læses.")

    loaded = load_calibration(CALIB_FILE)
    if loaded is None:
        pts = click_corners(frame0)
        src = np.array(pts, dtype=np.float32)
        H, _ = cv2.findHomography(src, COURT_DST)
        if H is None:
            raise RuntimeError("Homography kunne ikke beregnes.")
        save_calibration(CALIB_FILE, pts, H)
    else:
        pts, H = loaded

    src = np.array(pts, dtype=np.float32)
    court_poly_px = src.reshape(-1, 2).astype(np.int32)
    player_points = click_players(frame0, MAX_PLAYERS)

    cap.release()
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        raise RuntimeError(f"Kunne ikke genåbne videoen: {VIDEO_PATH}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or frame0.shape[1]
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or frame0.shape[0]
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    if not fps or math.isnan(fps) or fps <= 1:
        fps = SAVE_FPS_FALLBACK

    print("Loading YOLO...")
    model = YOLO("yolo11m.pt")

    tracker = sv.ByteTrack(
        track_activation_threshold=0.35,
        lost_track_buffer=45,
        minimum_matching_threshold=0.78,
        frame_rate=int(fps)
    )

    max_age_frames = int(MAX_AGE_SECONDS * fps)
    identity = IdentityManager(PLAYER_IDS, max_age_frames, MAX_MATCH_DIST_M)
    live_states: Dict[int, LiveState] = {}

    pose_options = PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path="pose_landmarker_full.task"),
        running_mode=mp_vision.RunningMode.IMAGE,
        num_poses=1,
        min_pose_detection_confidence=0.5,
        min_pose_presence_confidence=0.5,
        min_tracking_confidence=0.5,
        output_segmentation_masks=False
    )

    pose_landmarker = PoseLandmarker.create_from_options(pose_options)

    ensure_dir(SAVE_DIR)
    out_path = make_output_path(SAVE_DIR, VIDEO_PATH)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_path, fourcc, fps, (width, height))
    if not writer.isOpened():
        fourcc = cv2.VideoWriter_fourcc(*"avc1")
        writer = cv2.VideoWriter(out_path, fourcc, fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError("Kunne ikke oprette VideoWriter.")

    ensure_dir(EXPORT_DIR)
    csv_path, jsonl_path = make_export_paths(EXPORT_DIR, VIDEO_PATH)
    csv_f = open(csv_path, "w", newline="", encoding="utf-8")
    csv_w = csv.DictWriter(csv_f, fieldnames=[
        "frame_idx", "timestamp_s", "track_id", "stable_id", "side_group",
        "bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2",
        "foot_px_x", "foot_px_y", "x_m", "y_m", "zone",
        "speed_kmh", "distance_m", "accel_mps2",
        "knee_angle_L", "knee_angle_R", "knee_angle_avg",
        "ready_flag", "ready_pct", "rescue_flag",
    ])
    csv_w.writeheader()
    jsonl_f = open(jsonl_path, "w", encoding="utf-8")

    cv2.namedWindow("Badminton tracking", cv2.WINDOW_NORMAL)
    frame_idx = -1
    bootstrapped = False

    try:
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                break
            frame_idx += 1
            timestamp_s = frame_idx / fps

            results = model(frame, verbose=False, conf=CONF, iou=IOU, imgsz=IMGSZ)[0]
            detections = sv.Detections.from_ultralytics(results)
            annotated = frame.copy()
            det_rows: List[dict] = []

            if detections.class_id is not None and len(detections) > 0:
                detections = detections[detections.class_id == 0]
                if len(detections) > 0:
                    keep = []
                    for xyxy in detections.xyxy:
                        x1, y1, x2, y2 = xyxy
                        foot = (float((x1 + x2) / 2), float(y2))
                        inside = point_in_court_asymmetric_margin(foot, court_poly_px, COURT_MARGIN_PX, SIDE_BOTTOM_MARGIN_PX)
                        keep.append(inside)
                    detections = detections[np.array(keep, dtype=bool)]

                detections = tracker.update_with_detections(detections)

                if detections.tracker_id is not None and len(detections) > 0:
                    crops = []
                    temp_rows = []
                    H_img, W_img = frame.shape[:2]
                    do_pose_this_frame = (frame_idx % POSE_EVERY_N_FRAMES == 0)

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

                        pose_feat = None
                        knee_L = knee_R = knee_avg = hip_drop = None
                        ready_flag = 0
                        if do_pose_this_frame:
                            bx1, by1, bx2, by2 = expand_bbox(x1, y1, x2, y2, W_img, H_img, frac=POSE_BBOX_EXPAND)
                            pose_crop = frame[by1:by2, bx1:bx2]
                            if pose_crop.size > 0:
                                pose_crop_small, _ = resize_keep_aspect(pose_crop, POSE_CROP_MAX_W, POSE_CROP_MAX_H)
                                
                                # Ny MediaPipe Tasks API
                                mp_image = mp.Image(
                                    image_format=mp.ImageFormat.SRGB,
                                    data=cv2.cvtColor(pose_crop_small, cv2.COLOR_BGR2RGB)
                                )
                                
                                pose_result = pose_landmarker.detect(mp_image)
                                
                                # Tilpas til dit eksisterende kode (så du ikke skal ændre extract_pose_signature)
                                if pose_result.pose_landmarks and len(pose_result.pose_landmarks) > 0:
                                    class FakeLandmarks:
                                        def __init__(self, landmarks):
                                            self.landmark = landmarks

                                    class FakePoseResult:
                                        def __init__(self, landmarks):
                                            self.pose_landmarks = FakeLandmarks(landmarks)

                                    pose_res = FakePoseResult(pose_result.pose_landmarks[0])
                                else:
                                    pose_res = None

                                pose_feat = extract_pose_signature(pose_res)
                                knee_L, knee_R, knee_avg, hip_drop, ready_flag = compute_ready_from_pose(pose_res)

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
                            "box_w": box_w, "box_h": box_h,
                            "clip_feat": None,
                            "color_feat": color_feat,
                            "pose_feat": pose_feat,
                            "stable_id": -1,
                            "side_group": "ALL",
                            "knee_angle_L": knee_L,
                            "knee_angle_R": knee_R,
                            "knee_angle_avg": knee_avg,
                            "hip_drop": hip_drop,
                            "ready_flag": ready_flag,
                            "rescue_flag": 0,
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
                        occ_flags = identity._compute_occlusion_flags(det_rows)
                        amb_flags = identity._compute_ambiguity_flags(det_rows)
                        maybe_rescue_assignment(det_rows, amb_flags, occ_flags)
                        identity.assign(frame_idx, det_rows)

                    det_rows = [r for r in det_rows if int(r.get("stable_id", -1)) in PLAYER_IDS]
                    xyxy_by_track = {int(tid): xy for xy, tid in zip(detections.xyxy, detections.tracker_id)}

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

                        if r["ready_flag"] == 1 and r["speed_kmh"] > READY_MAX_SPEED_KMH:
                            r["ready_flag"] = 0

                        update_ready_state(live_states, sid, frame_idx, r.get("knee_angle_avg"), r.get("hip_drop"), int(r.get("ready_flag", 0)))
                        st = live_states.get(sid)
                        if st is not None:
                            if r.get("knee_angle_avg") is None:
                                r["knee_angle_avg"] = st.last_knee_angle
                            r["ready_pct"] = float(get_ready_pct(live_states, sid))
                        else:
                            r["ready_pct"] = 0.0

                        ring_col = knee_to_bgr(r.get("knee_angle_avg"))
                        draw_player_overlay(annotated, xyxy, sid, r.get("knee_angle_avg"), r.get("ready_pct"), ring_color=ring_col)

                    for r in det_rows:
                        row_out = dict(r)
                        for key in ["clip_feat", "color_feat", "pose_feat", "box_w", "box_h", "foot_poly_dist", "center_px_x", "center_px_y", "hip_drop"]:
                            row_out.pop(key, None)
                        row_out.setdefault("speed_kmh", 0.0)
                        row_out.setdefault("distance_m", 0.0)
                        row_out.setdefault("accel_mps2", 0.0)
                        row_out.setdefault("ready_pct", 0.0)
                        csv_w.writerow(row_out)
                        jsonl_f.write(json.dumps(row_out, ensure_ascii=False) + "\n")

            cv2.polylines(annotated, [court_poly_px], isClosed=True, color=(255, 255, 0), thickness=2)
            try:
                draw_court_guides(annotated, H)
            except Exception:
                pass
            cv2.putText(annotated, "ULTIMATE VERSION – Direction + Weighted + Better Bootstrap", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.72, (0, 255, 255), 2, cv2.LINE_AA)
            cv2.imshow("Badminton tracking", annotated)
            writer.write(annotated)
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q"), ord("Q")):
                break
    finally:
        cap.release()
        writer.release()
        csv_f.close()
        jsonl_f.close()
        cv2.destroyAllWindows()

    print(f"Saved video: {out_path}")
    print(f"Saved CSV:   {csv_path}")
    print(f"Saved JSONL: {jsonl_path}")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback
        traceback.print_exc()
        input("\nPress Enter to exit...")
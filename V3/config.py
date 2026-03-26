"""
V3 configuration — all tunable constants in one place.
Defaults match the tuned V2 values; adjust here without touching module code.
"""

import torch

# ── output directories ────────────────────────────────────────────────────────
OUTPUTS_ROOT = "outputs"          # relative to V3/
SAVE_FPS_FALLBACK = 30.0

# ── model paths (relative to V3/ or absolute) ────────────────────────────────
YOLO_MODEL_PATH   = "../V2/yolo11m.pt"
POSE_MODEL_PATH   = "../V2/pose_landmarker_full.task"
CALIB_DIR         = "../V2/calibration"
CALIB_SETUP_NAME  = "almind_viuf_hallen_baseline2"

# ── YOLO detection ────────────────────────────────────────────────────────────
CONF   = 0.45
IOU    = 0.60
IMGSZ  = 1088

# ── court geometry ────────────────────────────────────────────────────────────
COURT_W = 6.10
COURT_L = 13.40
COURT_MARGIN_PX       = 85
SIDE_BOTTOM_MARGIN_PX = 2

# ── player IDs ────────────────────────────────────────────────────────────────
PLAYER_IDS  = [1, 2, 3, 4]
MAX_PLAYERS = 4

# ── CLIP embedder ─────────────────────────────────────────────────────────────
CLIP_MODEL_NAME = "ViT-B-32"
CLIP_PRETRAINED = "laion2b_s34b_b79k"
CLIP_DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"

# ── identity manager ──────────────────────────────────────────────────────────
MAX_AGE_SECONDS   = 3.0
MAX_MATCH_DIST_M  = 3.2

W_CLIP      = 0.34
W_MOTION    = 0.18
W_COLOR     = 0.06
W_SIZE      = 0.04
W_SIDE      = 0.12
W_EDGE      = 0.06
W_POSE      = 0.20
W_DIRECTION = 0.28

SWITCH_MARGIN        = 0.14
SWITCH_CONFIRM_FRAMES = 8
UNMATCHED_COST       = 1.15

PAIR_SWAP_MAX_M         = 1.45
PAIR_SWAP_IOU_THRESH    = 0.015
PAIR_SWAP_COST_MARGIN   = 0.24
PAIR_SWAP_HIT_MIN       = 18
PAIR_SWAP_AMBIGUOUS_BONUS = 0.12

MAX_ACCEPT_COST      = 0.72
OUTSIDER_DIST_M      = 1.75
OUTSIDER_CLIP_MIN_SIM = 0.58

CLIP_EMA_ALPHA  = 0.14
COLOR_EMA_ALPHA = 0.16
VEL_ALPHA       = 0.40
SIZE_ALPHA      = 0.25
MEMORY_BANK_SIZE  = 18
MEMORY_MATCH_TOPK = 5
POSE_EMA_ALPHA  = 0.22
POSE_BANK_SIZE  = 14

VELOCITY_HISTORY_FRAMES    = 300
LONG_TERM_DIRECTION_FRAMES = 0.15
LINESPACE_FRAMES           = -3
DIRECTION_COST             = 1.35

OCCLUSION_IOU_THRESH         = 0.18
OCCLUSION_EXTRA_STICKINESS   = 0.26
OCCLUSION_MOTION_BOOST       = 0.08
OCCLUSION_CLIP_REDUCE        = 0.0
OCCLUSION_FREEZE_BANK_UPDATE = True
OCCLUSION_FREEZE_EMA         = True
OCCLUSION_POSITION_FREEZE    = True

POST_OCCLUSION_RECOVERY_FRAMES = 25
POST_OCCLUSION_CLIP_WEIGHT     = 0.45
POST_OCCLUSION_COLOR_WEIGHT    = 0.18

SIDE_PENALTY       = 0.22
SIDE_GRACE_FRAMES  = 20
HARD_SIDE_PENALTY  = 0.38
HARD_SIDE_HITS_MIN = 20

AMBIGUOUS_X_PX               = 115
AMBIGUOUS_Y_PX               = 265
AMBIGUOUS_IOU_THRESH         = 0.03
AMBIGUOUS_EXTRA_SWITCH_MARGIN  = 0.40
AMBIGUOUS_EXTRA_CONFIRM_FRAMES = 10
AMBIGUOUS_FREEZE_BANK_UPDATE   = True
AMBIGUOUS_MAX_ACCEPT_COST      = 0.42

ROLE_FRONT_BACK_MARGIN_M  = 1.10
ROLE_SWAP_CONFIRM_FRAMES  = 14
ROLE_HARD_PENALTY         = 0.22
TEAMMATE_MIN_SEP_M        = 0.45

POSE_NET_BOOST_FACTOR  = 1.65
LATERAL_CROSS_WEIGHT   = 0.95
NET_ROLE_EXTRA_PENALTY = 1.25
NET_DISTANCE_THRESHOLD_M = 2.80

RESCUE_MODE_ENABLED = True

# ── motion/speed smoothing ────────────────────────────────────────────────────
SPEED_SMOOTH_ALPHA = 0.35
SPEED_CAP_KMH      = 45.0
DIST_JUMP_CAP_M    = 2.5
ACCEL_SMOOTH_ALPHA = 0.65
ACCEL_CAP_MPS2     = 11.0

# ── pose extraction ───────────────────────────────────────────────────────────
POSE_EVERY_N_FRAMES = 2
POSE_MIN_VIS        = 0.25
POSE_BBOX_EXPAND    = 0.18
POSE_CROP_MAX_W     = 320
POSE_CROP_MAX_H     = 320
POSE_POOL_SIZE      = 4

# ── ready state ───────────────────────────────────────────────────────────────
READY_KNEE_ANGLE_MAX   = 160.0
READY_HIP_DROP_MIN     = 0.04
READY_MAX_SPEED_KMH    = 6.0
READY_SMOOTH_ALPHA     = 0.35
READY_HIP_SMOOTH_ALPHA = 0.35

# ── ByteTrack ─────────────────────────────────────────────────────────────────
BYTETRACK_ACTIVATION_THRESH  = 0.35
BYTETRACK_LOST_BUFFER        = 45
BYTETRACK_MATCHING_THRESH    = 0.78

# ── shuttle detection ────────────────────────────────────────────────────────
SHUTTLE_DIFF_THRESH      = 16
SHUTTLE_BRIGHT_MIN       = 200
SHUTTLE_MIN_AREA         = 2
SHUTTLE_MAX_AREA         = 160
SHUTTLE_ERODE_ITERS      = 1
SHUTTLE_DILATE_ITERS     = 1
SHUTTLE_GATE_PX          = 160
SHUTTLE_MAX_MISSES       = 25
SHUTTLE_MASK_PLAYERS     = True
SHUTTLE_PLAYER_PAD_PX    = 26
SHUTTLE_MASK_FOOT_ONLY   = False
SHUTTLE_FOOT_ZONE_FRAC   = 0.55
SHUTTLE_UNMASK_PRED_RADIUS = 90
SHUTTLE_MAX_WH           = 26
SHUTTLE_MAX_ASPECT       = 2.6
SHUTTLE_TRAIL_LEN        = 45
SHUTTLE_DIRECTION_HISTORY = 8    # frames for direction-change detection
SHUTTLE_DIR_DOT_THRESH   = -0.3  # dot product below this → direction change

# ── reaction time analysis ────────────────────────────────────────────────────
REACTION_WINDOW_S          = 0.8   # max seconds after event to count as reaction
REACTION_MOVEMENT_THRESH_MPS = 0.5 # min speed above baseline to count as moved
REACTION_HITTING_DIST_M    = 1.5   # player within this distance is the "hitter"

# ── shot quality ──────────────────────────────────────────────────────────────
CONTACT_DIST_PX     = 120  # wrist-to-shuttle pixel distance threshold
CONTACT_WINDOW_FRAMES = 6  # frames around contact to search

# ── web app ────────────────────────────────────────────────────────────────────
WEBAPP_PORT = 5173

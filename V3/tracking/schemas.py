"""
Output schemas for V3 skeleton tracking.
Defines the Parquet column layout for skeleton.parquet.
"""

from typing import List
import pyarrow as pa

# ── landmark column names ─────────────────────────────────────────────────────
LM_COLS: List[str] = []
for _i in range(33):
    LM_COLS += [f"lm{_i}_x", f"lm{_i}_y", f"lm{_i}_vis"]

# pixel-space joints pre-computed at write time (avoid recomputing frame dims downstream)
JOINT_PX_COLS: List[str] = [
    "shoulder_L_px_x", "shoulder_L_px_y",
    "shoulder_R_px_x", "shoulder_R_px_y",
    "elbow_L_px_x",    "elbow_L_px_y",
    "elbow_R_px_x",    "elbow_R_px_y",
    "wrist_L_px_x",    "wrist_L_px_y",
    "wrist_R_px_x",    "wrist_R_px_y",
    "hip_L_px_x",      "hip_L_px_y",
    "hip_R_px_x",      "hip_R_px_y",
]

# ── V2-compatible motion/pose columns ────────────────────────────────────────
BASE_COLS: List[str] = [
    "frame_idx", "timestamp_s",
    "track_id", "stable_id", "side_group",
    "bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2",
    "foot_px_x", "foot_px_y",
    "x_m", "y_m", "zone",
    "speed_kmh", "distance_m", "accel_mps2",
    "knee_angle_L", "knee_angle_R", "knee_angle_avg",
    "ready_flag", "ready_pct", "rescue_flag",
    # V2 8-element pose signature (kept for continuity)
    "pose_sig_0", "pose_sig_1", "pose_sig_2", "pose_sig_3",
    "pose_sig_4", "pose_sig_5", "pose_sig_6", "pose_sig_7",
]

ALL_SKELETON_COLS: List[str] = BASE_COLS + LM_COLS + JOINT_PX_COLS


# ── PyArrow schema (for strict typed parquet writes) ─────────────────────────
def _f32():
    return pa.float32()

def _i32():
    return pa.int32()

def _str():
    return pa.string()

def skeleton_schema() -> pa.Schema:
    fields = [
        pa.field("frame_idx",        pa.int32()),
        pa.field("timestamp_s",      pa.float32()),
        pa.field("track_id",         pa.int32()),
        pa.field("stable_id",        pa.int32()),
        pa.field("side_group",       pa.string()),
        pa.field("bbox_x1",          pa.float32()),
        pa.field("bbox_y1",          pa.float32()),
        pa.field("bbox_x2",          pa.float32()),
        pa.field("bbox_y2",          pa.float32()),
        pa.field("foot_px_x",        pa.float32()),
        pa.field("foot_px_y",        pa.float32()),
        pa.field("x_m",              pa.float32()),
        pa.field("y_m",              pa.float32()),
        pa.field("zone",             pa.string()),
        pa.field("speed_kmh",        pa.float32()),
        pa.field("distance_m",       pa.float32()),
        pa.field("accel_mps2",       pa.float32()),
        pa.field("knee_angle_L",     pa.float32()),
        pa.field("knee_angle_R",     pa.float32()),
        pa.field("knee_angle_avg",   pa.float32()),
        pa.field("ready_flag",       pa.int8()),
        pa.field("ready_pct",        pa.float32()),
        pa.field("rescue_flag",      pa.int8()),
        pa.field("pose_sig_0",       pa.float32()),
        pa.field("pose_sig_1",       pa.float32()),
        pa.field("pose_sig_2",       pa.float32()),
        pa.field("pose_sig_3",       pa.float32()),
        pa.field("pose_sig_4",       pa.float32()),
        pa.field("pose_sig_5",       pa.float32()),
        pa.field("pose_sig_6",       pa.float32()),
        pa.field("pose_sig_7",       pa.float32()),
    ]
    # 33 landmarks × 3 columns each
    for i in range(33):
        fields += [
            pa.field(f"lm{i}_x",   pa.float32()),
            pa.field(f"lm{i}_y",   pa.float32()),
            pa.field(f"lm{i}_vis", pa.float32()),
        ]
    # pixel-space joint positions
    for col in JOINT_PX_COLS:
        fields.append(pa.field(col, pa.float32()))
    return pa.schema(fields)

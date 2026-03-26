"""Shuttle parquet schema."""

from typing import List
import pyarrow as pa

SHUTTLE_COLS: List[str] = [
    "frame_idx", "timestamp_s",
    "shuttle_px_x", "shuttle_px_y",
    "shuttle_x_m",  "shuttle_y_m",
    "shuttle_visible",      # 1=detected, 0=predicted/lost
    "shuttle_conf",
    "direction_change_flag",
    "shuttle_speed_mps",
    "nearest_player_id",
    "nearest_player_dist_m",
]


def shuttle_schema() -> pa.Schema:
    return pa.schema([
        pa.field("frame_idx",              pa.int32()),
        pa.field("timestamp_s",            pa.float32()),
        pa.field("shuttle_px_x",           pa.float32()),
        pa.field("shuttle_px_y",           pa.float32()),
        pa.field("shuttle_x_m",            pa.float32()),
        pa.field("shuttle_y_m",            pa.float32()),
        pa.field("shuttle_visible",        pa.int8()),
        pa.field("shuttle_conf",           pa.float32()),
        pa.field("direction_change_flag",  pa.int8()),
        pa.field("shuttle_speed_mps",      pa.float32()),
        pa.field("nearest_player_id",      pa.int32()),
        pa.field("nearest_player_dist_m",  pa.float32()),
    ])

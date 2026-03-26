"""
V3MetricsEngine — aggregates all parquet files into per-player PlayerSummary.
Extends V2's metrics.py patterns with reaction, shot quality, and positioning data.
"""

import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config as C

LOW_KNEE_ANGLE_DEG  = 150.0
ROLE_HEAVY_THRESH   = 60.0
EMA_ALPHA           = 0.35
HEATMAP_BINS_X      = 60
HEATMAP_BINS_Y      = 120


# ── zone parsing (port from metrics.py) ──────────────────────────────────────
def _parse_zone(z: str):
    try:
        parts = str(z).split("-")
        if len(parts) == 3:
            return parts[0], parts[1], parts[2]
    except Exception:
        pass
    return "UNK", "UNK", "UNK"


def zone_breakdown(df_player: pd.DataFrame) -> dict:
    zones = df_player["zone"].astype(str).values
    total = len(zones)
    empty = {
        "time_front_pct": 0.0, "time_back_pct": 0.0,
        "time_left_pct": 0.0, "time_right_pct": 0.0,
        "time_far_pct": 0.0, "time_near_pct": 0.0,
        "far_front_pct": 0.0, "far_back_pct": 0.0,
        "near_front_pct": 0.0, "near_back_pct": 0.0,
        "role_profile": "N/A", "top_zone": "N/A",
    }
    if total == 0:
        return empty

    sides, depths, lrs = [], [], []
    for z in zones:
        s, d, lr = _parse_zone(z)
        sides.append(s); depths.append(d); lrs.append(lr)

    far   = sum(s == "FAR"   for s in sides)
    near  = sum(s == "NEAR"  for s in sides)
    front = sum(d == "FRONT" for d in depths)
    back  = sum(d == "BACK"  for d in depths)
    left  = sum(r == "LEFT"  for r in lrs)
    right = sum(r == "RIGHT" for r in lrs)

    ff  = sum(1 for s, d in zip(sides, depths) if s == "FAR"  and d == "FRONT")
    fb  = sum(1 for s, d in zip(sides, depths) if s == "FAR"  and d == "BACK")
    nf  = sum(1 for s, d in zip(sides, depths) if s == "NEAR" and d == "FRONT")
    nb  = sum(1 for s, d in zip(sides, depths) if s == "NEAR" and d == "BACK")

    vals, counts = np.unique(zones, return_counts=True)
    top_zone = str(vals[int(np.argmax(counts))])

    t_far  = 100.0 * far  / total
    t_near = 100.0 * near / total
    t_ff   = 100.0 * ff   / max(far,  1)
    t_fb   = 100.0 * fb   / max(far,  1)
    t_nf   = 100.0 * nf   / max(near, 1)
    t_nb   = 100.0 * nb   / max(near, 1)

    prim = "FAR" if t_far >= t_near else "NEAR"
    of, ob = (t_ff, t_fb) if prim == "FAR" else (t_nf, t_nb)
    if of >= ROLE_HEAVY_THRESH:   role = "Net-heavy"
    elif ob >= ROLE_HEAVY_THRESH: role = "Back-heavy"
    else:                          role = "Rotating"

    return {
        "time_front_pct": 100.0 * front / total,
        "time_back_pct":  100.0 * back  / total,
        "time_left_pct":  100.0 * left  / total,
        "time_right_pct": 100.0 * right / total,
        "time_far_pct":   t_far,
        "time_near_pct":  t_near,
        "far_front_pct":  t_ff,
        "far_back_pct":   t_fb,
        "near_front_pct": t_nf,
        "near_back_pct":  t_nb,
        "role_profile":   role,
        "top_zone":       top_zone,
    }


# ── summary dataclass ─────────────────────────────────────────────────────────
@dataclass
class PlayerSummary:
    player_id: int

    # V2-compatible motion
    samples: int = 0
    total_distance_m: float = 0.0
    mean_speed_mps: float = 0.0
    median_speed_mps: float = 0.0
    p95_speed_mps: float = 0.0
    max_speed_mps: float = 0.0

    # zones
    time_far_pct:   float = 0.0
    time_near_pct:  float = 0.0
    time_front_pct: float = 0.0
    time_back_pct:  float = 0.0
    time_left_pct:  float = 0.0
    time_right_pct: float = 0.0
    far_front_pct:  float = 0.0
    far_back_pct:   float = 0.0
    near_front_pct: float = 0.0
    near_back_pct:  float = 0.0
    role_profile:   str = "N/A"
    top_zone:       str = "N/A"

    # ready / knee
    ready_time_pct:      Optional[float] = None
    ready_time_s:        Optional[float] = None
    ready_pct_mean:      Optional[float] = None
    knee_angle_mean_deg:   Optional[float] = None
    knee_angle_median_deg: Optional[float] = None
    knee_angle_p25_deg:    Optional[float] = None
    knee_angle_p75_deg:    Optional[float] = None
    low_knee_time_pct:     Optional[float] = None

    # V3 reaction
    mean_reaction_time_s: Optional[float] = None
    p25_reaction_time_s:  Optional[float] = None
    p75_reaction_time_s:  Optional[float] = None
    reaction_events:      int = 0

    # V3 shots
    shots_total:           int = 0
    shot_type_breakdown:   dict = field(default_factory=dict)
    mean_elbow_angle_R:    Optional[float] = None
    mean_wrist_height_rel_shoulder: Optional[float] = None
    mean_hip_rotation:     Optional[float] = None

    # V3 positioning
    mean_team_spread_m:  Optional[float] = None
    crossing_events_count: int = 0


# ── metrics engine ────────────────────────────────────────────────────────────
class V3MetricsEngine:
    def compute_player_summary(
        self,
        pid: int,
        skeleton_df: pd.DataFrame,
        shuttle_df: Optional[pd.DataFrame] = None,
        positioning_df: Optional[pd.DataFrame] = None,
        reaction_df: Optional[pd.DataFrame] = None,
        shots_df: Optional[pd.DataFrame] = None,
    ) -> PlayerSummary:

        ps = PlayerSummary(player_id=pid)
        df = skeleton_df[skeleton_df["stable_id"] == pid].copy()
        df = df.sort_values("frame_idx").drop_duplicates(subset=["frame_idx"], keep="last")

        if df.empty:
            return ps

        ps.samples = len(df)

        # ── motion ────────────────────────────────────────────────────────────
        # re-derive smoothed speed from position deltas (same as metrics.py)
        df["dx"] = df["x_m"].diff()
        df["dy"] = df["y_m"].diff()
        df["dt"] = df["timestamp_s"].diff()
        df["step_m"] = np.sqrt(df["dx"] ** 2 + df["dy"] ** 2)
        df.loc[df["dt"] <= 0, "step_m"] = np.nan
        raw_speed = (df["step_m"] / df["dt"]).clip(0, 12).fillna(0).to_numpy()

        # EMA smooth
        speed = np.empty_like(raw_speed)
        speed[0] = raw_speed[0]
        for i in range(1, len(raw_speed)):
            speed[i] = EMA_ALPHA * raw_speed[i] + (1 - EMA_ALPHA) * speed[i - 1]

        valid = speed[np.isfinite(speed)]
        ps.total_distance_m  = float(df["step_m"].clip(0, C.DIST_JUMP_CAP_M).sum(skipna=True))
        ps.mean_speed_mps    = float(np.mean(valid))   if len(valid) else 0.0
        ps.median_speed_mps  = float(np.median(valid)) if len(valid) else 0.0
        ps.p95_speed_mps     = float(np.percentile(valid, 95)) if len(valid) else 0.0
        ps.max_speed_mps     = float(np.max(valid))    if len(valid) else 0.0

        # ── zones ─────────────────────────────────────────────────────────────
        zb = zone_breakdown(df)
        for k, v in zb.items():
            setattr(ps, k, v)

        # ── ready / knee ──────────────────────────────────────────────────────
        if "ready_flag" in df.columns:
            ts = df["timestamp_s"].to_numpy()
            dt_arr = np.diff(ts, prepend=np.nan)
            dt_arr[~np.isfinite(dt_arr)] = 0.0
            dt_arr = np.clip(dt_arr, 0, 1)
            rf = pd.to_numeric(df["ready_flag"], errors="coerce").fillna(0).astype(int).to_numpy()
            total_t = float(np.sum(dt_arr))
            ready_t = float(np.sum(dt_arr[rf == 1]))
            ps.ready_time_s   = ready_t
            ps.ready_time_pct = (100.0 * ready_t / total_t) if total_t > 1e-6 else 0.0

        if "ready_pct" in df.columns:
            rp = pd.to_numeric(df["ready_pct"], errors="coerce").dropna().to_numpy()
            ps.ready_pct_mean = float(np.mean(rp)) if len(rp) else None

        if "knee_angle_avg" in df.columns:
            ka = pd.to_numeric(df["knee_angle_avg"], errors="coerce").dropna().to_numpy()
            if len(ka):
                ps.knee_angle_mean_deg   = float(np.mean(ka))
                ps.knee_angle_median_deg = float(np.median(ka))
                ps.knee_angle_p25_deg    = float(np.percentile(ka, 25))
                ps.knee_angle_p75_deg    = float(np.percentile(ka, 75))
                full = pd.to_numeric(df["knee_angle_avg"], errors="coerce").to_numpy()
                m    = np.isfinite(full)
                if np.sum(m) > 0:
                    ps.low_knee_time_pct = float(100.0 * np.sum(full[m] <= LOW_KNEE_ANGLE_DEG) / np.sum(m))

        # ── reaction times ────────────────────────────────────────────────────
        if reaction_df is not None and not reaction_df.empty:
            col = f"p{pid}_reaction_time_s"
            if col in reaction_df.columns:
                rt = reaction_df[col].dropna().to_numpy()
                if len(rt):
                    ps.mean_reaction_time_s = float(np.mean(rt))
                    ps.p25_reaction_time_s  = float(np.percentile(rt, 25))
                    ps.p75_reaction_time_s  = float(np.percentile(rt, 75))
                    ps.reaction_events      = int(len(rt))

        # ── shot quality ──────────────────────────────────────────────────────
        if shots_df is not None and not shots_df.empty:
            p_shots = shots_df[shots_df["player_id"] == pid]
            ps.shots_total = len(p_shots)
            if not p_shots.empty:
                if "shot_type" in p_shots.columns:
                    ps.shot_type_breakdown = p_shots["shot_type"].value_counts().to_dict()
                if "elbow_angle_R" in p_shots.columns:
                    ps.mean_elbow_angle_R = float(p_shots["elbow_angle_R"].dropna().mean()) \
                        if p_shots["elbow_angle_R"].notna().any() else None
                if "wrist_height_rel_shoulder" in p_shots.columns:
                    ps.mean_wrist_height_rel_shoulder = float(p_shots["wrist_height_rel_shoulder"].dropna().mean()) \
                        if p_shots["wrist_height_rel_shoulder"].notna().any() else None
                if "hip_rotation_at_contact" in p_shots.columns:
                    ps.mean_hip_rotation = float(p_shots["hip_rotation_at_contact"].dropna().mean()) \
                        if p_shots["hip_rotation_at_contact"].notna().any() else None

        # ── positioning ───────────────────────────────────────────────────────
        if positioning_df is not None and not positioning_df.empty:
            spread_col = None
            for c in ["team_spread_far_m", "team_spread_near_m"]:
                if c in positioning_df.columns:
                    if pid in [1, 2] and c == "team_spread_far_m":
                        spread_col = c
                    elif pid in [3, 4] and c == "team_spread_near_m":
                        spread_col = c
            if spread_col:
                ps.mean_team_spread_m = float(positioning_df[spread_col].dropna().mean())

            if "crossing_event" in positioning_df.columns:
                crosses = positioning_df[
                    (positioning_df["crossing_event"] == 1) &
                    (positioning_df.get("side_crossing_player_id", pd.Series()) == pid)
                ]
                ps.crossing_events_count = len(crosses)

        return ps

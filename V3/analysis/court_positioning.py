"""
CourtPositioningAnalyzer — Module 2.

Reads skeleton.parquet and computes per-frame:
- Pairwise player distances
- Team spread (near/far side)
- Court overlap percentage
- Net-crossing events
"""

import math
import os
from typing import List, Optional

import numpy as np
import pandas as pd

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config as C
from shared.io_utils import ensure_dir

_CROSSING_ROLLING_FRAMES = 15   # median window for crossing detection


class CourtPositioningAnalyzer:
    def __init__(self, court_l: float = None):
        self._court_l = court_l or C.COURT_L

    def run(self, skeleton_path: str, output_dir: str) -> str:
        """Returns path to positioning.parquet."""
        ensure_dir(output_dir)

        df = pd.read_parquet(skeleton_path, columns=[
            "frame_idx", "timestamp_s", "stable_id", "x_m", "y_m",
        ])
        df = df[df["stable_id"].isin(C.PLAYER_IDS)].copy()
        df.sort_values(["frame_idx", "stable_id"], inplace=True)

        rows = []
        for frame_idx, grp in df.groupby("frame_idx", sort=True):
            row = self._process_frame(int(frame_idx), grp)
            rows.append(row)

        positioning_df = pd.DataFrame(rows)

        # ── crossing detection via rolling median ────────────────────────────
        # For each player, compute a rolling median of y_m to smooth noise,
        # then detect when the median crosses COURT_L / 2.0.
        pivot_y = df.pivot_table(index="frame_idx", columns="stable_id", values="y_m", aggfunc="first")
        mid = self._court_l / 2.0
        crossing_evt = pd.Series(0, index=pivot_y.index)
        crossing_pid = pd.Series(-1, index=pivot_y.index)

        for pid in C.PLAYER_IDS:
            if pid not in pivot_y.columns:
                continue
            y_series = pivot_y[pid].fillna(method="ffill").fillna(method="bfill")
            rolling  = y_series.rolling(_CROSSING_ROLLING_FRAMES, center=True, min_periods=3).median()
            side_prev = (rolling < mid).astype(int)
            side_prev_shifted = side_prev.shift(1).fillna(side_prev)
            crossed = (side_prev != side_prev_shifted).astype(int)
            idx_cross = crossed[crossed == 1].index
            for fi in idx_cross:
                crossing_evt.loc[fi] = 1
                crossing_pid.loc[fi] = pid

        positioning_df = positioning_df.set_index("frame_idx")
        positioning_df["crossing_event"]       = crossing_evt.reindex(positioning_df.index).fillna(0).astype(int)
        positioning_df["side_crossing_player_id"] = crossing_pid.reindex(positioning_df.index).fillna(-1).astype(int)
        positioning_df = positioning_df.reset_index()

        out_path = os.path.join(output_dir, "positioning.parquet")
        positioning_df.to_parquet(out_path, engine="pyarrow", compression="snappy", index=False)
        print(f"V3 — positioning.parquet written: {len(positioning_df)} rows")
        return out_path

    def _process_frame(self, frame_idx: int, grp: pd.DataFrame) -> dict:
        """Compute all positioning metrics for a single frame group."""
        players = {int(r.stable_id): (float(r.x_m), float(r.y_m))
                   for r in grp.itertuples(index=False)
                   if int(r.stable_id) in C.PLAYER_IDS}

        row: dict = {"frame_idx": frame_idx}

        # pairwise distances
        pid_list = sorted(players.keys())
        for i in range(len(pid_list)):
            for j in range(i + 1, len(pid_list)):
                a, b = pid_list[i], pid_list[j]
                xa, ya = players[a]
                xb, yb = players[b]
                dist = math.hypot(xa - xb, ya - yb)
                row[f"dist_p{a}_p{b}_m"] = float(dist)

        # fill missing pairs with NaN
        for i in range(1, 5):
            for j in range(i + 1, 5):
                k = f"dist_p{i}_p{j}_m"
                if k not in row:
                    row[k] = float("nan")

        # team spread (players 1+2 = far team, 3+4 = near team by convention)
        for team_ids, key in [([1, 2], "team_spread_far_m"), ([3, 4], "team_spread_near_m")]:
            pts = [players[p] for p in team_ids if p in players]
            if len(pts) == 2:
                row[key] = float(math.hypot(pts[0][0] - pts[1][0], pts[0][1] - pts[1][1]))
            else:
                row[key] = float("nan")

        # court overlap: fraction of court width where two players are within 1m laterally
        all_x = [v[0] for v in players.values()]
        if len(all_x) >= 2:
            x_range = max(all_x) - min(all_x)
            row["court_overlap_pct"] = float(max(0.0, 1.0 - x_range / C.COURT_W) * 100.0)
        else:
            row["court_overlap_pct"] = float("nan")

        return row

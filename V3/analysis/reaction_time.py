"""
ReactionAnalyzer — Module 4.

Measures how quickly non-hitting players move toward the shuttle
after each shuttle event (direction change or re-appearance).
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

# minimum frames the shuttle must be lost before re-appearance counts as event
_REAPPEAR_MIN_LOST = 8


class ReactionAnalyzer:
    def run(self, shuttle_path: str, skeleton_path: str, fps: float, output_dir: str) -> str:
        """Returns path to reaction.parquet."""
        ensure_dir(output_dir)

        shuttle = pd.read_parquet(shuttle_path)
        skel    = pd.read_parquet(skeleton_path, columns=[
            "frame_idx", "stable_id", "x_m", "y_m", "speed_kmh",
        ])
        skel = skel[skel["stable_id"].isin(C.PLAYER_IDS)]

        events = self._find_events(shuttle)
        rows = []
        for ev in events:
            row = self._measure_reactions(ev, shuttle, skel, fps)
            rows.append(row)

        df = pd.DataFrame(rows)
        out_path = os.path.join(output_dir, "reaction.parquet")
        df.to_parquet(out_path, engine="pyarrow", compression="snappy", index=False)
        print(f"V3 — reaction.parquet written: {len(df)} events")
        return out_path

    # ── event detection ───────────────────────────────────────────────────────
    def _find_events(self, shuttle: pd.DataFrame) -> List[dict]:
        events = []

        # direction change events
        dir_frames = shuttle[shuttle["direction_change_flag"] == 1]
        for row in dir_frames.itertuples(index=False):
            events.append({
                "event_frame_idx":  int(row.frame_idx),
                "event_timestamp_s": float(row.timestamp_s),
                "shuttle_event_type": "direction_change",
                "shuttle_x_m":       float(row.shuttle_x_m),
                "shuttle_y_m":       float(row.shuttle_y_m),
                "nearest_player_id": int(row.nearest_player_id),
            })

        # re-appearance events: shuttle visible after ≥ _REAPPEAR_MIN_LOST lost frames
        vis   = shuttle["shuttle_visible"].to_numpy()
        frame = shuttle["frame_idx"].to_numpy()
        lost_count = 0
        for i in range(1, len(vis)):
            if vis[i - 1] == 0:
                lost_count += 1
            else:
                lost_count = 0
            if vis[i] == 1 and vis[i - 1] == 0 and lost_count >= _REAPPEAR_MIN_LOST:
                row = shuttle.iloc[i]
                events.append({
                    "event_frame_idx":   int(row.frame_idx),
                    "event_timestamp_s": float(row.timestamp_s),
                    "shuttle_event_type": "shuttle_appears",
                    "shuttle_x_m":        float(row.shuttle_x_m),
                    "shuttle_y_m":        float(row.shuttle_y_m),
                    "nearest_player_id":  int(row.nearest_player_id),
                })

        # sort by time, remove events within 0.5s of each other
        events.sort(key=lambda e: e["event_frame_idx"])
        filtered = []
        last_t = -999.0
        for ev in events:
            if ev["event_timestamp_s"] - last_t >= 0.5:
                filtered.append(ev)
                last_t = ev["event_timestamp_s"]
        return filtered

    # ── reaction measurement ──────────────────────────────────────────────────
    def _measure_reactions(self, event: dict, shuttle: pd.DataFrame,
                           skel: pd.DataFrame, fps: float) -> dict:
        ev_frame = event["event_frame_idx"]
        ev_t     = event["event_timestamp_s"]
        window_frames = int(C.REACTION_WINDOW_S * fps) + 1

        row = {
            "event_frame_idx":    ev_frame,
            "event_timestamp_s":  ev_t,
            "shuttle_event_type": event["shuttle_event_type"],
            "shuttle_x_m":        event["shuttle_x_m"],
            "shuttle_y_m":        event["shuttle_y_m"],
            "hitting_player_id":  event["nearest_player_id"],
        }

        for pid in C.PLAYER_IDS:
            row[f"p{pid}_reaction_time_s"] = float("nan")

        for pid in C.PLAYER_IDS:
            if pid == event["nearest_player_id"]:
                continue  # skip the hitting player

            # get this player's speed series around the event
            player_data = skel[(skel["stable_id"] == pid) &
                                (skel["frame_idx"] >= ev_frame) &
                                (skel["frame_idx"] <= ev_frame + window_frames)].copy()
            if player_data.empty:
                continue

            # baseline speed: mean speed in 1s before event
            pre_data = skel[(skel["stable_id"] == pid) &
                            (skel["frame_idx"] >= ev_frame - int(fps)) &
                            (skel["frame_idx"] < ev_frame)]
            baseline_mps = 0.0
            if not pre_data.empty:
                baseline_mps = float(pre_data["speed_kmh"].mean() / 3.6)

            threshold = baseline_mps + C.REACTION_MOVEMENT_THRESH_MPS

            # find first frame where speed exceeds threshold
            for rr in player_data.sort_values("frame_idx").itertuples(index=False):
                sp_mps = float(rr.speed_kmh) / 3.6
                if sp_mps >= threshold:
                    react_t = float(rr.frame_idx) / fps - ev_t
                    if react_t >= 0.0:
                        row[f"p{pid}_reaction_time_s"] = round(react_t, 4)
                    break

        return row

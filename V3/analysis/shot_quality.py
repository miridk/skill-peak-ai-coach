"""
ShotQualityAnalyzer — Module 5.

Detects contact frames (wrist pixel distance to shuttle < threshold)
and extracts biomechanical features for each shot.
"""

import math
import os
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config as C
from shared.io_utils import ensure_dir


def _angle_deg(a: Tuple, b: Tuple, c: Tuple) -> float:
    v1 = np.array([a[0] - b[0], a[1] - b[1]])
    v2 = np.array([c[0] - b[0], c[1] - b[1]])
    n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
    if n1 < 1e-6 or n2 < 1e-6:
        return 180.0
    cos = np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0)
    return float(np.degrees(np.arccos(cos)))


class ShotQualityAnalyzer:
    def run(self, shuttle_path: str, skeleton_path: str, output_dir: str) -> str:
        """Returns path to shots.parquet."""
        ensure_dir(output_dir)

        shuttle = pd.read_parquet(shuttle_path)
        skel    = pd.read_parquet(skeleton_path)

        # keep only visible shuttle frames
        vis_shuttle = shuttle[shuttle["shuttle_visible"] == 1].copy()

        # build per-frame wrist lookup for efficiency
        wrist_cols = [
            "frame_idx", "stable_id",
            "wrist_L_px_x", "wrist_L_px_y",
            "wrist_R_px_x", "wrist_R_px_y",
            "elbow_L_px_x", "elbow_L_px_y",
            "elbow_R_px_x", "elbow_R_px_y",
            "shoulder_L_px_x", "shoulder_L_px_y",
            "shoulder_R_px_x", "shoulder_R_px_y",
            "hip_L_px_x", "hip_L_px_y",
            "hip_R_px_x", "hip_R_px_y",
            "x_m", "y_m",
            "lm23_x", "lm23_y",   # hip_L normalised
            "lm24_x", "lm24_y",   # hip_R
            "lm11_x", "lm11_y",   # shoulder_L
            "lm12_x", "lm12_y",   # shoulder_R
        ]
        avail = [c for c in wrist_cols if c in skel.columns]
        wrist_df = skel[avail].copy()

        contacts = self._detect_contacts(vis_shuttle, wrist_df)
        shots = [self._extract_features(c, wrist_df, shuttle) for c in contacts]

        df = pd.DataFrame(shots)
        out_path = os.path.join(output_dir, "shots.parquet")
        df.to_parquet(out_path, engine="pyarrow", compression="snappy", index=False)
        print(f"V3 — shots.parquet written: {len(df)} shots detected")
        return out_path

    # ── contact detection ─────────────────────────────────────────────────────
    def _detect_contacts(self, vis_shuttle: pd.DataFrame, wrist_df: pd.DataFrame) -> List[dict]:
        """
        For each shuttle-visible frame find the player/wrist combination
        closest to the shuttle. When distance dips below CONTACT_DIST_PX
        then rises again, record the trough as the contact frame.
        """
        contacts = []
        window = C.CONTACT_WINDOW_FRAMES

        # build per-frame min distance series: (frame_idx, min_dist, player_id, arm)
        series = []
        for row in vis_shuttle.itertuples(index=False):
            fi = int(row.frame_idx)
            sx = float(row.shuttle_px_x)
            sy = float(row.shuttle_px_y)
            if not (math.isfinite(sx) and math.isfinite(sy)):
                continue
            frame_players = wrist_df[wrist_df["frame_idx"] == fi]
            best_dist = 1e9
            best_pid  = -1
            best_arm  = "R"
            for pr in frame_players.itertuples(index=False):
                for arm, xcol, ycol in [("L", "wrist_L_px_x", "wrist_L_px_y"),
                                         ("R", "wrist_R_px_x", "wrist_R_px_y")]:
                    wx = getattr(pr, xcol, float("nan"))
                    wy = getattr(pr, ycol, float("nan"))
                    if not (math.isfinite(wx) and math.isfinite(wy)):
                        continue
                    d = math.hypot(sx - wx, sy - wy)
                    if d < best_dist:
                        best_dist = d
                        best_pid  = int(pr.stable_id)
                        best_arm  = arm
            series.append({"frame_idx": fi, "min_dist": best_dist,
                            "player_id": best_pid, "arm": best_arm})

        if not series:
            return []

        # find local minima below CONTACT_DIST_PX
        dists = [s["min_dist"] for s in series]
        for i in range(1, len(dists) - 1):
            if (dists[i] < C.CONTACT_DIST_PX
                    and dists[i] <= dists[i - 1]
                    and dists[i] <= dists[i + 1]
                    and series[i]["player_id"] > 0):
                contacts.append({
                    "contact_frame_idx": series[i]["frame_idx"],
                    "player_id":         series[i]["player_id"],
                    "dominant_arm":      series[i]["arm"],
                    "min_dist_px":       dists[i],
                })

        # deduplicate: if two contacts within 5 frames keep closer one
        deduped = []
        last_fi = -999
        for c in sorted(contacts, key=lambda x: x["contact_frame_idx"]):
            if c["contact_frame_idx"] - last_fi > 5:
                deduped.append(c)
                last_fi = c["contact_frame_idx"]
            elif c["min_dist_px"] < deduped[-1]["min_dist_px"]:
                deduped[-1] = c
        return deduped

    # ── feature extraction ────────────────────────────────────────────────────
    def _extract_features(self, contact: dict, wrist_df: pd.DataFrame,
                          shuttle: pd.DataFrame) -> dict:
        fi     = contact["contact_frame_idx"]
        pid    = contact["player_id"]
        arm    = contact["dominant_arm"]

        pr_rows = wrist_df[(wrist_df["frame_idx"] == fi) & (wrist_df["stable_id"] == pid)]
        pr = pr_rows.iloc[0] if not pr_rows.empty else None

        # shuttle position at contact
        sh_row = shuttle[shuttle["frame_idx"] == fi]
        sh_xm = float(sh_row["shuttle_x_m"].iloc[0]) if not sh_row.empty else float("nan")
        sh_ym = float(sh_row["shuttle_y_m"].iloc[0]) if not sh_row.empty else float("nan")
        sh_t  = float(sh_row["timestamp_s"].iloc[0]) if not sh_row.empty else float("nan")

        row = {
            "contact_frame_idx":  fi,
            "contact_timestamp_s": sh_t,
            "player_id":          pid,
            "shuttle_x_m":        sh_xm,
            "shuttle_y_m":        sh_ym,
            "dominant_arm":       arm,
        }

        # biomechanical features (all float, NaN if landmarks unavailable)
        elbow_L = elbow_R = float("nan")
        shoulder_abduction_R = float("nan")
        wrist_height_rel_shoulder = float("nan")
        body_lean = float("nan")
        hip_rotation = float("nan")

        if pr is not None:
            # elbow angles
            def _px(col):
                v = getattr(pr, col, float("nan"))
                return float(v) if math.isfinite(float(v)) else float("nan")

            sLx, sLy = _px("shoulder_L_px_x"), _px("shoulder_L_px_y")
            sRx, sRy = _px("shoulder_R_px_x"), _px("shoulder_R_px_y")
            eLx, eLy = _px("elbow_L_px_x"),    _px("elbow_L_px_y")
            eRx, eRy = _px("elbow_R_px_x"),    _px("elbow_R_px_y")
            wLx, wLy = _px("wrist_L_px_x"),    _px("wrist_L_px_y")
            wRx, wRy = _px("wrist_R_px_x"),    _px("wrist_R_px_y")
            hLx, hLy = _px("hip_L_px_x"),      _px("hip_L_px_y")
            hRx, hRy = _px("hip_R_px_x"),      _px("hip_R_px_y")

            if all(math.isfinite(v) for v in [sLx, eLx, wLx]):
                elbow_L = _angle_deg((sLx, sLy), (eLx, eLy), (wLx, wLy))
            if all(math.isfinite(v) for v in [sRx, eRx, wRx]):
                elbow_R = _angle_deg((sRx, sRy), (eRx, eRy), (wRx, wRy))

            # wrist height relative to shoulder (normalised)
            if arm == "R" and math.isfinite(wRy) and math.isfinite(sRy):
                # negative = wrist above shoulder (overhead shot)
                wrist_height_rel_shoulder = float(wRy - sRy)
            elif arm == "L" and math.isfinite(wLy) and math.isfinite(sLy):
                wrist_height_rel_shoulder = float(wLy - sLy)

            # body lean (hip midpoint lateral offset from shoulder midpoint, normalised)
            if all(math.isfinite(v) for v in [sLx, sRx, hLx, hRx]):
                smid_x = (sLx + sRx) / 2.0
                hmid_x = (hLx + hRx) / 2.0
                body_w  = max(1.0, abs(sLx - sRx))
                body_lean = float((hmid_x - smid_x) / body_w)

            # hip rotation angle (shoulder line vs hip line in degrees)
            if all(math.isfinite(v) for v in [sLx, sRx, hLx, hRx]):
                shoulder_angle = math.degrees(math.atan2(sRy - sLy, sRx - sLx))
                hip_angle      = math.degrees(math.atan2(hRy - hLy, hRx - hLx))
                hip_rotation   = float(abs(shoulder_angle - hip_angle))
                if hip_rotation > 180.0:
                    hip_rotation = 360.0 - hip_rotation

        row.update({
            "elbow_angle_L":             elbow_L,
            "elbow_angle_R":             elbow_R,
            "shoulder_abduction_R":      shoulder_abduction_R,
            "wrist_height_rel_shoulder": wrist_height_rel_shoulder,
            "body_lean_at_contact":      body_lean,
            "hip_rotation_at_contact":   hip_rotation,
            "shot_type":                 self._classify_shot(
                wrist_height_rel_shoulder, sh_ym),
        })

        # prep_frames: frames since last direction change before contact
        sh_before = shuttle[shuttle["frame_idx"] <= fi]
        dir_changes = sh_before[sh_before["direction_change_flag"] == 1]
        if not dir_changes.empty:
            last_dc_fi = int(dir_changes.iloc[-1]["frame_idx"])
            row["prep_frames"] = fi - last_dc_fi
        else:
            row["prep_frames"] = -1

        return row

    @staticmethod
    def _classify_shot(wrist_height_rel_shoulder: float, shuttle_y_m: float) -> str:
        if not math.isfinite(wrist_height_rel_shoulder):
            return "unknown"
        if math.isfinite(shuttle_y_m) and shuttle_y_m < 2.0:
            return "net_drop"
        if wrist_height_rel_shoulder < -20:   # wrist well above shoulder (pixels)
            return "overhead_clear"
        if wrist_height_rel_shoulder < 0:
            return "smash"
        return "drive"

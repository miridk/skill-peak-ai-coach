"""
IdentityManager — verbatim port from V2/badminton_analyzer.py (lines 827-1451).
Reads constants from V3/config.py instead of V2 module-level globals.
"""

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.optimize import linear_sum_assignment

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config as C
from shared.calibration import COURT_W, COURT_L
from shared.court import side_from_y, depth_role_from_y
from shared.io_utils import clamp01, bbox_iou_xyxy


# ── helpers reused from V2 ───────────────────────────────────────────────────
def cosine_sim(a, b) -> float:
    if a is None or b is None:
        return 0.0
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na < 1e-8 or nb < 1e-8:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def safe_hist_corr(a, b) -> float:
    import cv2
    if a is None or b is None:
        return 0.0
    try:
        sim = cv2.compareHist(a.astype(np.float32), b.astype(np.float32), cv2.HISTCMP_CORREL)
        return clamp01((sim + 1.0) * 0.5)
    except Exception:
        return 0.0


def best_bank_similarity(bank: List[np.ndarray], feat: Optional[np.ndarray], topk: int = 5) -> float:
    if feat is None or not bank:
        return 0.0
    weights = np.linspace(0.4, 1.0, len(bank))
    weights /= weights.sum()
    sims = [clamp01((cosine_sim(b, feat) + 1.0) * 0.5) * weights[i] for i, b in enumerate(bank)]
    sims.sort(reverse=True)
    return float(sum(sims[:topk]) / max(1, len(sims[:topk])))


# ── data classes ─────────────────────────────────────────────────────────────
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

    vx_history: List[float] = field(default_factory=list)
    vy_history: List[float] = field(default_factory=list)
    hits: int = 1
    pending_raw_tid: Optional[int] = None
    pending_count: int = 0
    preferred_side: Optional[str] = None
    side_stable_frames: int = 0
    preferred_role: Optional[str] = None
    role_stable_frames: int = 0
    pre_occlusion_clip: Optional[np.ndarray] = None
    pre_occlusion_color: Optional[np.ndarray] = None
    pre_occlusion_pose: Optional[np.ndarray] = None
    was_occluded: bool = False
    post_occlusion_frames: int = 0


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
            best_score = -float("inf")
            for i, det in enumerate(detections):
                if i in used:
                    continue
                cx = 0.5 * (det["bbox_x1"] + det["bbox_x2"])
                cy = 0.5 * (det["bbox_y1"] + det["bbox_y2"])
                dist = math.hypot(cx - click_x, cy - click_y)
                inside_bonus = 400.0 if (
                    det["bbox_x1"] <= click_x <= det["bbox_x2"]
                    and det["bbox_y1"] <= click_y <= det["bbox_y2"]
                ) else 0.0
                score = inside_bonus - dist * 10.0
                if score > best_score:
                    best_score = score
                    best_i = i
            if best_i is not None:
                used.add(best_i)
                det = detections[best_i]
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
                    clip_bank=[det["clip_feat"]] if det.get("clip_feat") is not None else [],
                    color_bank=[det["color_feat"]] if det.get("color_feat") is not None else [],
                    pose_bank=[det["pose_feat"]] if det.get("pose_feat") is not None else [],
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
        history_len = min(len(tr.vx_history), C.VELOCITY_HISTORY_FRAMES)
        if history_len == 0:
            return tr.last_x, tr.last_y
        weights = np.exp(np.linspace(0, C.LINESPACE_FRAMES, history_len))
        weights /= weights.sum()
        weighted_vx = np.sum(np.array(tr.vx_history[-history_len:]) * weights)
        weighted_vy = np.sum(np.array(tr.vy_history[-history_len:]) * weights)
        dtf = max(1, frame_idx - tr.last_seen_frame)
        return tr.last_x + weighted_vx * dtf, tr.last_y + weighted_vy * dtf

    def _direction_cost(self, tr: IdentityTrack, det: dict, frame_idx: int) -> float:
        if tr.hits < 6:
            return 0.0
        pred_x, pred_y = self._predict_xy(tr, frame_idx)
        dx_pred, dy_pred = pred_x - tr.last_x, pred_y - tr.last_y
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
        dx_pred, dy_pred = pred_x - tr.last_x, pred_y - tr.last_y
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
        return direction_cost * C.DIRECTION_COST if cos_sim < -0.2 else direction_cost * 0.35

    def _clip_cost(self, tr: IdentityTrack, det: dict, occluded: bool) -> float:
        feat = det.get("clip_feat")
        sim_bank = best_bank_similarity(tr.clip_bank, feat, topk=C.MEMORY_MATCH_TOPK)
        sim_ema = cosine_sim(tr.clip_feat, feat) if tr.clip_feat is not None and feat is not None else 0.0
        sim_ema = clamp01((sim_ema + 1.0) * 0.5)
        if feat is None:
            return 0.45
        return 1.0 - max(sim_bank, sim_ema)

    def _motion_cost(self, tr: IdentityTrack, det: dict, frame_idx: int, occluded: bool) -> float:
        px, py = self._predict_xy(tr, frame_idx)
        d = math.hypot(float(det["x_m"]) - px, float(det["y_m"]) - py)
        base = clamp01(d / max(1e-6, self.max_match_dist_m))
        if tr.hits >= 12:
            base *= 0.92
        if occluded:
            base = max(0.0, base - C.OCCLUSION_MOTION_BOOST)
        return base

    def _color_cost(self, tr: IdentityTrack, det: dict) -> float:
        feat = det.get("color_feat")
        sim_bank = 0.0
        if feat is not None and tr.color_bank:
            sims = [safe_hist_corr(b, feat) for b in tr.color_bank]
            sims.sort(reverse=True)
            sim_bank = float(sum(sims[:4]) / max(1, len(sims[:4])))
        sim_ema = safe_hist_corr(tr.color_feat, feat) if feat is not None and tr.color_feat is not None else 0.0
        if feat is None:
            return 0.35
        return 1.0 - max(sim_bank, sim_ema)

    def _pose_cost(self, tr: IdentityTrack, det: dict, ambiguous: bool) -> float:
        feat = det.get("pose_feat")
        sim_bank = best_bank_similarity(tr.pose_bank, feat, topk=3)
        sim_ema = cosine_sim(tr.pose_feat, feat) if tr.pose_feat is not None and feat is not None else 0.0
        sim_ema = clamp01((sim_ema + 1.0) * 0.5)
        if feat is None:
            return 0.42 if ambiguous else 0.32
        return 1.0 - max(sim_bank, sim_ema)

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
        if tr.hits >= C.HARD_SIDE_HITS_MIN and tr.side_stable_frames >= C.SIDE_GRACE_FRAMES:
            return C.HARD_SIDE_PENALTY
        return C.SIDE_PENALTY if tr.side_stable_frames < C.SIDE_GRACE_FRAMES else C.SIDE_PENALTY * 0.55

    def _role_cost(self, tr: IdentityTrack, det: dict) -> float:
        if tr.preferred_role is None:
            return 0.0
        role_now = depth_role_from_y(float(det["y_m"]))
        if role_now == tr.preferred_role:
            return 0.0
        return C.ROLE_HARD_PENALTY if tr.role_stable_frames >= C.ROLE_SWAP_CONFIRM_FRAMES else C.ROLE_HARD_PENALTY * 0.55

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
        clip_c      = self._clip_cost(tr, det, occluded)
        motion_c    = self._motion_cost(tr, det, frame_idx, occluded)
        color_c     = self._color_cost(tr, det)
        size_c      = self._size_cost(tr, det)
        side_c      = self._side_cost(tr, det)
        edge_c      = self._edge_cost(det)
        pose_c      = self._pose_cost(tr, det, ambiguous)
        role_c      = self._role_cost(tr, det)
        lateral_c   = self._lateral_cross_penalty(tr, det)
        direction_c = self._direction_cost(tr, det, frame_idx)
        lt_dir_c    = self._long_term_direction_cost(tr, det, frame_idx)

        dist_to_net = abs(float(det["y_m"]) - COURT_L / 2.0)
        near_net    = dist_to_net < C.NET_DISTANCE_THRESHOLD_M
        w_pose      = C.W_POSE * (C.POSE_NET_BOOST_FACTOR if near_net else 1.0)
        role_final  = role_c * (C.NET_ROLE_EXTRA_PENALTY if near_net else 1.0)

        base = (
            C.W_CLIP      * clip_c
            + C.W_MOTION  * motion_c
            + C.W_COLOR   * color_c
            + C.W_SIZE    * size_c
            + C.W_SIDE    * side_c
            + C.W_EDGE    * edge_c
            + w_pose       * pose_c
            + role_final
            + C.LATERAL_CROSS_WEIGHT * lateral_c
            + C.W_DIRECTION           * direction_c
            + C.LONG_TERM_DIRECTION_FRAMES * lt_dir_c
        )

        if tr.post_occlusion_frames > 0:
            recovery_w = tr.post_occlusion_frames / float(C.POST_OCCLUSION_RECOVERY_FRAMES)
            if tr.pre_occlusion_clip is not None:
                snap_sim = cosine_sim(tr.pre_occlusion_clip, det.get("clip_feat"))
                snap_sim = clamp01((snap_sim + 1.0) * 0.5)
                base += C.POST_OCCLUSION_CLIP_WEIGHT * recovery_w * (1.0 - snap_sim)
            if tr.pre_occlusion_color is not None:
                snap_color_sim = safe_hist_corr(tr.pre_occlusion_color, det.get("color_feat"))
                base += C.POST_OCCLUSION_COLOR_WEIGHT * recovery_w * (1.0 - snap_color_sim)
        return base

    def _update_side_memory(self, tr: IdentityTrack, det: dict):
        current_side = side_from_y(float(det["y_m"]))
        if tr.preferred_side is None:
            tr.preferred_side = current_side
            tr.side_stable_frames = 1
            return
        if current_side == tr.preferred_side:
            tr.side_stable_frames += 1
        else:
            if tr.side_stable_frames > C.SIDE_GRACE_FRAMES:
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
            if tr.role_stable_frames > C.ROLE_SWAP_CONFIRM_FRAMES:
                tr.preferred_role = role_now
                tr.role_stable_frames = 1
            else:
                tr.role_stable_frames = max(0, tr.role_stable_frames - 1)

    def _compute_ambiguity_flags(self, detections: List[dict]) -> List[bool]:
        flags = [False] * len(detections)
        for i in range(len(detections)):
            xi = float(detections[i].get("center_px_x", 0.0))
            yi = float(detections[i].get("center_px_y", 0.0))
            box_i = (float(detections[i]["bbox_x1"]), float(detections[i]["bbox_y1"]),
                     float(detections[i]["bbox_x2"]), float(detections[i]["bbox_y2"]))
            for j in range(i + 1, len(detections)):
                xj = float(detections[j].get("center_px_x", 0.0))
                yj = float(detections[j].get("center_px_y", 0.0))
                box_j = (float(detections[j]["bbox_x1"]), float(detections[j]["bbox_y1"]),
                         float(detections[j]["bbox_x2"]), float(detections[j]["bbox_y2"]))
                dx, dy = abs(xi - xj), abs(yi - yj)
                iou = bbox_iou_xyxy(box_i, box_j)
                if (dx <= C.AMBIGUOUS_X_PX and dy <= C.AMBIGUOUS_Y_PX) or iou >= C.AMBIGUOUS_IOU_THRESH:
                    flags[i] = flags[j] = True
        return flags

    def _compute_occlusion_flags(self, detections: List[dict]) -> List[bool]:
        flags = [False] * len(detections)
        for i in range(len(detections)):
            box_i = (float(detections[i]["bbox_x1"]), float(detections[i]["bbox_y1"]),
                     float(detections[i]["bbox_x2"]), float(detections[i]["bbox_y2"]))
            for j in range(i + 1, len(detections)):
                box_j = (float(detections[j]["bbox_x1"]), float(detections[j]["bbox_y1"]),
                         float(detections[j]["bbox_x2"]), float(detections[j]["bbox_y2"]))
                if bbox_iou_xyxy(box_i, box_j) >= C.OCCLUSION_IOU_THRESH:
                    flags[i] = flags[j] = True
        return flags

    def _det_pair_is_close(self, a: dict, b: dict) -> bool:
        dist_m = math.hypot(float(a["x_m"]) - float(b["x_m"]), float(a["y_m"]) - float(b["y_m"]))
        box_a = (float(a["bbox_x1"]), float(a["bbox_y1"]), float(a["bbox_x2"]), float(a["bbox_y2"]))
        box_b = (float(b["bbox_x1"]), float(b["bbox_y1"]), float(b["bbox_x2"]), float(b["bbox_y2"]))
        return dist_m <= C.PAIR_SWAP_MAX_M or bbox_iou_xyxy(box_a, box_b) >= C.PAIR_SWAP_IOU_THRESH

    def _apply_teammate_constraints(self, proposed: Dict[int, int], detections: List[dict]) -> Dict[int, int]:
        if len(proposed) < 4:
            return proposed
        out = dict(proposed)
        for a, b in [(1, 2), (3, 4)]:
            ia, ib = out.get(a), out.get(b)
            if ia is None or ib is None:
                continue
            sep = math.hypot(
                float(detections[ia]["x_m"]) - float(detections[ib]["x_m"]),
                float(detections[ia]["y_m"]) - float(detections[ib]["y_m"]),
            )
            if sep < C.TEAMMATE_MIN_SEP_M:
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
                if tr_a.hits < C.PAIR_SWAP_HIT_MIN or tr_b.hits < C.PAIR_SWAP_HIT_MIN:
                    continue
                if not self._det_pair_is_close(det_a, det_b):
                    continue
                amb = ambiguity_flags[det_a_idx] or ambiguity_flags[det_b_idx]
                occ = occlusion_flags[det_a_idx] or occlusion_flags[det_b_idx]
                row_a = track_ids.index(sid_a)
                row_b = track_ids.index(sid_b)
                chosen = float(cost_matrix[row_a, det_a_idx] + cost_matrix[row_b, det_b_idx])
                alt    = float(cost_matrix[row_a, det_b_idx] + cost_matrix[row_b, det_a_idx])
                margin = C.PAIR_SWAP_COST_MARGIN + (C.PAIR_SWAP_AMBIGUOUS_BONUS if amb else 0.0) + (0.06 if occ else 0.0)
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
        entering_occlusion = occluded and not tr.was_occluded
        exiting_occlusion  = not occluded and tr.was_occluded

        if entering_occlusion:
            tr.pre_occlusion_clip  = tr.clip_feat.copy()  if tr.clip_feat  is not None else None
            tr.pre_occlusion_color = tr.color_feat.copy() if tr.color_feat is not None else None
            tr.pre_occlusion_pose  = tr.pose_feat.copy()  if tr.pose_feat  is not None else None
        if exiting_occlusion:
            tr.post_occlusion_frames = C.POST_OCCLUSION_RECOVERY_FRAMES
        if tr.post_occlusion_frames > 0:
            tr.post_occlusion_frames -= 1

        if occluded and C.OCCLUSION_POSITION_FREEZE:
            pred_x, pred_y = self._predict_xy(tr, frame_idx)
            tr.last_x = float(np.clip(pred_x, 0.0, COURT_W))
            tr.last_y = float(np.clip(pred_y, 0.0, COURT_L))
        else:
            dtf = max(1, frame_idx - tr.last_seen_frame)
            nvx = (float(det["x_m"]) - tr.last_x) / dtf
            nvy = (float(det["y_m"]) - tr.last_y) / dtf
            tr.vx_history.append(nvx)
            tr.vy_history.append(nvy)
            if len(tr.vx_history) > C.VELOCITY_HISTORY_FRAMES:
                tr.vx_history = tr.vx_history[-C.VELOCITY_HISTORY_FRAMES:]
                tr.vy_history = tr.vy_history[-C.VELOCITY_HISTORY_FRAMES:]
            tr.last_x = float(det["x_m"])
            tr.last_y = float(det["y_m"])
        tr.last_seen_frame = int(frame_idx)

        freeze_bank = (occluded and C.OCCLUSION_FREEZE_BANK_UPDATE) or (ambiguous and C.AMBIGUOUS_FREEZE_BANK_UPDATE)
        freeze_ema  = occluded and C.OCCLUSION_FREEZE_EMA

        new_clip = det.get("clip_feat")
        if new_clip is not None and not freeze_ema:
            if tr.clip_feat is None:
                tr.clip_feat = new_clip.astype(np.float32)
            else:
                tr.clip_feat = (C.CLIP_EMA_ALPHA * new_clip + (1.0 - C.CLIP_EMA_ALPHA) * tr.clip_feat).astype(np.float32)
                n = float(np.linalg.norm(tr.clip_feat))
                if n > 1e-8:
                    tr.clip_feat /= n
            if not freeze_bank:
                tr.clip_bank = self._push_bank(tr.clip_bank, new_clip, C.MEMORY_BANK_SIZE)

        new_color = det.get("color_feat")
        if new_color is not None and not freeze_ema:
            if tr.color_feat is None:
                tr.color_feat = new_color.astype(np.float32)
            else:
                tr.color_feat = (C.COLOR_EMA_ALPHA * new_color + (1.0 - C.COLOR_EMA_ALPHA) * tr.color_feat).astype(np.float32)
            if not freeze_bank:
                tr.color_bank = self._push_bank(tr.color_bank, new_color, C.MEMORY_BANK_SIZE)

        new_pose = det.get("pose_feat")
        if new_pose is not None and not freeze_ema:
            if tr.pose_feat is None:
                tr.pose_feat = new_pose.astype(np.float32)
            else:
                tr.pose_feat = (C.POSE_EMA_ALPHA * new_pose + (1.0 - C.POSE_EMA_ALPHA) * tr.pose_feat).astype(np.float32)
                n = float(np.linalg.norm(tr.pose_feat))
                if n > 1e-8:
                    tr.pose_feat /= n
            if not freeze_bank:
                tr.pose_bank = self._push_bank(tr.pose_bank, new_pose, C.POSE_BANK_SIZE)

        if not occluded:
            tr.box_h = C.SIZE_ALPHA * float(det.get("box_h", tr.box_h)) + (1.0 - C.SIZE_ALPHA) * tr.box_h
            tr.box_w = C.SIZE_ALPHA * float(det.get("box_w", tr.box_w)) + (1.0 - C.SIZE_ALPHA) * tr.box_w
            self._update_side_memory(tr, det)
            self._update_role_memory(tr, det)

        raw_tid = det.get("track_id")
        if raw_tid is not None:
            tr.raw_tracker_id = int(raw_tid)
        tr.hits += 1
        tr.was_occluded = occluded

    def assign(self, frame_idx: int, detections: List[dict]) -> None:
        if not detections or not self.is_bootstrapped():
            return
        occlusion_flags = self._compute_occlusion_flags(detections)
        ambiguity_flags = self._compute_ambiguity_flags(detections)
        track_ids = self.stable_ids
        num_tracks = len(track_ids)
        num_dets   = len(detections)
        cost_matrix = np.full((num_tracks, num_dets), C.UNMATCHED_COST, dtype=np.float32)

        for r, sid in enumerate(track_ids):
            tr = self.tracks[sid]
            for c, det in enumerate(detections):
                occluded  = occlusion_flags[c]
                ambiguous = ambiguity_flags[c]
                base_cost = self._total_cost(tr, det, frame_idx, occluded, ambiguous)
                if ambiguous:
                    base_cost += 0.08
                raw_tid = det.get("track_id")
                if tr.raw_tracker_id is not None and raw_tid is not None and int(raw_tid) == int(tr.raw_tracker_id):
                    stickiness = C.OCCLUSION_EXTRA_STICKINESS if occluded else 0.08
                    base_cost = max(0.0, base_cost - stickiness)
                cost_matrix[r, c] = base_cost

        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        proposed: Dict[int, int] = {}
        for r, c in zip(row_ind, col_ind):
            sid = track_ids[r]
            tr  = self.tracks[sid]
            det = detections[c]
            occluded  = occlusion_flags[c]
            ambiguous = ambiguity_flags[c]
            total_cost  = float(cost_matrix[r, c])
            motion_cost = self._motion_cost(tr, det, frame_idx, occluded)
            clip_cost   = self._clip_cost(tr, det, occluded)
            pose_cost   = self._pose_cost(tr, det, ambiguous)
            clip_sim    = 1.0 - clip_cost
            pose_sim    = 1.0 - pose_cost
            motion_dist_ok = motion_cost <= clamp01(C.OUTSIDER_DIST_M / max(1e-6, self.max_match_dist_m))
            strong_identity = clip_sim >= C.OUTSIDER_CLIP_MIN_SIM or pose_sim >= 0.66
            local_max_accept = C.AMBIGUOUS_MAX_ACCEPT_COST if ambiguous else C.MAX_ACCEPT_COST
            if total_cost <= local_max_accept and (motion_dist_ok or strong_identity):
                proposed[sid] = c

        proposed = self._anti_swap_pairs(proposed, detections, cost_matrix, track_ids, occlusion_flags, ambiguity_flags, frame_idx)
        proposed = self._apply_teammate_constraints(proposed, detections)

        for sid, det_idx in proposed.items():
            tr  = self.tracks[sid]
            det = detections[det_idx]
            occluded  = occlusion_flags[det_idx]
            ambiguous = ambiguity_flags[det_idx]
            row_ix   = track_ids.index(sid)
            assigned_cost = float(cost_matrix[row_ix, det_idx])
            row_costs     = cost_matrix[row_ix]
            others        = [float(v) for j, v in enumerate(row_costs) if j != det_idx]
            second_best   = min(others) if others else None
            raw_tid    = det.get("track_id")
            same_raw   = tr.raw_tracker_id is not None and raw_tid is not None and int(raw_tid) == int(tr.raw_tracker_id)
            accept     = True
            local_switch_margin = C.SWITCH_MARGIN + (C.OCCLUSION_EXTRA_STICKINESS if occluded else 0.0) + (C.AMBIGUOUS_EXTRA_SWITCH_MARGIN if ambiguous else 0.0)
            local_confirm = C.SWITCH_CONFIRM_FRAMES + (2 if occluded else 0) + (C.AMBIGUOUS_EXTRA_CONFIRM_FRAMES if ambiguous else 0)
            if not same_raw and second_best is not None:
                improvement = second_best - assigned_cost
                if improvement >= local_switch_margin:
                    candidate = int(raw_tid) if raw_tid is not None else -1
                    if tr.pending_raw_tid == candidate:
                        tr.pending_count += 1
                    else:
                        tr.pending_raw_tid = candidate
                        tr.pending_count   = 1
                    if tr.pending_count < local_confirm:
                        accept = False
                    else:
                        tr.pending_raw_tid = None
                        tr.pending_count   = 0
                else:
                    tr.pending_raw_tid = None
                    tr.pending_count   = 0
            else:
                tr.pending_raw_tid = None
                tr.pending_count   = 0
            if accept:
                det["stable_id"]  = sid
                det["side_group"] = "ALL"
                self._update_track(sid, det, frame_idx, occluded, ambiguous)

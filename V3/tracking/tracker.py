"""
SkeletonTracker — Pass 1 of the V3 pipeline.

Runs YOLO + ByteTrack + IdentityManager + full MediaPipe pose extraction
and writes skeleton.parquet + annotated.mp4 + session_meta.json.
"""

import os
import json
import math
import threading
import queue as _queue
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd
import torch
import open_clip
from PIL import Image
from ultralytics import YOLO
import supervision as sv

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config as C
from shared.calibration import (
    COURT_W, COURT_L, COURT_DST,
    px_to_meters, load_calibration, save_calibration,
    click_corners, click_players,
)
from shared.court import (
    classify_zone, draw_court_guides,
    point_in_court_asymmetric_margin, point_to_polygon_signed_distance,
)
from shared.drawing import draw_player_panel, knee_to_bgr
from shared.io_utils import (
    ensure_dir, now_stamp, expand_bbox, resize_keep_aspect,
    clamp01,
)
from tracking.identity import IdentityManager
from tracking.pose_extractor import PoseExtractor
from tracking.schemas import ALL_SKELETON_COLS, skeleton_schema


# ── CLIP embedder ─────────────────────────────────────────────────────────────
class ClipEmbedder:
    def __init__(self):
        device = C.CLIP_DEVICE
        self.device = device
        self.use_fp16 = (device == "cuda")
        precision = "fp16" if self.use_fp16 else "fp32"
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            C.CLIP_MODEL_NAME, pretrained=C.CLIP_PRETRAINED, precision=precision
        )
        self.model = self.model.to(device).eval()

    def encode_batch(self, crops: List[Optional[np.ndarray]]) -> List[Optional[np.ndarray]]:
        tensors, idxs = [], []
        outputs: List[Optional[np.ndarray]] = [None] * len(crops)
        for i, crop in enumerate(crops):
            if crop is None or crop.size == 0:
                continue
            img = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
            tensors.append(self.preprocess(img))
            idxs.append(i)
        if not tensors:
            return outputs
        batch = torch.stack(tensors).to(self.device)
        if self.use_fp16:
            batch = batch.half()
        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.float16, enabled=self.use_fp16):
            feat = self.model.encode_image(batch).float()
            feat = feat / feat.norm(dim=-1, keepdim=True)
            feat = feat.detach().cpu().numpy().astype(np.float32)
        for i, f in zip(idxs, feat):
            outputs[i] = f
        return outputs


# ── colour histogram helper ───────────────────────────────────────────────────
def _extract_color_feat(frame: np.ndarray, box_xyxy: Tuple) -> Optional[np.ndarray]:
    x1, y1, x2, y2 = box_xyxy
    H_img, W_img = frame.shape[:2]
    bx1, by1, bx2, by2 = expand_bbox(x1, y1, x2, y2, W_img, H_img, frac=0.02)
    crop = frame[by1:by2, bx1:bx2]
    if crop is None or crop.size == 0 or crop.shape[0] < 12 or crop.shape[1] < 8:
        return None
    h, w = crop.shape[:2]
    tx1, tx2 = int(w * 0.22), int(w * 0.78)
    ty1, ty2 = int(h * 0.16), int(h * 0.62)
    torso = crop[ty1:ty2, tx1:tx2]
    if torso.size == 0:
        return None
    hsv = cv2.cvtColor(torso, cv2.COLOR_BGR2HSV)
    _, s, v = cv2.split(hsv)
    mask = ((s > 28) & (v > 35) & (v < 245)).astype(np.uint8) * 255
    hist_hs = cv2.calcHist([hsv], [0, 1], mask, [18, 16], [0, 180, 0, 256])
    hist_v  = cv2.calcHist([hsv], [2],    mask, [16],      [0, 256])
    return np.concatenate([
        cv2.normalize(hist_hs, hist_hs).flatten(),
        cv2.normalize(hist_v,  hist_v).flatten(),
    ]).astype(np.float32)


# ── per-player motion state ───────────────────────────────────────────────────
class _MotionState:
    __slots__ = ("last_t", "last_x", "last_y",
                 "speed_kmh", "speed_mps", "accel_mps2", "prev_speed_mps",
                 "total_dist_m",
                 "knee_ema", "hip_ema",
                 "ready_frames", "total_frames", "last_knee", "last_ready")

    def __init__(self, t, x, y):
        self.last_t = t; self.last_x = x; self.last_y = y
        self.speed_kmh = 0.0; self.speed_mps = 0.0
        self.accel_mps2 = 0.0; self.prev_speed_mps = 0.0
        self.total_dist_m = 0.0
        self.knee_ema = 175.0; self.hip_ema = 0.0
        self.ready_frames = 0; self.total_frames = 0
        self.last_knee = None; self.last_ready = 0


def _update_motion(states: Dict[int, _MotionState], sid: int, t: float, x: float, y: float):
    st = states.get(sid)
    if st is None:
        states[sid] = _MotionState(t, x, y)
        return 0.0, 0.0, 0.0
    dt = t - st.last_t
    if dt <= 1e-6:
        return st.speed_kmh, st.total_dist_m, st.accel_mps2
    dx, dy = x - st.last_x, y - st.last_y
    step = math.sqrt(dx * dx + dy * dy)
    if step <= C.DIST_JUMP_CAP_M:
        st.total_dist_m += step
        sp_inst = min(step / dt * 3.6, C.SPEED_CAP_KMH)
        st.speed_kmh = C.SPEED_SMOOTH_ALPHA * sp_inst + (1.0 - C.SPEED_SMOOTH_ALPHA) * st.speed_kmh
        st.speed_mps = st.speed_kmh / 3.6
        ac = float(np.clip((st.speed_mps - st.prev_speed_mps) / dt, -C.ACCEL_CAP_MPS2, C.ACCEL_CAP_MPS2))
        st.accel_mps2 = C.ACCEL_SMOOTH_ALPHA * ac + (1.0 - C.ACCEL_SMOOTH_ALPHA) * st.accel_mps2
        st.prev_speed_mps = st.speed_mps
    st.last_t = t; st.last_x = x; st.last_y = y
    return st.speed_kmh, st.total_dist_m, st.accel_mps2


def _update_ready(states: Dict[int, _MotionState], sid: int, frame_idx: int,
                  knee_avg, hip_drop, ready_flag: int):
    st = states.get(sid)
    if st is None:
        return
    st.total_frames += 1
    if ready_flag == 1:
        st.ready_frames += 1
    if knee_avg is not None:
        st.knee_ema = C.READY_SMOOTH_ALPHA * knee_avg + (1.0 - C.READY_SMOOTH_ALPHA) * st.knee_ema
        st.last_knee = float(st.knee_ema)
    if hip_drop is not None:
        st.hip_ema = C.READY_HIP_SMOOTH_ALPHA * hip_drop + (1.0 - C.READY_HIP_SMOOTH_ALPHA) * st.hip_ema
    st.last_ready = int(ready_flag)


def _ready_pct(states: Dict[int, _MotionState], sid: int) -> float:
    st = states.get(sid)
    if st is None or st.total_frames <= 0:
        return 0.0
    return 100.0 * st.ready_frames / st.total_frames


# ── threaded frame reader ─────────────────────────────────────────────────────
class FrameReader:
    def __init__(self, cap: cv2.VideoCapture, buffer_size: int = 4):
        self._cap = cap
        self._q   = _queue.Queue(maxsize=buffer_size)
        self._stopped = False
        self._thread  = threading.Thread(target=self._worker, daemon=True)
        self._thread.start()

    def _worker(self):
        while not self._stopped:
            if self._q.full():
                threading.Event().wait(0.001)
                continue
            ok, frame = self._cap.read()
            self._q.put((ok, frame))
            if not ok:
                break

    def read(self):
        return self._q.get()

    def stop(self):
        self._stopped = True
        self._thread.join(timeout=2.0)


# ── main tracker class ────────────────────────────────────────────────────────
class SkeletonTracker:
    """
    Runs YOLO + ByteTrack + IdentityManager + MediaPipe full-skeleton and
    writes skeleton.parquet, annotated.mp4, and session_meta.json.
    """

    def run(self, video_path: str, calib_path: str, output_dir: str,
            player_points: Optional[List[List[int]]] = None) -> str:
        """
        Returns path to skeleton.parquet.
        player_points: list of 4 [x,y] pixel positions for bootstrapping.
                       If None, user is prompted to click.
        """
        ensure_dir(output_dir)

        print("V3 — Loading CLIP...")
        clip_embedder = ClipEmbedder()
        print(f"CLIP on {C.CLIP_DEVICE}")

        # ── calibration ───────────────────────────────────────────────────────
        loaded = load_calibration(calib_path)
        cap0 = cv2.VideoCapture(video_path)
        ok0, frame0 = cap0.read()
        cap0.release()
        if not ok0:
            raise RuntimeError(f"Cannot read first frame: {video_path}")

        if loaded is None:
            pts = click_corners(frame0)
            src = np.array(pts, dtype=np.float32)
            H, _ = cv2.findHomography(src, COURT_DST)
            if H is None:
                raise RuntimeError("Homography failed.")
            save_calibration(calib_path, "v3_calib", pts, H)
        else:
            pts, H = loaded

        court_poly_px = np.array(pts, dtype=np.int32).reshape(-1, 2)

        # ── bootstrap player points ────────────────────────────────────────────
        if player_points is None:
            player_points = click_players(frame0, C.MAX_PLAYERS)

        # ── open video ────────────────────────────────────────────────────────
        cap = cv2.VideoCapture(video_path)
        fps = float(cap.get(cv2.CAP_PROP_FPS)) or C.SAVE_FPS_FALLBACK
        if math.isnan(fps) or fps <= 1:
            fps = C.SAVE_FPS_FALLBACK
        width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))  or frame0.shape[1]
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or frame0.shape[0]
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_reader = FrameReader(cap, buffer_size=4)

        # ── models ────────────────────────────────────────────────────────────
        print("V3 — Loading YOLO...")
        yolo = YOLO(C.YOLO_MODEL_PATH)
        if torch.cuda.is_available():
            yolo.to("cuda")

        byte_tracker = sv.ByteTrack(
            track_activation_threshold=C.BYTETRACK_ACTIVATION_THRESH,
            lost_track_buffer=C.BYTETRACK_LOST_BUFFER,
            minimum_matching_threshold=C.BYTETRACK_MATCHING_THRESH,
            frame_rate=int(fps),
        )

        max_age_frames = int(C.MAX_AGE_SECONDS * fps)
        identity = IdentityManager(C.PLAYER_IDS, max_age_frames, C.MAX_MATCH_DIST_M)
        motion_states: Dict[int, _MotionState] = {}

        print("V3 — Loading MediaPipe pose pool...")
        pose_extractor = PoseExtractor(C.POSE_MODEL_PATH, C.POSE_POOL_SIZE)

        # ── video writer ──────────────────────────────────────────────────────
        video_out = os.path.join(output_dir, "annotated.mp4")
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(video_out, fourcc, fps, (width, height))
        if not writer.isOpened():
            fourcc = cv2.VideoWriter_fourcc(*"avc1")
            writer = cv2.VideoWriter(video_out, fourcc, fps, (width, height))

        # ── output buffer ─────────────────────────────────────────────────────
        all_rows: List[dict] = []
        frame_idx   = -1
        bootstrapped = False

        print(f"V3 — Pass 1 tracking ({total_frames} frames)...")
        try:
            while True:
                ok, frame = frame_reader.read()
                if not ok or frame is None:
                    break
                frame_idx  += 1
                timestamp_s = frame_idx / fps
                annotated   = frame.copy()

                # YOLO
                results    = yolo(frame, verbose=False, conf=C.CONF, iou=C.IOU, imgsz=C.IMGSZ,
                                  half=torch.cuda.is_available())[0]
                detections = sv.Detections.from_ultralytics(results)

                det_rows: List[dict] = []

                if detections.class_id is not None and len(detections) > 0:
                    detections = detections[detections.class_id == 0]

                    # court filter
                    if len(detections) > 0:
                        keep = []
                        for xyxy in detections.xyxy:
                            x1, y1, x2, y2 = xyxy
                            foot = (float((x1 + x2) / 2), float(y2))
                            keep.append(point_in_court_asymmetric_margin(
                                foot, court_poly_px, C.COURT_MARGIN_PX, C.SIDE_BOTTOM_MARGIN_PX))
                        detections = detections[np.array(keep, dtype=bool)]

                    detections = byte_tracker.update_with_detections(detections)

                    if detections.tracker_id is not None and len(detections) > 0:
                        H_img, W_img = frame.shape[:2]
                        do_pose = (frame_idx % C.POSE_EVERY_N_FRAMES == 0)
                        raw_dets = list(zip(detections.xyxy, detections.tracker_id))

                        # submit pose futures
                        pose_futures = []
                        for det_i, (xyxy, _) in enumerate(raw_dets):
                            x1, y1, x2, y2 = xyxy.astype(float)
                            future = None
                            if do_pose:
                                bx1, by1, bx2, by2 = expand_bbox(x1, y1, x2, y2, W_img, H_img,
                                                                   frac=C.POSE_BBOX_EXPAND)
                                pose_crop = frame[by1:by2, bx1:bx2]
                                if pose_crop.size > 0:
                                    small, _ = resize_keep_aspect(pose_crop, C.POSE_CROP_MAX_W, C.POSE_CROP_MAX_H)
                                    crops_with_idx = [(det_i, small)]
                                    futs = pose_extractor.submit_batch(crops_with_idx)
                                    future = futs[0]
                            pose_futures.append(future)

                        # build detection rows
                        crops = []
                        temp_rows: List[dict] = []
                        for det_i, (xyxy, track_id) in enumerate(raw_dets):
                            x1, y1, x2, y2 = xyxy.astype(float)
                            foot_px_x = (x1 + x2) / 2.0
                            foot_px_y = y2
                            center_px_x = float((x1 + x2) / 2.0)
                            center_px_y = float((y1 + y2) / 2.0)
                            x_m, y_m = px_to_meters(H, foot_px_x, foot_px_y)
                            x_m = float(np.clip(x_m, 0.0, COURT_W))
                            y_m = float(np.clip(y_m, 0.0, COURT_L))
                            zone     = classify_zone(x_m, y_m)
                            box_w    = float(max(1.0, x2 - x1))
                            box_h    = float(max(1.0, y2 - y1))
                            foot_dist = point_to_polygon_signed_distance(
                                (float(foot_px_x), float(foot_px_y)), court_poly_px)

                            # expand crop for CLIP
                            bx1c, by1c, bx2c, by2c = expand_bbox(x1, y1, x2, y2, W_img, H_img, frac=0.04)
                            crop = frame[by1c:by2c, bx1c:bx2c]
                            if crop.size == 0 or crop.shape[0] < 12 or crop.shape[1] < 8:
                                crop = None
                            color_feat = _extract_color_feat(frame, (x1, y1, x2, y2))

                            # pose result
                            pose_feat = None
                            knee_L = knee_R = knee_avg = hip_drop = None
                            ready_flag = 0
                            lm_row: dict = {}
                            future = pose_futures[det_i]
                            if future is not None:
                                pose_res = future.result()
                                pose_feat = PoseExtractor.extract_signature(pose_res)
                                knee_L, knee_R, knee_avg, hip_drop, ready_flag = PoseExtractor.extract_ready_state(pose_res)
                                lm_row = PoseExtractor.landmarks_to_row(
                                    pose_res, -1, frame_idx, x1, y1, x2, y2)

                            row = {
                                "frame_idx": frame_idx,
                                "timestamp_s": timestamp_s,
                                "track_id": int(track_id),
                                "bbox_x1": float(x1), "bbox_y1": float(y1),
                                "bbox_x2": float(x2), "bbox_y2": float(y2),
                                "foot_px_x": float(foot_px_x), "foot_px_y": float(foot_px_y),
                                "center_px_x": center_px_x, "center_px_y": center_px_y,
                                "foot_poly_dist": float(foot_dist),
                                "x_m": x_m, "y_m": y_m, "zone": zone,
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
                            }
                            row.update(lm_row)
                            # pose signature for parquet
                            if pose_feat is not None:
                                for k, v in enumerate(pose_feat[:8]):
                                    row[f"pose_sig_{k}"] = float(v)
                            else:
                                for k in range(8):
                                    row[f"pose_sig_{k}"] = float("nan")

                            temp_rows.append(row)
                            crops.append(crop)

                        clip_feats = clip_embedder.encode_batch(crops)
                        for r, feat in zip(temp_rows, clip_feats):
                            r["clip_feat"] = feat
                        det_rows = temp_rows

                        # bootstrap / assign
                        if not bootstrapped:
                            identity.bootstrap(det_rows, player_points)
                            bootstrapped = identity.is_bootstrapped()
                        else:
                            occ_flags = identity._compute_occlusion_flags(det_rows)
                            amb_flags = identity._compute_ambiguity_flags(det_rows)
                            for det in det_rows:
                                det["rescue_flag"] = int(
                                    bool(amb_flags[det_rows.index(det)]
                                         or occ_flags[det_rows.index(det)]))
                            identity.assign(frame_idx, det_rows)

                        det_rows = [r for r in det_rows if int(r.get("stable_id", -1)) in C.PLAYER_IDS]

                        xyxy_by_track = {int(tid): xy for xy, tid in zip(detections.xyxy, detections.tracker_id)}

                        for r in det_rows:
                            sid = int(r["stable_id"])
                            sp_kmh, dist_m, accel = _update_motion(motion_states, sid, timestamp_s, r["x_m"], r["y_m"])
                            r["speed_kmh"]  = float(sp_kmh)
                            r["distance_m"] = float(dist_m)
                            r["accel_mps2"] = float(accel)

                            if r["ready_flag"] == 1 and r["speed_kmh"] > C.READY_MAX_SPEED_KMH:
                                r["ready_flag"] = 0

                            _update_ready(motion_states, sid, frame_idx,
                                          r.get("knee_angle_avg"), r.get("hip_drop"), int(r.get("ready_flag", 0)))
                            st = motion_states.get(sid)
                            if st is not None:
                                if r.get("knee_angle_avg") is None:
                                    r["knee_angle_avg"] = st.last_knee
                                r["ready_pct"] = _ready_pct(motion_states, sid)
                            else:
                                r["ready_pct"] = 0.0

                            # draw
                            xyxy = xyxy_by_track.get(int(r["track_id"]))
                            if xyxy is not None:
                                ring_col = knee_to_bgr(r.get("knee_angle_avg"))
                                draw_player_panel(annotated, xyxy, sid,
                                                  r.get("knee_angle_avg"), r.get("ready_pct"),
                                                  ring_color=ring_col)

                        # strip internal-only columns before saving
                        _INTERNAL = {"clip_feat", "color_feat", "pose_feat", "box_w", "box_h",
                                     "foot_poly_dist", "center_px_x", "center_px_y", "hip_drop"}
                        for r in det_rows:
                            save_row = {k: v for k, v in r.items() if k not in _INTERNAL}
                            all_rows.append(save_row)

                # draw court overlay
                cv2.polylines(annotated, [court_poly_px], isClosed=True, color=(255, 255, 0), thickness=2)
                try:
                    draw_court_guides(annotated, H)
                except Exception:
                    pass
                cv2.putText(annotated, "V3 Skeleton Tracker", (20, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.72, (0, 255, 255), 2, cv2.LINE_AA)
                writer.write(annotated)

                if frame_idx % 300 == 0:
                    print(f"  frame {frame_idx}/{total_frames}  rows={len(all_rows)}")

        finally:
            frame_reader.stop()
            cap.release()
            writer.release()
            pose_extractor.close()

        # ── write parquet ─────────────────────────────────────────────────────
        skeleton_path = os.path.join(output_dir, "skeleton.parquet")
        if all_rows:
            df = pd.DataFrame(all_rows)
            # ensure all expected columns exist (NaN for missing)
            for col in ALL_SKELETON_COLS:
                if col not in df.columns:
                    df[col] = float("nan")
            df = df[[c for c in ALL_SKELETON_COLS if c in df.columns]]
            df.to_parquet(skeleton_path, engine="pyarrow", compression="snappy", index=False)
            print(f"V3 — skeleton.parquet written: {len(df)} rows, {len(df.columns)} cols")
        else:
            pd.DataFrame(columns=ALL_SKELETON_COLS).to_parquet(
                skeleton_path, engine="pyarrow", index=False)

        # ── session meta ──────────────────────────────────────────────────────
        meta = {
            "video_path": video_path,
            "calib_path": calib_path,
            "fps": fps,
            "width": width,
            "height": height,
            "total_frames": frame_idx + 1,
            "player_ids": C.PLAYER_IDS,
            "created_at": datetime.now().isoformat(),
        }
        with open(os.path.join(output_dir, "session_meta.json"), "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)

        print(f"V3 — Pass 1 done. annotated.mp4 → {video_out}")
        return skeleton_path

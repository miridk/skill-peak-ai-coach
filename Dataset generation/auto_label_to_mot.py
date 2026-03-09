import argparse
import json
import math
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from ultralytics import YOLO


# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def iou_xyxy(a: np.ndarray, b: np.ndarray) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    iw = max(0.0, inter_x2 - inter_x1)
    ih = max(0.0, inter_y2 - inter_y1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    denom = area_a + area_b - inter + 1e-6
    return float(inter / denom)


def center_of(box: np.ndarray) -> Tuple[float, float]:
    x1, y1, x2, y2 = box
    return float((x1 + x2) * 0.5), float((y1 + y2) * 0.5)


def bottom_center_of(box: np.ndarray) -> Tuple[float, float]:
    x1, y1, x2, y2 = box
    return float((x1 + x2) * 0.5), float(y2)


def point_in_poly(point: Tuple[float, float], poly: np.ndarray) -> bool:
    x, y = point
    return cv2.pointPolygonTest(poly.astype(np.float32), (float(x), float(y)), False) >= 0


# ------------------------------------------------------------
# Calibration / court polygon parsing
# ------------------------------------------------------------


def _coerce_points(value) -> Optional[np.ndarray]:
    if not isinstance(value, (list, tuple)):
        return None
    pts = []
    for item in value:
        if isinstance(item, dict):
            if "x" in item and "y" in item:
                pts.append([float(item["x"]), float(item["y"])] )
        elif isinstance(item, (list, tuple)) and len(item) >= 2:
            pts.append([float(item[0]), float(item[1])])
    if len(pts) >= 4:
        return np.asarray(pts, dtype=np.float32)
    return None


def load_court_polygon(calib_file: Optional[str]) -> Optional[np.ndarray]:
    if not calib_file:
        return None
    p = Path(calib_file)
    if not p.exists():
        print(f"[WARN] Calibration file not found: {p}")
        return None

    with open(p, "r", encoding="utf-8") as f:
        data = json.load(f)

    candidates = []
    if isinstance(data, dict):
        # Common top-level keys
        for key in [
            "court_points",
            "court_polygon",
            "image_points",
            "points",
            "corners",
            "court_corners",
            "src_points",
        ]:
            if key in data:
                candidates.append(data[key])

        # Nested common structures
        for key in ["court", "calibration", "homography"]:
            if key in data and isinstance(data[key], dict):
                for subkey in [
                    "court_points",
                    "court_polygon",
                    "image_points",
                    "points",
                    "corners",
                    "court_corners",
                    "src_points",
                ]:
                    if subkey in data[key]:
                        candidates.append(data[key][subkey])

    for candidate in candidates:
        pts = _coerce_points(candidate)
        if pts is not None:
            print(f"[INFO] Loaded court polygon from: {p}")
            return pts

    print(f"[WARN] Could not parse court polygon from: {p}")
    return None


# ------------------------------------------------------------
# Simple tracker
# ------------------------------------------------------------


@dataclass
class Track:
    track_id: int
    box: np.ndarray
    last_frame_idx: int
    hits: int = 1
    age: int = 0
    history: List[np.ndarray] = field(default_factory=list)

    def __post_init__(self):
        self.history.append(self.box.copy())

    def update(self, box: np.ndarray, frame_idx: int) -> None:
        self.box = box.copy()
        self.last_frame_idx = frame_idx
        self.hits += 1
        self.age = 0
        self.history.append(self.box.copy())
        if len(self.history) > 8:
            self.history.pop(0)

    def predict_box(self) -> np.ndarray:
        if len(self.history) < 2:
            return self.box.copy()
        prev = self.history[-2]
        curr = self.history[-1]
        vel = curr - prev
        return curr + vel


class SimpleIOUTracker:
    def __init__(self, iou_threshold: float = 0.25, max_age: int = 20, max_tracks: int = 12):
        self.iou_threshold = iou_threshold
        self.max_age = max_age
        self.max_tracks = max_tracks
        self.next_id = 1
        self.tracks: Dict[int, Track] = {}

    def update(self, detections: List[np.ndarray], frame_idx: int) -> List[Tuple[int, np.ndarray]]:
        results: List[Tuple[int, np.ndarray]] = []

        for track in self.tracks.values():
            track.age += 1

        unmatched_det_indices = set(range(len(detections)))
        matched_track_ids = set()

        if self.tracks and detections:
            pairs = []
            for tid, track in self.tracks.items():
                pred = track.predict_box()
                for di, det in enumerate(detections):
                    score = iou_xyxy(pred, det)
                    if score >= self.iou_threshold:
                        pairs.append((score, tid, di))

            pairs.sort(reverse=True, key=lambda x: x[0])
            used_dets = set()
            used_tracks = set()
            for score, tid, di in pairs:
                if tid in used_tracks or di in used_dets:
                    continue
                self.tracks[tid].update(detections[di], frame_idx)
                matched_track_ids.add(tid)
                used_tracks.add(tid)
                used_dets.add(di)
                unmatched_det_indices.discard(di)
                results.append((tid, detections[di]))

        # Create new tracks
        for di in sorted(unmatched_det_indices):
            if len(self.tracks) >= self.max_tracks:
                continue
            det = detections[di]
            tid = self.next_id
            self.next_id += 1
            self.tracks[tid] = Track(track_id=tid, box=det.copy(), last_frame_idx=frame_idx)
            matched_track_ids.add(tid)
            results.append((tid, det))

        # Keep matched tracks in results if not already included (for stable ordering)
        present_ids = {tid for tid, _ in results}
        for tid in matched_track_ids:
            if tid not in present_ids:
                results.append((tid, self.tracks[tid].box.copy()))

        # Remove stale tracks
        stale = [tid for tid, tr in self.tracks.items() if tr.age > self.max_age]
        for tid in stale:
            del self.tracks[tid]

        results.sort(key=lambda x: x[0])
        return results


# ------------------------------------------------------------
# Main pipeline
# ------------------------------------------------------------


def select_person_class(model: YOLO) -> int:
    names = model.names
    if isinstance(names, dict):
        for cls_id, name in names.items():
            if str(name).lower() == "person":
                return int(cls_id)
    raise ValueError("Could not find 'person' class in model names.")


def make_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Auto-label badminton video to MOT/CVAT-friendly dataset.")
    p.add_argument("--video", required=True, help="Input video path")
    p.add_argument("--model", default="yolov8n.pt", help="YOLO model path")
    p.add_argument("--output", default="dataset_out", help="Output folder")
    p.add_argument("--imgsz", type=int, default=1280)
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--iou", type=float, default=0.45, help="YOLO NMS IoU")
    p.add_argument("--device", default=None, help="cuda device or cpu")
    p.add_argument("--save-every-nth-frame", type=int, default=3)
    p.add_argument("--calibration", default=None, help="Optional calibration JSON with court polygon")
    p.add_argument("--keep-outside-court", action="store_true", help="Also keep people outside court")
    p.add_argument("--min-box-area", type=float, default=2500.0)
    p.add_argument("--max-tracks", type=int, default=12)
    p.add_argument("--track-iou-threshold", type=float, default=0.25)
    p.add_argument("--track-max-age", type=int, default=20)
    p.add_argument("--draw-preview", action="store_true", help="Save preview video with boxes and track IDs")
    return p


def main() -> None:
    args = make_parser().parse_args()

    video_path = Path(args.video)
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    out_root = Path(args.output)
    img_dir = out_root / "img1"
    gt_dir = out_root / "gt"
    ensure_dir(out_root)
    ensure_dir(img_dir)
    ensure_dir(gt_dir)

    model = YOLO(args.model)
    person_cls = select_person_class(model)
    court_poly = load_court_polygon(args.calibration)
    tracker = SimpleIOUTracker(
        iou_threshold=args.track_iou_threshold,
        max_age=args.track_max_age,
        max_tracks=args.max_tracks,
    )

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    src_fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    saved_idx = 0
    frame_idx = 0
    mot_lines: List[str] = []

    preview_writer = None
    if args.draw_preview:
        preview_path = str(out_root / "preview.mp4")
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        preview_writer = cv2.VideoWriter(preview_path, fourcc, src_fps / max(1, args.save_every_nth_frame), (width, height))

    print(f"[INFO] Video: {video_path}")
    print(f"[INFO] Source FPS: {src_fps:.2f}, size: {width}x{height}, frames: {total_frames}")
    print(f"[INFO] Saving every {args.save_every_nth_frame} frame(s)")

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        if frame_idx % args.save_every_nth_frame != 0:
            frame_idx += 1
            continue

        results = model.predict(
            source=frame,
            imgsz=args.imgsz,
            conf=args.conf,
            iou=args.iou,
            classes=[person_cls],
            device=args.device,
            verbose=False,
        )[0]

        detections: List[np.ndarray] = []
        det_kinds: List[str] = []

        if results.boxes is not None and len(results.boxes) > 0:
            xyxy = results.boxes.xyxy.detach().cpu().numpy()
            for box in xyxy:
                x1, y1, x2, y2 = box.astype(np.float32)
                area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
                if area < args.min_box_area:
                    continue

                if court_poly is not None:
                    inside = point_in_poly(bottom_center_of(box), court_poly)
                    if not inside and not args.keep_outside_court:
                        continue
                    det_kinds.append("player" if inside else "person_outside_court")
                else:
                    det_kinds.append("player")
                detections.append(np.asarray([x1, y1, x2, y2], dtype=np.float32))

        tracks = tracker.update(detections, saved_idx + 1)

        # Save frame
        saved_idx += 1
        img_name = f"{saved_idx:06d}.jpg"
        cv2.imwrite(str(img_dir / img_name), frame)

        # Save MOT gt lines
        for tid, box in tracks:
            x1, y1, x2, y2 = box
            w = x2 - x1
            h = y2 - y1
            # frame,id,x,y,w,h,conf,class,visibility
            mot_lines.append(f"{saved_idx},{tid},{x1:.2f},{y1:.2f},{w:.2f},{h:.2f},1,1,1")

            if preview_writer is not None:
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                cv2.putText(
                    frame,
                    f"ID {tid}",
                    (int(x1), max(20, int(y1) - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2,
                    cv2.LINE_AA,
                )

        if preview_writer is not None:
            if court_poly is not None:
                cv2.polylines(frame, [court_poly.astype(np.int32)], True, (255, 255, 0), 2)
            preview_writer.write(frame)

        if saved_idx % 100 == 0:
            print(f"[INFO] Saved {saved_idx} frames")

        frame_idx += 1

    cap.release()
    if preview_writer is not None:
        preview_writer.release()

    with open(gt_dir / "gt.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(mot_lines))

    seqinfo = (
        "[Sequence]\n"
        f"name={video_path.stem}\n"
        "imDir=img1\n"
        f"frameRate={max(1, int(round(src_fps / max(1, args.save_every_nth_frame))))}\n"
        f"seqLength={saved_idx}\n"
        f"imWidth={width}\n"
        f"imHeight={height}\n"
        "imExt=.jpg\n"
    )
    with open(out_root / "seqinfo.ini", "w", encoding="utf-8") as f:
        f.write(seqinfo)

    classes_txt = "1\nperson\n"
    with open(out_root / "labels.txt", "w", encoding="utf-8") as f:
        f.write(classes_txt)

    print("\n[OK] Dataset created")
    print(f"[OK] Output folder: {out_root.resolve()}")
    print("[OK] Import this folder into CVAT as a MOT dataset.")
    print("[TIP] If IDs drift too much, lower --save-every-nth-frame or raise detection quality.")


if __name__ == "__main__":
    main()

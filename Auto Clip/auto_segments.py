import argparse
import csv
import time
import os   # ← TILFØJ DENNE

from dataclasses import dataclass
from typing import List, Tuple, Optional

import cv2
import numpy as np

# ----------------------------
# Optional progress bar (tqdm)
# ----------------------------
try:
    from tqdm import tqdm  # type: ignore
    HAVE_TQDM = True
except Exception:
    HAVE_TQDM = False

    class tqdm:  # noqa: N801
        def __init__(self, iterable, total=None, desc="", unit="it", mininterval=0.5):
            self.iterable = iterable
            self.total = total if total is not None else (len(iterable) if hasattr(iterable, "__len__") else None)
            self.desc = desc
            self.unit = unit
            self.mininterval = mininterval
            self._last_print = 0.0
            self._i = 0

        def __iter__(self):
            for x in self.iterable:
                self._i += 1
                now = time.time()
                if now - self._last_print >= self.mininterval:
                    self._last_print = now
                    if self.total:
                        pct = (self._i / self.total) * 100.0
                        print(f"{self.desc}: {pct:5.1f}% ({self._i}/{self.total} {self.unit})", end="\r")
                    else:
                        print(f"{self.desc}: {self._i} {self.unit}", end="\r")
                yield x
            print("")


@dataclass
class Seg:
    start_s: float
    end_s: float
    name: str


def sec_to_ts(t: float) -> str:
    t = max(0.0, float(t))
    h = int(t // 3600)
    t -= 3600 * h
    m = int(t // 60)
    s = t - 60 * m
    if h > 0:
        return f"{h:02d}:{m:02d}:{s:06.3f}"
    return f"{m:02d}:{s:06.3f}"


def smooth_ema(x: np.ndarray, alpha: float) -> np.ndarray:
    y = np.zeros_like(x, dtype=np.float32)
    if len(x) == 0:
        return y
    y[0] = x[0]
    for i in range(1, len(x)):
        y[i] = alpha * x[i] + (1 - alpha) * y[i - 1]
    return y


def segments_from_mask(times: np.ndarray, mask: np.ndarray, min_len_s: float) -> List[Tuple[float, float]]:
    segs = []
    start: Optional[float] = None
    for t, m in zip(times, mask):
        if m and start is None:
            start = float(t)
        if (not m) and start is not None:
            end = float(t)
            if end - start >= min_len_s:
                segs.append((start, end))
            start = None
    if start is not None:
        end = float(times[-1])
        if end - start >= min_len_s:
            segs.append((start, end))
    return segs


def motion_score(prev_gray: np.ndarray, gray: np.ndarray) -> float:
    """
    Valgfri: “der sker noget” score (0..1).
    Bruges kun hvis --use-motion er slået til.
    """
    h, w = gray.shape[:2]
    target_w = 640
    if w > target_w:
        scale = target_w / w
        gray_s = cv2.resize(gray, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
        prev_s = cv2.resize(prev_gray, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    else:
        gray_s, prev_s = gray, prev_gray

    hh, ww = gray_s.shape[:2]
    x0, x1 = int(ww * 0.15), int(ww * 0.85)
    y0, y1 = int(hh * 0.12), int(hh * 0.92)
    roi = gray_s[y0:y1, x0:x1]
    prev_roi = prev_s[y0:y1, x0:x1]

    diff = cv2.absdiff(roi, prev_roi)
    diff = cv2.GaussianBlur(diff, (5, 5), 0)
    _, th = cv2.threshold(diff, 18, 255, cv2.THRESH_BINARY)
    th = cv2.morphologyEx(th, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8), iterations=1)

    ratio = float(np.count_nonzero(th)) / float(th.size)

    # tuned map
    score = (ratio - 0.006) / (0.030 - 0.006)
    return float(np.clip(score, 0.0, 1.0))


class AngleMatcher:
    """
    Matcher frames mod et reference-billede (ref.png) vha. ORB feature matching.
    Returnerer en score 0..1 der er høj når det er “samme kameravinkel”.
    """
    def __init__(self, ref_bgr: np.ndarray, roi: Tuple[float, float, float, float] = (0.05, 0.05, 0.95, 0.95)):
        self.roi = roi
        self.orb = cv2.ORB_create(nfeatures=1200, fastThreshold=12)
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

        ref = self._prep(ref_bgr)
        self.ref_kp, self.ref_des = self.orb.detectAndCompute(ref, None)
        if self.ref_des is None or len(self.ref_kp) < 40:
            raise SystemExit("Reference-billedet gav for få features. Brug et skarpere ref.png.")

    def _prep(self, bgr: np.ndarray) -> np.ndarray:
        # downscale til stabil hastighed
        h, w = bgr.shape[:2]
        target_w = 960
        if w > target_w:
            scale = target_w / w
            bgr = cv2.resize(bgr, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)

        h, w = bgr.shape[:2]
        x0 = int(w * self.roi[0]); y0 = int(h * self.roi[1])
        x1 = int(w * self.roi[2]); y1 = int(h * self.roi[3])
        crop = bgr[y0:y1, x0:x1]

        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (3, 3), 0)
        return gray

    def score(self, frame_bgr: np.ndarray) -> float:
        img = self._prep(frame_bgr)
        kp, des = self.orb.detectAndCompute(img, None)
        if des is None or kp is None or len(kp) < 30:
            return 0.0

        matches = self.bf.knnMatch(des, self.ref_des, k=2)
        good = []
        for m, n in matches:
            if m.distance < 0.75 * n.distance:
                good.append(m)

        if len(good) < 18:
            return 0.0

        # Homography inliers = stærkt tegn på samme vinkel/scene
        src_pts = np.float32([kp[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst_pts = np.float32([self.ref_kp[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
        H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        if mask is None:
            return 0.0
        inliers = int(mask.sum())
        inlier_ratio = inliers / max(1, len(good))

        # score kombinerer “hvor mange gode matches” + “hvor geometrisk konsistent”
        match_strength = min(1.0, len(good) / 80.0)
        score = 0.55 * inlier_ratio + 0.45 * match_strength
        return float(np.clip(score, 0.0, 1.0))


def main():
    ap = argparse.ArgumentParser(description="Auto-detect segments based on a specific camera angle (reference image).")
    ap.add_argument("--input", default="input.mp4")
    ap.add_argument("--ref", default="ref.png", help="Reference image for desired camera angle (e.g., screenshot).")
    ap.add_argument("--out_csv", default="segments.csv")

    ap.add_argument("--step", type=float, default=0.6, help="sek mellem samples (0.4-1.0 typisk)")
    ap.add_argument("--angle_thr", type=float, default=0.55, help="threshold for 'rigtig kameravinkel' (0..1)")
    ap.add_argument("--min_len", type=float, default=40.0, help="min segment længde i sek")
    ap.add_argument("--pad_pre", type=float, default=2.0)
    ap.add_argument("--pad_post", type=float, default=3.0)
    ap.add_argument("--ema", type=float, default=0.35, help="EMA smoothing (0.25-0.5)")

    # optional: kræv også “aktivitet”
    ap.add_argument("--use-motion", action="store_true", help="Kræv også motion for at tælle som segment")
    ap.add_argument("--play_thr", type=float, default=0.30, help="threshold for motion (kun hvis --use-motion)")
    args = ap.parse_args()

    if not os.path.exists(args.ref):
        raise SystemExit(f"Ref image not found: {args.ref}")

    print("\n==============================")
    print("AUTO CUT BY CAMERA ANGLE")
    print("==============================")
    print(f"Input:   {args.input}")
    print(f"Ref:     {args.ref}")
    print(f"Output:  {args.out_csv}")
    print(f"Step:    {args.step}s | angle_thr={args.angle_thr} | min_len={args.min_len}s")
    if args.use_motion:
        print(f"Motion:  ON  (play_thr={args.play_thr})")
    else:
        print("Motion:  OFF (angle only)")
    if not HAVE_TQDM:
        print("(Tip) Installér progress bar: python -m pip install tqdm")
    print("")

    ref_bgr = cv2.imread(args.ref, cv2.IMREAD_COLOR)
    if ref_bgr is None:
        raise SystemExit("Kunne ikke læse ref.png. Er filen et rigtigt billede?")

    matcher = AngleMatcher(ref_bgr)

    cap = cv2.VideoCapture(args.input)
    if not cap.isOpened():
        raise SystemExit(f"Kunne ikke åbne video: {args.input}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if total_frames <= 0:
        cap.release()
        raise SystemExit("Kunne ikke læse framecount/varighed fra videoen.")

    duration = total_frames / fps
    print(f"FPS: {fps:.3f} | Frames: {total_frames} | Duration: {duration/60:.1f} min\n")

    times = np.arange(0, duration, args.step, dtype=np.float32)
    angle_scores = np.zeros_like(times, dtype=np.float32)
    play_scores = np.zeros_like(times, dtype=np.float32)

    print("Phase 1/3: Scanning for desired camera angle…")
    prev_gray = None

    it = tqdm(times, total=len(times), desc="Analyzing", unit="sample")
    for i, t in enumerate(it):
        cap.set(cv2.CAP_PROP_POS_MSEC, float(t) * 1000.0)
        ok, frame = cap.read()
        if not ok:
            break

        angle_scores[i] = matcher.score(frame)

        if args.use_motion:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            play_scores[i] = motion_score(prev_gray, gray) if prev_gray is not None else 0.0
            prev_gray = gray
        else:
            play_scores[i] = 1.0  # ignore

        if HAVE_TQDM and (i % 25 == 0):
            if args.use_motion:
                it.set_postfix(angle=f"{angle_scores[i]:.2f}", motion=f"{play_scores[i]:.2f}")
            else:
                it.set_postfix(angle=f"{angle_scores[i]:.2f}")

    cap.release()

    print("\nPhase 2/3: Smoothing + building segments…")
    angle_s = smooth_ema(angle_scores, args.ema)
    play_s = smooth_ema(play_scores, args.ema)

    angle_ok = angle_s >= args.angle_thr
    play_ok = play_s >= args.play_thr if args.use_motion else np.ones_like(angle_ok, dtype=bool)
    good = angle_ok & play_ok

    raw = segments_from_mask(times, good, min_len_s=args.min_len)

    # pad + merge
    padded = []
    for a, b in raw:
        a2 = max(0.0, a - args.pad_pre)
        b2 = min(float(duration), b + args.pad_post)
        padded.append((a2, b2))

    padded.sort()
    merged: List[List[float]] = []
    for a, b in padded:
        if not merged:
            merged.append([a, b])
        else:
            if a <= merged[-1][1] + 0.5:
                merged[-1][1] = max(merged[-1][1], b)
            else:
                merged.append([a, b])

    print("Phase 3/3: Writing segments.csv…")
    with open(args.out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["start", "end", "name"])
        for idx, (a, b) in enumerate(merged, start=1):
            w.writerow([sec_to_ts(a), sec_to_ts(b), f"auto_cam_{idx:02d}"])

    print("\n✅ segments.csv genereret:", args.out_csv)
    print("Segmenter:", len(merged))
    for i, (a, b) in enumerate(merged, start=1):
        print(f"  {i:02d}: {sec_to_ts(a)} -> {sec_to_ts(b)}  ({b - a:.1f}s)")
    print("")


if __name__ == "__main__":
    main()
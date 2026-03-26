"""
V3 Pipeline Orchestrator
Usage:
    python run_pipeline.py --video path/to/video.mp4
    python run_pipeline.py --video path/to/video.mp4 --calib path/to/calib.json
    python run_pipeline.py --video path/to/video.mp4 --serve
    python run_pipeline.py --session <session_id> --serve   # re-serve an existing session
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd

# ── path setup ────────────────────────────────────────────────────────────────
ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

import config as C
from shared.io_utils import ensure_dir, now_stamp, make_output_paths
from shared.calibration import load_calibration, click_corners, select_video


def _find_latest_calib() -> str | None:
    """Return the most recently modified JSON in the calibration/ directory."""
    calib_dir = os.path.join(os.path.dirname(ROOT), "calibration")
    if not os.path.isdir(calib_dir):
        return None
    jsons = [
        os.path.join(calib_dir, f)
        for f in os.listdir(calib_dir)
        if f.endswith(".json")
    ]
    if not jsons:
        return None
    return max(jsons, key=os.path.getmtime)


def run_pipeline(
    video_path: str,
    calib_path: str | None = None,
    serve: bool = False,
    skip_pass1: bool = False,
    skip_pass2: bool = False,
    existing_session_dir: str | None = None,
) -> str:
    """
    Full V3 pipeline. Returns path to the outputs directory.
    If existing_session_dir is given, skips video passes and re-runs analysis.
    """

    # ── resolve calibration ──────────────────────────────────────────────────
    if calib_path is None:
        calib_path = _find_latest_calib()
    if calib_path and os.path.isfile(calib_path):
        _, H = load_calibration(calib_path)
        print(f"V3 — calibration loaded: {calib_path}")
    else:
        print("V3 — no calibration found, user must click court corners.")
        import cv2
        from shared.calibration import COURT_DST
        cap = cv2.VideoCapture(video_path)
        ok, frame = cap.read()
        cap.release()
        if not ok:
            raise RuntimeError(f"Cannot read first frame from {video_path}")
        corners = click_corners(frame)
        src = np.array(corners, dtype=np.float32)
        dst = np.array(COURT_DST, dtype=np.float32)
        H, _ = cv2.findHomography(src, dst)


    # ── output paths ──────────────────────────────────────────────────────────
    if existing_session_dir:
        out_dir = existing_session_dir
        session_id = os.path.basename(out_dir)
    else:
        session_id = now_stamp()
        outputs_root = os.path.join(ROOT, "outputs")
        out_dir = os.path.join(outputs_root, session_id)
    ensure_dir(out_dir)

    paths = make_output_paths(out_dir, session_id)

    # ── Pass 1 — skeleton extraction ─────────────────────────────────────────
    skeleton_path = paths["skeleton"]
    meta_path     = paths["meta"]

    if existing_session_dir and os.path.isfile(skeleton_path):
        print(f"V3 — skeleton.parquet found, skipping Pass 1.")
        with open(meta_path) as f:
            session_meta = json.load(f)
    elif not skip_pass1:
        print("V3 ─── Pass 1: skeleton extraction ───────────────────────────")
        from tracking.tracker import SkeletonTracker
        tracker = SkeletonTracker()
        session_meta = tracker.run(
            video_path=video_path,
            calib_path=calib_path,
            output_dir=out_dir,
        )
    else:
        raise RuntimeError("skip_pass1 set but no existing skeleton.parquet found.")

    fps = float(session_meta.get("fps", C.SAVE_FPS_FALLBACK))

    # ── Pass 2 — shuttle detection ────────────────────────────────────────────
    shuttle_path = paths["shuttle"]
    if existing_session_dir and os.path.isfile(shuttle_path):
        print("V3 — shuttle.parquet found, skipping Pass 2.")
    elif not skip_pass2:
        print("V3 ─── Pass 2: shuttle detection ─────────────────────────────")
        from shuttle.detector import ShuttleDetector
        ShuttleDetector().run(
            video_path=video_path,
            skeleton_path=skeleton_path,
            H=H,
            output_dir=out_dir,
        )
    else:
        shuttle_path = None

    # ── Analysis stage 1 — court positioning ──────────────────────────────────
    print("V3 ─── Analysis: court positioning ───────────────────────────────")
    from analysis.court_positioning import CourtPositioningAnalyzer
    positioning_path = CourtPositioningAnalyzer().run(
        skeleton_path=skeleton_path,
        output_dir=out_dir,
    )

    # ── Analysis stage 2 — reaction times ────────────────────────────────────
    reaction_path = None
    if shuttle_path and os.path.isfile(shuttle_path):
        print("V3 ─── Analysis: reaction times ──────────────────────────────")
        from analysis.reaction_time import ReactionAnalyzer
        reaction_path = ReactionAnalyzer().run(
            shuttle_path=shuttle_path,
            skeleton_path=skeleton_path,
            fps=fps,
            output_dir=out_dir,
        )

    # ── Analysis stage 3 — shot quality ──────────────────────────────────────
    shots_path = None
    if shuttle_path and os.path.isfile(shuttle_path):
        print("V3 ─── Analysis: shot quality ────────────────────────────────")
        from analysis.shot_quality import ShotQualityAnalyzer
        shots_path = ShotQualityAnalyzer().run(
            shuttle_path=shuttle_path,
            skeleton_path=skeleton_path,
            output_dir=out_dir,
        )

    # ── Metrics + Feedback ────────────────────────────────────────────────────
    print("V3 ─── Metrics + feedback report ─────────────────────────────────")
    skeleton_df    = pd.read_parquet(skeleton_path)
    shuttle_df     = pd.read_parquet(shuttle_path)     if shuttle_path and os.path.isfile(shuttle_path) else None
    positioning_df = pd.read_parquet(positioning_path) if os.path.isfile(positioning_path) else None
    reaction_df    = pd.read_parquet(reaction_path)    if reaction_path and os.path.isfile(reaction_path) else None
    shots_df       = pd.read_parquet(shots_path)       if shots_path and os.path.isfile(shots_path) else None

    from report.metrics_v3 import V3MetricsEngine
    engine = V3MetricsEngine()
    summaries = {}
    for pid in C.PLAYER_IDS:
        summaries[pid] = engine.compute_player_summary(
            pid=pid,
            skeleton_df=skeleton_df,
            shuttle_df=shuttle_df,
            positioning_df=positioning_df,
            reaction_df=reaction_df,
            shots_df=shots_df,
        )

    from report.feedback import CoachFeedbackEngine
    insights = CoachFeedbackEngine().generate(summaries)

    from report.html_builder import build_report
    report_path = paths["report"]
    build_report(
        summaries=summaries,
        insights=insights,
        session_meta=session_meta,
        output_path=report_path,
        skeleton_df=skeleton_df,
        shuttle_df=shuttle_df,
    )

    # ── events JSON (for webapp) ──────────────────────────────────────────────
    _write_events_json(insights, reaction_df, shots_df, shuttle_df, paths["events"])

    print(f"\nV3 ─── Pipeline complete ─────────────────────────────────────────")
    print(f"  Session ID : {session_id}")
    print(f"  Output dir : {out_dir}")
    print(f"  Report     : {report_path}")

    if serve:
        _serve(out_dir=os.path.join(ROOT, "outputs"))

    return out_dir


# ── events JSON helper ─────────────────────────────────────────────────────────
def _write_events_json(
    insights,
    reaction_df,
    shots_df,
    shuttle_df,
    out_path: str,
):
    events = []

    # coaching insight events (no timestamp — they show up in the sidebar)
    for ins in insights:
        events.append({
            "type":      "insight",
            "player_id": ins.player_id,
            "category":  ins.category,
            "severity":  ins.severity,
            "text":      ins.text,
        })

    # reaction events with timestamps
    if reaction_df is not None and not reaction_df.empty:
        for _, row in reaction_df.iterrows():
            events.append({
                "type":       "shuttle_event",
                "timestamp_s": float(row["event_timestamp_s"]),
                "subtype":    str(row.get("shuttle_event_type", "event")),
                "shuttle_x_m": float(row.get("shuttle_x_m", float("nan"))),
                "shuttle_y_m": float(row.get("shuttle_y_m", float("nan"))),
            })

    # shot contact events
    if shots_df is not None and not shots_df.empty:
        for _, row in shots_df.iterrows():
            events.append({
                "type":       "shot",
                "timestamp_s": float(row.get("contact_timestamp_s", float("nan"))),
                "player_id":  int(row.get("player_id", -1)),
                "shot_type":  str(row.get("shot_type", "unknown")),
                "shuttle_x_m": float(row.get("shuttle_x_m", float("nan"))),
                "shuttle_y_m": float(row.get("shuttle_y_m", float("nan"))),
            })

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(events, f, allow_nan=False, default=str)
    print(f"V3 — events_v3.json written: {len(events)} events")


# ── Flask serving ─────────────────────────────────────────────────────────────
def _serve(out_dir: str):
    try:
        from webapp.app import create_app
    except ImportError as e:
        print(f"[warn] Flask webapp not available: {e}")
        return
    app = create_app(outputs_root=out_dir)
    port = C.WEBAPP_PORT
    print(f"\nV3 — serving on http://localhost:{port}  (Ctrl-C to quit)")
    try:
        from waitress import serve as waitress_serve
        waitress_serve(app, host="0.0.0.0", port=port)
    except ImportError:
        app.run(host="0.0.0.0", port=port, debug=False)


# ── CLI ───────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="SkillPeak V3 pipeline")
    parser.add_argument("--video",   help="Path to input video file")
    parser.add_argument("--calib",   help="Path to calibration JSON (optional)")
    parser.add_argument("--session", help="Existing session ID to re-analyse/serve")
    parser.add_argument("--serve",   action="store_true",
                        help="Launch Flask webapp after pipeline completes")
    parser.add_argument("--skip-pass1", action="store_true",
                        help="Skip skeleton extraction (skeleton.parquet must exist)")
    parser.add_argument("--skip-pass2", action="store_true",
                        help="Skip shuttle detection (shuttle.parquet must exist)")
    args = parser.parse_args()

    if args.session:
        session_dir = os.path.join(ROOT, "outputs", args.session)
        if not os.path.isdir(session_dir):
            print(f"ERROR: session dir not found: {session_dir}")
            sys.exit(1)
        video_path = None
        # Try to recover video path from session meta
        meta_path = os.path.join(session_dir, "session_meta.json")
        if os.path.isfile(meta_path):
            with open(meta_path) as f:
                meta = json.load(f)
            video_path = meta.get("video_path")
        if args.serve and not video_path:
            # Just serve without re-running pipeline
            _serve(out_dir=os.path.join(ROOT, "outputs"))
            return
        run_pipeline(
            video_path=video_path or "",
            calib_path=args.calib,
            serve=args.serve,
            skip_pass1=True,
            skip_pass2=True,
            existing_session_dir=session_dir,
        )
    elif args.video:
        if not os.path.isfile(args.video):
            print(f"ERROR: video not found: {args.video}")
            sys.exit(1)
        run_pipeline(
            video_path=args.video,
            calib_path=args.calib,
            serve=args.serve,
            skip_pass1=args.skip_pass1,
            skip_pass2=args.skip_pass2,
        )
    else:
        # Interactive: open file dialog
        video_path = select_video()
        if not video_path:
            print("No video selected. Exiting.")
            sys.exit(0)
        run_pipeline(
            video_path=video_path,
            calib_path=args.calib,
            serve=args.serve,
        )


if __name__ == "__main__":
    main()

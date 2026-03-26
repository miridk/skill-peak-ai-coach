"""
Flask routes — all API endpoints and page routes for the V3 webapp.
"""

from __future__ import annotations

import json
import math
import os
from typing import Optional

import numpy as np
import pandas as pd
from flask import (
    Blueprint, abort, current_app, jsonify, render_template,
    request, send_file, Response,
)

bp = Blueprint("main", __name__)


# ── helpers ────────────────────────────────────────────────────────────────────

def _outputs_root() -> str:
    return current_app.config["OUTPUTS_ROOT"]


def _session_dir(session_id: str) -> str:
    path = os.path.join(_outputs_root(), session_id)
    if not os.path.isdir(path):
        abort(404, f"Session '{session_id}' not found.")
    return path


def _read_parquet(session_id: str, name: str) -> Optional[pd.DataFrame]:
    path = os.path.join(_session_dir(session_id), f"{name}.parquet")
    if not os.path.isfile(path):
        return None
    return pd.read_parquet(path)


def _df_to_json(df: pd.DataFrame) -> Response:
    """Serialise DataFrame to JSON, safely handling NaN/inf."""
    records = df.replace([float("inf"), float("-inf")], None).where(df.notna(), other=None).to_dict(orient="records")
    return jsonify(records)


def _list_sessions() -> list[dict]:
    root = _outputs_root()
    if not os.path.isdir(root):
        return []
    sessions = []
    for name in sorted(os.listdir(root), reverse=True):
        path = os.path.join(root, name)
        if not os.path.isdir(path):
            continue
        meta_path = os.path.join(path, "session_meta.json")
        meta = {}
        if os.path.isfile(meta_path):
            with open(meta_path) as f:
                meta = json.load(f)
        sessions.append({
            "session_id":  name,
            "video_name":  meta.get("video_name", name),
            "duration_min": round(meta.get("total_frames", 0) / max(meta.get("fps", 30), 1) / 60, 1),
            "has_report":  os.path.isfile(os.path.join(path, "report.html")),
            "has_video":   os.path.isfile(os.path.join(path, "annotated.mp4")),
        })
    return sessions


# ── page routes ────────────────────────────────────────────────────────────────

@bp.route("/")
def index():
    sessions = _list_sessions()
    return render_template("index.html", sessions=sessions)


@bp.route("/session/<session_id>")
def session_view(session_id: str):
    _session_dir(session_id)   # 404 if missing
    meta_path = os.path.join(_session_dir(session_id), "session_meta.json")
    meta = {}
    if os.path.isfile(meta_path):
        with open(meta_path) as f:
            meta = json.load(f)
    return render_template("session.html", session_id=session_id, meta=meta)


@bp.route("/session/<session_id>/report")
def session_report(session_id: str):
    path = os.path.join(_session_dir(session_id), "report.html")
    if not os.path.isfile(path):
        abort(404, "report.html not found for this session.")
    return send_file(path, mimetype="text/html")


@bp.route("/session/<session_id>/player/<int:pid>")
def player_view(session_id: str, pid: int):
    _session_dir(session_id)
    return render_template("player.html", session_id=session_id, pid=pid)


# ── media ──────────────────────────────────────────────────────────────────────

@bp.route("/media/<session_id>/annotated.mp4")
def serve_video(session_id: str):
    path = os.path.join(_session_dir(session_id), "annotated.mp4")
    if not os.path.isfile(path):
        abort(404, "annotated.mp4 not found.")

    file_size = os.path.getsize(path)
    range_header = request.headers.get("Range")

    if range_header:
        # HTTP range support for video seeking
        ranges = range_header.replace("bytes=", "").split("-")
        start = int(ranges[0])
        end   = int(ranges[1]) if ranges[1] else file_size - 1
        length = end - start + 1

        with open(path, "rb") as f:
            f.seek(start)
            data = f.read(length)

        rv = Response(
            data, 206,
            headers={
                "Content-Range":  f"bytes {start}-{end}/{file_size}",
                "Accept-Ranges":  "bytes",
                "Content-Length": str(length),
                "Content-Type":   "video/mp4",
            },
        )
        return rv
    return send_file(path, mimetype="video/mp4")


# ── API endpoints ──────────────────────────────────────────────────────────────

@bp.route("/api/sessions")
def api_sessions():
    return jsonify(_list_sessions())


@bp.route("/api/session/<session_id>/meta")
def api_meta(session_id: str):
    meta_path = os.path.join(_session_dir(session_id), "session_meta.json")
    if not os.path.isfile(meta_path):
        abort(404, "session_meta.json not found.")
    with open(meta_path) as f:
        return jsonify(json.load(f))


@bp.route("/api/session/<session_id>/events")
def api_events(session_id: str):
    events_path = os.path.join(_session_dir(session_id), "events_v3.json")
    if not os.path.isfile(events_path):
        return jsonify([])
    with open(events_path) as f:
        return jsonify(json.load(f))


@bp.route("/api/session/<session_id>/shuttle")
def api_shuttle(session_id: str):
    df = _read_parquet(session_id, "shuttle")
    if df is None:
        return jsonify([])
    # Downsample to ~2000 rows for API response size
    step = max(1, len(df) // 2000)
    return _df_to_json(df.iloc[::step].reset_index(drop=True))


@bp.route("/api/session/<session_id>/skeleton/<int:pid>")
def api_skeleton_player(session_id: str, pid: int):
    df = _read_parquet(session_id, "skeleton")
    if df is None:
        return jsonify([])
    pdf = df[df["stable_id"] == pid].copy()
    # Return only lightweight columns for the frontend minimap
    cols = ["frame_idx", "timestamp_s", "x_m", "y_m", "speed_kmh", "zone"]
    avail = [c for c in cols if c in pdf.columns]
    step = max(1, len(pdf) // 3000)
    return _df_to_json(pdf[avail].iloc[::step].reset_index(drop=True))


@bp.route("/api/session/<session_id>/reaction")
def api_reaction(session_id: str):
    df = _read_parquet(session_id, "reaction")
    if df is None:
        return jsonify([])
    return _df_to_json(df)


@bp.route("/api/session/<session_id>/shots")
def api_shots(session_id: str):
    df = _read_parquet(session_id, "shots")
    if df is None:
        return jsonify([])
    return _df_to_json(df)


@bp.route("/api/session/<session_id>/positioning")
def api_positioning(session_id: str):
    df = _read_parquet(session_id, "positioning")
    if df is None:
        return jsonify([])
    step = max(1, len(df) // 2000)
    return _df_to_json(df.iloc[::step].reset_index(drop=True))


@bp.route("/api/session/<session_id>/comparison")
def api_comparison(session_id: str):
    """Returns per-player summary metrics as a comparison dict."""
    skel = _read_parquet(session_id, "skeleton")
    shuttle   = _read_parquet(session_id, "shuttle")
    pos       = _read_parquet(session_id, "positioning")
    reaction  = _read_parquet(session_id, "reaction")
    shots     = _read_parquet(session_id, "shots")

    if skel is None:
        abort(503, "skeleton.parquet not ready yet.")

    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    import config as C
    from report.metrics_v3 import V3MetricsEngine
    from report.feedback import CoachFeedbackEngine

    engine = V3MetricsEngine()
    summaries = {}
    for pid in C.PLAYER_IDS:
        s = engine.compute_player_summary(
            pid=pid,
            skeleton_df=skel,
            shuttle_df=shuttle,
            positioning_df=pos,
            reaction_df=reaction,
            shots_df=shots,
        )
        summaries[pid] = s.__dict__

    insights = CoachFeedbackEngine().generate(
        {pid: engine.compute_player_summary(
            pid=pid, skeleton_df=skel, shuttle_df=shuttle,
            positioning_df=pos, reaction_df=reaction, shots_df=shots,
        ) for pid in C.PLAYER_IDS}
    )

    return jsonify({
        "summaries": summaries,
        "insights": [
            {"player_id": i.player_id, "category": i.category,
             "severity": i.severity, "text": i.text}
            for i in insights
        ],
    })


@bp.route("/api/session/<session_id>/heatmap/<int:pid>")
def api_heatmap(session_id: str, pid: int):
    """Return PNG heatmap for one player (generated on demand)."""
    import io
    import importlib.util

    df = _read_parquet(session_id, "skeleton")
    if df is None:
        abort(404, "No skeleton data.")

    pdf = df[df["stable_id"] == pid].dropna(subset=["x_m", "y_m"])
    if pdf.empty:
        abort(404, "No position data for this player.")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.colors as mcolors
    except ImportError:
        abort(503, "matplotlib not available.")

    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    import config as C

    court_w = C.COURT_W
    court_l = C.COURT_L

    fig, ax = plt.subplots(figsize=(4, 8), facecolor="#121212")
    ax.set_facecolor("#1e1e1e")

    # Court outline
    rect = plt.Rectangle((0, 0), court_w, court_l,
                          fill=False, edgecolor="#555", linewidth=1.5)
    ax.add_patch(rect)
    ax.axhline(court_l / 2.0, color="#555", linewidth=1)

    # Heatmap
    ax.hist2d(
        pdf["x_m"].clip(0, court_w),
        pdf["y_m"].clip(0, court_l),
        bins=[30, 60],
        range=[[0, court_w], [0, court_l]],
        cmap="hot",
        density=True,
    )

    ax.set_xlim(0, court_w)
    ax.set_ylim(0, court_l)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(f"Player {pid}", color="#e0e0e0", fontsize=11)

    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight",
                facecolor=fig.get_facecolor(), dpi=120)
    plt.close(fig)
    buf.seek(0)
    return send_file(buf, mimetype="image/png")


@bp.route("/api/run", methods=["POST"])
def api_run():
    """Trigger a new pipeline run. POST body: {video_path, calib_path (opt)}."""
    data = request.get_json(silent=True) or {}
    video_path = data.get("video_path")
    calib_path = data.get("calib_path")

    if not video_path or not os.path.isfile(video_path):
        return jsonify({"error": "video_path missing or not found"}), 400

    import threading
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    from run_pipeline import run_pipeline

    def _run():
        try:
            run_pipeline(video_path=video_path, calib_path=calib_path)
        except Exception as e:
            print(f"[api_run] pipeline error: {e}")

    t = threading.Thread(target=_run, daemon=True)
    t.start()
    return jsonify({"status": "started", "video": video_path}), 202

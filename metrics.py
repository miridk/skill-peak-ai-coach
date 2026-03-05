import os
import glob
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import matplotlib.patches as patches


# ----------------------------
# CONFIG (match your tracker)
# ----------------------------
COURT_W = 6.10
COURT_L = 13.40

ONLY_PRIMARY_SINGLE = False  # set True if you want
EMA_ALPHA = 0.35

HEATMAP_BINS_X = 60
HEATMAP_BINS_Y = 120

OUT_DIR = "metrics_out"
HEATMAP_DIR = os.path.join(OUT_DIR, "heatmaps")
PLOTS_DIR = os.path.join(OUT_DIR, "plots")
SERIES_DIR = os.path.join(OUT_DIR, "series")

# ✅ ID selection
ID_COL_DEFAULT = "stable_id"
VALID_IDS = [1, 2, 3, 4]

# ✅ READY / KNEE (optional columns from tracker)
LOW_KNEE_ANGLE_DEG = 150.0

# ✅ doubles role thresholds (tunable)
ROLE_HEAVY_THRESH = 60.0  # >=60% on own side front/back => "heavy"


# ----------------------------
# Helpers
# ----------------------------
def ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)


def pick_latest_csv(exports_dir: str = "exports") -> str:
    files = sorted(glob.glob(os.path.join(exports_dir, "*.csv")), key=os.path.getmtime)
    if not files:
        raise FileNotFoundError(f"No CSV files found in {exports_dir}/")
    return files[-1]


def ema_smooth(x: np.ndarray, alpha: float) -> np.ndarray:
    if len(x) == 0:
        return x
    y = np.empty_like(x, dtype=float)
    y[0] = x[0]
    for i in range(1, len(x)):
        y[i] = alpha * x[i] + (1 - alpha) * y[i - 1]
    return y


def fmt(v, decimals=2) -> str:
    try:
        if v is None or (isinstance(v, float) and not np.isfinite(v)):
            return "N/A"
        return f"{float(v):.{decimals}f}"
    except Exception:
        return str(v)


# ----------------------------
# Court drawing (full doubles court + singles lines)
# ----------------------------
def draw_badminton_court(ax, court_w: float, court_l: float) -> None:
    white = "#ffffff"
    lw_outer = 4.0
    lw_inner = 2.8

    W = court_w
    L = court_l
    net_y = L / 2.0

    # Doubles outer rectangle
    ax.plot([0, W, W, 0, 0], [0, 0, L, L, 0], color=white, linewidth=lw_outer, zorder=6)

    # Net
    ax.plot([0, W], [net_y, net_y], color=white, linewidth=lw_outer, zorder=6)

    # Singles sidelines (5.18m width, centered)
    singles_w = 5.18
    margin = (W - singles_w) / 2.0
    sx0 = margin
    sx1 = W - margin
    ax.plot([sx0, sx0], [0, L], color=white, linewidth=lw_inner, zorder=6)
    ax.plot([sx1, sx1], [0, L], color=white, linewidth=lw_inner, zorder=6)

    # Short service lines (1.98m from net)
    short = 1.98
    ax.plot([0, W], [net_y - short, net_y - short], color=white, linewidth=lw_inner, zorder=6)
    ax.plot([0, W], [net_y + short, net_y + short], color=white, linewidth=lw_inner, zorder=6)

    # Center line (only between net and short service lines)
    cx = W / 2.0
    ax.plot([cx, cx], [net_y - short, net_y], color=white, linewidth=lw_inner, zorder=6)
    ax.plot([cx, cx], [net_y, net_y + short], color=white, linewidth=lw_inner, zorder=6)

    # Doubles long service line (0.76m from baseline)
    long_doubles = 0.76
    ax.plot([0, W], [long_doubles, long_doubles], color=white, linewidth=lw_inner, zorder=6)
    ax.plot([0, W], [L - long_doubles, L - long_doubles], color=white, linewidth=lw_inner, zorder=6)


# ----------------------------
# Heatmap with court background
# ----------------------------
def plot_heatmap(df_player: pd.DataFrame, out_path: str, title: str) -> None:
    x = df_player["x_m"].to_numpy()
    y = df_player["y_m"].to_numpy()
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]

    if len(x) < 5:
        return

    x = np.clip(x, 0, COURT_W)
    y = np.clip(y, 0, COURT_L)

    H, _, _ = np.histogram2d(
        x, y,
        bins=[HEATMAP_BINS_X, HEATMAP_BINS_Y],
        range=[[0, COURT_W], [0, COURT_L]]
    )
    Z = H.T

    if np.any(Z > 0):
        vmax = float(np.percentile(Z[Z > 0], 99.5))
        vmin = float(np.percentile(Z[Z > 0], 5))
    else:
        vmax, vmin = 1.0, 1e-6
    vmax = max(vmax, 1e-6)
    vmin = max(vmin, 1e-6)
    norm = LogNorm(vmin=vmin, vmax=vmax)

    fig_w = 6.0
    fig_h = fig_w * (COURT_L / COURT_W)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=260, constrained_layout=True)

    green = "#1f7a3a"
    ax.add_patch(patches.Rectangle((0, 0), COURT_W, COURT_L, facecolor=green, edgecolor="none", zorder=0))

    im = ax.imshow(
        Z,
        origin="upper",
        extent=[0, COURT_W, COURT_L, 0],
        cmap="inferno",
        norm=norm,
        interpolation="bilinear",
        alpha=0.72,
        zorder=2
    )

    draw_badminton_court(ax, COURT_W, COURT_L)

    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(0, COURT_W)
    ax.set_ylim(COURT_L, 0)

    ax.set_title(title)
    ax.set_xlabel("x (m) - left → right")
    ax.set_ylabel("y (m) - front → back")

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Occupancy (log)")
    fig.savefig(out_path, dpi=260)
    plt.close(fig)


def plot_speed_series(df_player: pd.DataFrame, out_path: str, title: str) -> None:
    plt.figure(figsize=(10, 3))
    plt.plot(df_player["timestamp_s"], df_player["speed_mps"])
    plt.title(title)
    plt.xlabel("time (s)")
    plt.ylabel("speed (m/s)")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def compute_player_series(df_player: pd.DataFrame) -> pd.DataFrame:
    dfp = df_player.sort_values(["frame_idx"]).drop_duplicates(subset=["frame_idx"], keep="last").copy()

    dfp["dx"] = dfp["x_m"].diff()
    dfp["dy"] = dfp["y_m"].diff()
    dfp["dt"] = dfp["timestamp_s"].diff()

    dfp["step_m"] = np.sqrt(dfp["dx"] ** 2 + dfp["dy"] ** 2)
    dfp.loc[dfp["dt"] <= 0, ["step_m"]] = np.nan

    dfp["speed_mps_raw"] = dfp["step_m"] / dfp["dt"]
    dfp["speed_mps_raw"] = dfp["speed_mps_raw"].clip(lower=0, upper=12)

    raw = dfp["speed_mps_raw"].to_numpy()
    dfp["speed_mps"] = ema_smooth(np.nan_to_num(raw, nan=0.0), EMA_ALPHA)
    return dfp


# ----------------------------
# ✅ Doubles-aware zone parsing + breakdown
# ----------------------------
def parse_zone(z: str) -> Tuple[str, str, str]:
    """
    Expected formats:
      - "FAR-FRONT-LEFT"
      - "NEAR-BACK-RIGHT"
    Returns: side_group, depth, lr
    """
    try:
        parts = str(z).split("-")
        if len(parts) == 3:
            return parts[0], parts[1], parts[2]
    except Exception:
        pass
    return "UNK", "UNK", "UNK"


def classify_role_profile(time_far_pct: float, time_near_pct: float,
                          far_front_pct: float, far_back_pct: float,
                          near_front_pct: float, near_back_pct: float) -> str:
    """
    Role profile based on FRONT/BACK on the player's "own" side (where they spend most time FAR vs NEAR).
    """
    primary_side = "FAR" if time_far_pct >= time_near_pct else "NEAR"
    if primary_side == "FAR":
        own_front, own_back = far_front_pct, far_back_pct
    else:
        own_front, own_back = near_front_pct, near_back_pct

    if own_front >= ROLE_HEAVY_THRESH:
        return "Net-heavy"
    if own_back >= ROLE_HEAVY_THRESH:
        return "Back-heavy"
    return "Rotating"


def zone_breakdown(df_player: pd.DataFrame) -> dict:
    zones = df_player["zone"].astype(str).values
    total = len(zones)
    if total == 0:
        return {
            "time_front_pct": 0.0, "time_back_pct": 0.0,
            "time_left_pct": 0.0, "time_right_pct": 0.0,
            "time_far_pct": 0.0, "time_near_pct": 0.0,
            "far_front_pct": 0.0, "far_back_pct": 0.0,
            "near_front_pct": 0.0, "near_back_pct": 0.0,
            "role_profile": "N/A",
            "top_zone": "N/A",
        }

    sides, depths, lrs = [], [], []
    for z in zones:
        s, d, lr = parse_zone(z)
        sides.append(s)
        depths.append(d)
        lrs.append(lr)

    # overall depth + left/right (stadig nyttigt)
    front = np.sum([d == "FRONT" for d in depths])
    back  = np.sum([d == "BACK" for d in depths])
    left  = np.sum([lr == "LEFT" for lr in lrs])
    right = np.sum([lr == "RIGHT" for lr in lrs])

    # FAR/NEAR (side of net)
    far  = np.sum([s == "FAR" for s in sides])
    near = np.sum([s == "NEAR" for s in sides])

    # per side: FRONT/BACK distribution (normalized within that side)
    far_front = np.sum([(s == "FAR" and d == "FRONT") for s, d in zip(sides, depths)])
    far_back  = np.sum([(s == "FAR" and d == "BACK")  for s, d in zip(sides, depths)])
    near_front = np.sum([(s == "NEAR" and d == "FRONT") for s, d in zip(sides, depths)])
    near_back  = np.sum([(s == "NEAR" and d == "BACK")  for s, d in zip(sides, depths)])

    vals, counts = np.unique(zones, return_counts=True)
    top_zone = vals[int(np.argmax(counts))]

    time_far_pct = 100.0 * float(far) / float(total)
    time_near_pct = 100.0 * float(near) / float(total)

    far_front_pct = 100.0 * float(far_front) / float(max(far, 1))
    far_back_pct  = 100.0 * float(far_back)  / float(max(far, 1))
    near_front_pct = 100.0 * float(near_front) / float(max(near, 1))
    near_back_pct  = 100.0 * float(near_back)  / float(max(near, 1))

    role_profile = classify_role_profile(
        time_far_pct, time_near_pct,
        far_front_pct, far_back_pct,
        near_front_pct, near_back_pct
    )

    return {
        "time_front_pct": 100.0 * float(front) / float(total),
        "time_back_pct":  100.0 * float(back)  / float(total),
        "time_left_pct":  100.0 * float(left)  / float(total),
        "time_right_pct": 100.0 * float(right) / float(total),

        "time_far_pct": time_far_pct,
        "time_near_pct": time_near_pct,

        "far_front_pct": far_front_pct,
        "far_back_pct": far_back_pct,
        "near_front_pct": near_front_pct,
        "near_back_pct": near_back_pct,

        "role_profile": role_profile,
        "top_zone": str(top_zone),
    }


# ----------------------------
# ✅ Ready / knee metrics (optional columns)
# ----------------------------
def compute_ready_knee_stats(dfp_raw: pd.DataFrame) -> dict:
    out = {
        "ready_time_s": None,
        "ready_time_pct": None,
        "ready_pct_mean": None,
        "knee_angle_mean_deg": None,
        "knee_angle_median_deg": None,
        "knee_angle_p25_deg": None,
        "knee_angle_p75_deg": None,
        "low_knee_time_pct": None,
    }

    has_ready_flag = "ready_flag" in dfp_raw.columns
    has_ready_pct = "ready_pct" in dfp_raw.columns
    has_knee = "knee_angle_avg" in dfp_raw.columns
    if not (has_ready_flag or has_ready_pct or has_knee):
        return out

    dfp = dfp_raw.sort_values(["frame_idx"]).drop_duplicates(subset=["frame_idx"], keep="last").copy()

    ts = pd.to_numeric(dfp["timestamp_s"], errors="coerce").to_numpy()
    dt = np.diff(ts, prepend=np.nan)
    dt[~np.isfinite(dt)] = 0.0
    dt = np.clip(dt, 0.0, 1.0)

    if has_ready_flag:
        rf = pd.to_numeric(dfp["ready_flag"], errors="coerce").fillna(0).astype(int).to_numpy()
        total_time = float(np.sum(dt))
        ready_time = float(np.sum(dt[rf == 1]))
        out["ready_time_s"] = ready_time
        out["ready_time_pct"] = (100.0 * ready_time / total_time) if total_time > 1e-6 else 0.0

    if has_ready_pct:
        rp = pd.to_numeric(dfp["ready_pct"], errors="coerce").to_numpy()
        rp = rp[np.isfinite(rp)]
        out["ready_pct_mean"] = float(np.mean(rp)) if len(rp) else None

    if has_knee:
        ka = pd.to_numeric(dfp["knee_angle_avg"], errors="coerce").to_numpy()
        ka = ka[np.isfinite(ka)]
        if len(ka):
            out["knee_angle_mean_deg"] = float(np.mean(ka))
            out["knee_angle_median_deg"] = float(np.median(ka))
            out["knee_angle_p25_deg"] = float(np.percentile(ka, 25))
            out["knee_angle_p75_deg"] = float(np.percentile(ka, 75))

        ka_full = pd.to_numeric(dfp["knee_angle_avg"], errors="coerce").to_numpy()
        mask = np.isfinite(ka_full)
        total_time = float(np.sum(dt[mask]))
        low_time = float(np.sum(dt[(mask) & (ka_full <= LOW_KNEE_ANGLE_DEG)]))
        out["low_knee_time_pct"] = (100.0 * low_time / total_time) if total_time > 1e-6 else 0.0

    return out


# ----------------------------
# ✅ Doubles-aware coaching bullets (no single-style judging)
# ----------------------------
def coaching_bullets(player_stats: dict) -> List[str]:
    bullets = []

    dist = float(player_stats.get("total_distance_m", 0.0))
    p95 = float(player_stats.get("p95_speed_mps", 0.0))
    mean_speed = float(player_stats.get("mean_speed_mps", 0.0))

    left = float(player_stats.get("time_left_pct", 0.0))
    right = float(player_stats.get("time_right_pct", 0.0))

    top_zone = str(player_stats.get("top_zone", "N/A"))

    # Movement volume (neutral/pro)
    if dist > 900:
        bullets.append("Meget høj bevægelsesvolumen — matcher ofte højt rally-tempo. Kig efter mikrosparer: kortere recovery og tidligere base-stop.")
    elif dist > 650:
        bullets.append("Høj bevægelsesvolumen — næste step: minimér små ekstra-skridt på sidste meter ind til base.")
    else:
        bullets.append("Kontrolleret bevægelsesvolumen — kan indikere god anticipation. Tjek om du stadig tager nok dybde når modstanderen presser bagud.")

    # Speed bursts (pro calibration)
    if p95 >= 3.8:
        bullets.append("Stærke bursts (P95) — hold fokus på stabil landing og retning efter split-step.")
    elif p95 >= 3.0:
        bullets.append("Gode bursts — næste step er lidt mere ‘snap’ i de første 2 skridt efter retningsskift.")
    else:
        bullets.append("Mere konservative bursts i denne sekvens — kan være taktisk. Hvis ikke: målret eksplosivitet i første step.")

    # Mean speed (tempo proxy)
    if mean_speed >= 1.9:
        bullets.append("Højt gennemsnitstempo — flot arbejdsrate. Fokusér på ‘effort efficiency’ (samme tempo med mindre energispild).")
    elif mean_speed >= 1.4:
        bullets.append("Stabilt tempo — næste step: skift gear hurtigere (neutral → burst) på modstanders kontakt.")
    else:
        bullets.append("Lavere gennemsnitstempo i denne sekvens — kan skyldes korte point/pauser. Mål evt. på rene rally-segmenter for mere retvisende tempo.")

    # Doubles-aware positioning: describe role/rotation instead of judging
    far_pct = float(player_stats.get("time_far_pct", 0.0))
    near_pct = float(player_stats.get("time_near_pct", 0.0))
    role_profile = str(player_stats.get("role_profile", "N/A"))

    primary_side = "FAR" if far_pct >= near_pct else "NEAR"
    bullets.append(f"Doubles position: primært på **{primary_side}**-siden (FAR {far_pct:.0f}% / NEAR {near_pct:.0f}%). Rolleprofil: **{role_profile}**.")

    # Side bias
    if abs(left - right) > 25:
        side = "venstre" if left > right else "højre"
        bullets.append(f"Side-bias mod {side} — kan være taktik. Vurdér om modsatte side bliver ‘åben’ i transitions.")
    else:
        bullets.append("Sidefordeling ser relativt balanceret ud — godt udgangspunkt for symmetric coverage.")

    # Stance / ready (optional, neutral)
    ready_time_pct = player_stats.get("ready_time_pct", None)
    knee_med = player_stats.get("knee_angle_median_deg", None)
    low_knee_pct = player_stats.get("low_knee_time_pct", None)

    if ready_time_pct is not None:
        bullets.append(f"Ready time: {ready_time_pct:.1f}% — brug den som KPI (timing på cues fremfor konstant lav stance).")

    if knee_med is not None and low_knee_pct is not None:
        bullets.append(f"Knæ median: {knee_med:.0f}° | Lav knæ (≤{LOW_KNEE_ANGLE_DEG:.0f}°): {low_knee_pct:.1f}% — brug til at følge stance over tid.")

    bullets.append(f"Mest brugte zone: **{top_zone}** — brug som reference og check om du resetter dertil efter egne slag.")
    return bullets[:7]


# ----------------------------
# ✅ HTML REPORT: Summary first, then "everything per player"
# ----------------------------
def build_html_report(
    summary_df_with_paths: pd.DataFrame,
    bullets_by_player: Dict[int, List[str]],
    out_path: str,
) -> None:
    cols_to_hide = {"heatmap_png", "speed_png", "series_csv"}
    summary_visible = summary_df_with_paths.drop(
        columns=[c for c in cols_to_hide if c in summary_df_with_paths.columns],
        errors="ignore",
    ).copy()

    rows_html = summary_visible.to_html(index=False, float_format=lambda x: f"{x:.2f}")

    report_dir = os.path.dirname(out_path) or "."

    player_sections_html = ""
    df_sorted = summary_df_with_paths.sort_values(["player_id"]).copy()

    for _, r in df_sorted.iterrows():
        pid = int(r["player_id"])

        heatmap_path = str(r.get("heatmap_png", ""))
        speed_path = str(r.get("speed_png", ""))
        series_path = str(r.get("series_csv", ""))

        heatmap_rel = os.path.relpath(heatmap_path, start=report_dir) if heatmap_path else ""
        speed_rel = os.path.relpath(speed_path, start=report_dir) if speed_path else ""
        series_rel = os.path.relpath(series_path, start=report_dir) if series_path else ""

        heatmap_tag = (
            f'<img src="{heatmap_rel}" alt="Heatmap Player {pid}" '
            f'style="max-width:520px;width:100%;height:auto;border-radius:10px;border:1px solid #e6e6e6;">'
            if heatmap_rel else "<em>No heatmap</em>"
        )
        speed_tag = (
            f'<img src="{speed_rel}" alt="Speed Player {pid}" '
            f'style="max-width:820px;width:100%;height:auto;border-radius:10px;border:1px solid #e6e6e6;">'
            if speed_rel else "<em>No speed plot</em>"
        )
        series_link = (f'<a href="{series_rel}" download>Download series CSV</a>' if series_rel else "")

        # Key stats table (includes doubles role stats)
        key_stats_rows = [
            ("Samples", str(int(r.get("samples", 0)))),
            ("Total distance (m)", fmt(r.get("total_distance_m", 0.0))),
            ("Mean speed (m/s)", fmt(r.get("mean_speed_mps", 0.0))),
            ("Median speed (m/s)", fmt(r.get("median_speed_mps", 0.0))),
            ("P95 speed (m/s)", fmt(r.get("p95_speed_mps", 0.0))),
            ("Max speed (m/s)", fmt(r.get("max_speed_mps", 0.0))),
            ("Time left/right (%)", f"{fmt(r.get('time_left_pct',0.0),1)} / {fmt(r.get('time_right_pct',0.0),1)}"),
            ("Role profile", str(r.get("role_profile", "N/A"))),
            ("Time FAR/NEAR (%)", f"{fmt(r.get('time_far_pct',0.0),1)} / {fmt(r.get('time_near_pct',0.0),1)}"),
            ("FAR front/back (%)", f"{fmt(r.get('far_front_pct',0.0),1)} / {fmt(r.get('far_back_pct',0.0),1)}"),
            ("NEAR front/back (%)", f"{fmt(r.get('near_front_pct',0.0),1)} / {fmt(r.get('near_back_pct',0.0),1)}"),
            ("Top zone", str(r.get("top_zone", "N/A"))),
        ]

        # stance rows if present
        if pd.notna(r.get("ready_time_pct", np.nan)):
            key_stats_rows += [
                ("Ready time (%)", fmt(r.get("ready_time_pct", None), 1)),
                ("Ready time (s)", fmt(r.get("ready_time_s", None), 1)),
            ]
        if pd.notna(r.get("ready_pct_mean", np.nan)):
            key_stats_rows += [("Ready% mean (tracker)", fmt(r.get("ready_pct_mean", None), 1))]
        if pd.notna(r.get("knee_angle_median_deg", np.nan)):
            key_stats_rows += [
                ("Knee median (deg)", fmt(r.get("knee_angle_median_deg", None), 0)),
                ("Knee mean (deg)", fmt(r.get("knee_angle_mean_deg", None), 0)),
                ("Knee p25/p75 (deg)", f"{fmt(r.get('knee_angle_p25_deg', None),0)} / {fmt(r.get('knee_angle_p75_deg', None),0)}"),
                (f"Low knee ≤{LOW_KNEE_ANGLE_DEG:.0f} (%)", fmt(r.get("low_knee_time_pct", None), 1)),
            ]

        key_stats_html = "<table class='mini'><tbody>"
        for k, v in key_stats_rows:
            key_stats_html += f"<tr><th>{k}</th><td>{v}</td></tr>"
        key_stats_html += "</tbody></table>"

        bullets = bullets_by_player.get(pid, [])
        bullets_html = "<ul>" + "".join([f"<li>{b}</li>" for b in bullets]) + "</ul>" if bullets else "<em>No bullets</em>"

        player_sections_html += f"""
        <section class="card" id="player-{pid}">
          <div class="card-header">
            <h2 style="margin:0">Player {pid}</h2>
            <div class="muted">{series_link}</div>
          </div>

          <div class="grid3">
            <div>
              <h3 style="margin:0 0 10px 0">Key stats</h3>
              {key_stats_html}
            </div>

            <div>
              <h3 style="margin:0 0 10px 0">Coaching bullets</h3>
              {bullets_html}
            </div>

            <div>
              <h3 style="margin:0 0 10px 0">Plots</h3>
              <div>
                <div class="plot-title">Heatmap</div>
                {heatmap_tag}
              </div>
              <div style="margin-top:14px;">
                <div class="plot-title">Speed over time</div>
                {speed_tag}
              </div>
            </div>
          </div>
        </section>
        """

    html = f"""
<!doctype html>
<html>
<head>
  <meta charset="utf-8"/>
  <title>Badminton Metrics Report</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 24px; color:#111; }}
    table {{ border-collapse: collapse; width: 100%; font-size: 14px; }}
    th, td {{ border: 1px solid #ddd; padding: 8px; }}
    th {{ background: #f5f5f5; text-align: left; }}

    h1 {{ margin: 0 0 8px 0; }}
    h2 {{ margin: 0; }}
    h3 {{ font-size: 16px; }}
    .muted {{ color:#666; font-size: 13px; }}

    .card {{
      margin-top: 18px;
      padding: 16px;
      border: 1px solid #e6e6e6;
      border-radius: 12px;
      background: #fff;
    }}
    .card-header {{
      display:flex;
      align-items:baseline;
      justify-content:space-between;
      gap: 12px;
      margin-bottom: 14px;
      flex-wrap: wrap;
    }}

    .grid3 {{
      display:grid;
      grid-template-columns: 1fr;
      gap: 16px;
    }}
    @media (min-width: 1100px) {{
      .grid3 {{ grid-template-columns: 380px 1fr 1fr; }}
    }}

    table.mini {{
      width:100%;
      font-size: 13px;
    }}
    table.mini th {{
      width: 58%;
      background: #fafafa;
    }}
    table.mini td {{
      width: 42%;
      text-align: right;
      font-variant-numeric: tabular-nums;
    }}

    .plot-title {{
      font-size: 13px;
      color: #444;
      margin: 0 0 6px 0;
    }}
  </style>
</head>
<body>
  <h1>Badminton Metrics Report</h1>
  <div class="muted">Generated: {pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")}</div>

  <h2 style="margin-top:20px;">Overall Summary</h2>
  {rows_html}

  {player_sections_html}

  <p style="margin-top:24px;color:#666">
    Note: Metrics based on foot-point tracking in court coordinates. Doubles-aware zone stats use FAR/NEAR + FRONT/BACK on each side.
  </p>
</body>
</html>
"""
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html)


# ----------------------------
# MAIN ANALYSIS
# ----------------------------
def main(csv_path: Optional[str] = None) -> None:
    ensure_dir(OUT_DIR)
    ensure_dir(HEATMAP_DIR)
    ensure_dir(PLOTS_DIR)
    ensure_dir(SERIES_DIR)

    if csv_path is None:
        csv_path = pick_latest_csv("exports")

    print(f"Loading: {csv_path}")
    df = pd.read_csv(csv_path)

    id_col = ID_COL_DEFAULT if ID_COL_DEFAULT in df.columns else "track_id"

    needed = {"frame_idx", "timestamp_s", id_col, "x_m", "y_m", "zone"}
    missing = needed - set(df.columns)
    if missing:
        raise ValueError(f"CSV missing required columns: {missing}")

    if ONLY_PRIMARY_SINGLE and "is_primary_single" in df.columns:
        df = df[df["is_primary_single"] == 1].copy()

    df[id_col] = pd.to_numeric(df[id_col], errors="coerce").fillna(-1).astype(int)
    df["x_m"] = pd.to_numeric(df["x_m"], errors="coerce")
    df["y_m"] = pd.to_numeric(df["y_m"], errors="coerce")
    df["timestamp_s"] = pd.to_numeric(df["timestamp_s"], errors="coerce")
    df["frame_idx"] = pd.to_numeric(df["frame_idx"], errors="coerce").fillna(-1).astype(int)

    df = df[np.isfinite(df["x_m"]) & np.isfinite(df["y_m"]) & np.isfinite(df["timestamp_s"])].copy()
    df["x_m"] = df["x_m"].clip(0, COURT_W)
    df["y_m"] = df["y_m"].clip(0, COURT_L)

    for c in ["ready_flag", "ready_pct", "knee_angle_avg"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    if id_col == "stable_id":
        df = df[df["stable_id"].isin(VALID_IDS)].copy()

    player_ids = sorted(df[id_col].unique().tolist())
    print(f"Found players ({id_col}): {player_ids}")

    summary_rows = []
    bullets_by_player: Dict[int, List[str]] = {}

    for pid in player_ids:
        dfp_raw = df[df[id_col] == pid].copy()
        if len(dfp_raw) < 10:
            continue

        dfp = compute_player_series(dfp_raw)

        total_distance = float(np.nansum(dfp["step_m"].to_numpy()))
        speeds = dfp["speed_mps"].to_numpy()
        speeds = speeds[np.isfinite(speeds)]
        mean_speed = float(np.mean(speeds)) if len(speeds) else 0.0
        med_speed = float(np.median(speeds)) if len(speeds) else 0.0
        max_speed = float(np.max(speeds)) if len(speeds) else 0.0
        p95_speed = float(np.percentile(speeds, 95)) if len(speeds) else 0.0

        zb = zone_breakdown(dfp_raw)
        stance = compute_ready_knee_stats(dfp_raw)

        series_path = os.path.join(SERIES_DIR, f"player_{pid}_series.csv")
        dfp[["frame_idx", "timestamp_s", "x_m", "y_m", "step_m", "speed_mps"]].to_csv(series_path, index=False)

        heatmap_path = os.path.join(HEATMAP_DIR, f"player_{pid}_heatmap.png")
        plot_heatmap(dfp_raw, heatmap_path, title=f"Heatmap - Player {pid}")

        speed_plot_path = os.path.join(PLOTS_DIR, f"player_{pid}_speed.png")
        plot_speed_series(dfp, speed_plot_path, title=f"Speed over time - Player {pid}")

        stats = {
            "player_id": int(pid),
            "id_col": id_col,
            "samples": int(len(dfp_raw)),
            "total_distance_m": total_distance,
            "mean_speed_mps": mean_speed,
            "median_speed_mps": med_speed,
            "p95_speed_mps": p95_speed,
            "max_speed_mps": max_speed,
            **zb,
            **stance,
            "heatmap_png": heatmap_path,
            "speed_png": speed_plot_path,
            "series_csv": series_path,
        }

        summary_rows.append(stats)
        bullets_by_player[int(pid)] = coaching_bullets(stats)

    if not summary_rows:
        raise RuntimeError("No player had enough samples to compute metrics.")

    summary_df = pd.DataFrame(summary_rows).sort_values("total_distance_m", ascending=False)

    summary_csv = os.path.join(OUT_DIR, "summary.csv")
    summary_df.to_csv(summary_csv, index=False)
    print(f"✅ Summary CSV: {summary_csv}")

    report_html = os.path.join(OUT_DIR, "report.html")
    build_html_report(summary_df, bullets_by_player, report_html)
    print(f"✅ HTML report: {report_html}")

    print("\nDone.")
    print(f"Outputs in: {OUT_DIR}/")
    print(f" - heatmaps: {HEATMAP_DIR}/")
    print(f" - plots:    {PLOTS_DIR}/")
    print(f" - series:   {SERIES_DIR}/")


if __name__ == "__main__":
    main()
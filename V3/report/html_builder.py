"""
html_builder — generates a self-contained HTML report from PlayerSummary objects
and CoachingInsight lists. All chart data is embedded as JSON in <script> tags;
no external CDN dependencies beyond Chart.js (loaded from CDN, gracefully
degrades when offline).
"""

from __future__ import annotations

import json
import os
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from report.metrics_v3 import PlayerSummary
from report.feedback import CoachingInsight

# ── colour palette ──────────────────────────────────────────────────────────
_PLAYER_COLOURS = {1: "#4fc3f7", 2: "#f06292", 3: "#aed581", 4: "#ffb74d"}
_SEVERITY_COLOUR = {"positive": "#69f0ae", "warning": "#ff8a65", "info": "#90caf9"}


def _safe(v, fmt=".1f"):
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return "—"
    return format(v, fmt)


def build_report(
    summaries: Dict[int, PlayerSummary],
    insights: List[CoachingInsight],
    session_meta: dict,
    output_path: str,
    skeleton_df: Optional[pd.DataFrame] = None,
    shuttle_df: Optional[pd.DataFrame] = None,
) -> str:
    """Write report.html and return its path."""

    pids = sorted(summaries.keys())

    # ── speed timeline data ──────────────────────────────────────────────────
    speed_data: dict = {}
    if skeleton_df is not None and not skeleton_df.empty:
        for pid in pids:
            pdf = skeleton_df[skeleton_df["stable_id"] == pid].sort_values("timestamp_s")
            if "speed_kmh" in pdf.columns:
                speed_data[pid] = {
                    "t": pdf["timestamp_s"].clip(lower=0).round(2).tolist(),
                    "v": pdf["speed_kmh"].clip(lower=0, upper=50).round(2).tolist(),
                }

    # ── shuttle direction change markers ────────────────────────────────────
    dc_times: list = []
    if shuttle_df is not None and not shuttle_df.empty:
        dc = shuttle_df[shuttle_df["direction_change_flag"] == 1]
        dc_times = dc["timestamp_s"].round(2).tolist()

    # ── shot type data ───────────────────────────────────────────────────────
    shot_labels = ["smash", "overhead_clear", "drive", "net_drop", "unknown"]
    shot_data: dict = {}
    for pid in pids:
        ps = summaries[pid]
        shot_data[pid] = [ps.shot_type_breakdown.get(sl, 0) for sl in shot_labels]

    # ── summary table rows ───────────────────────────────────────────────────
    table_rows_html = ""
    fields = [
        ("Total Distance (m)", lambda ps: _safe(ps.total_distance_m, ".0f")),
        ("Mean Speed (km/h)",  lambda ps: _safe(ps.mean_speed_mps * 3.6 if ps.mean_speed_mps else None)),
        ("Max Speed (km/h)",   lambda ps: _safe(ps.max_speed_mps * 3.6 if ps.max_speed_mps else None)),
        ("Ready Time (%)",     lambda ps: _safe(ps.ready_time_pct)),
        ("Knee Angle Mean°",   lambda ps: _safe(ps.knee_angle_mean_deg)),
        ("Reaction Time (ms)", lambda ps: _safe(ps.mean_reaction_time_s * 1000 if ps.mean_reaction_time_s else None, ".0f")),
        ("Reaction Events",    lambda ps: str(ps.reaction_events)),
        ("Shots Total",        lambda ps: str(ps.shots_total)),
        ("Mean Elbow Angle°",  lambda ps: _safe(ps.mean_elbow_angle_R)),
        ("Mean Hip Rotation°", lambda ps: _safe(ps.mean_hip_rotation)),
        ("Team Spread (m)",    lambda ps: _safe(ps.mean_team_spread_m)),
        ("Role Profile",       lambda ps: ps.role_profile or "—"),
        ("Top Zone",           lambda ps: ps.top_zone or "—"),
        ("Court Crossings",    lambda ps: str(ps.crossing_events_count)),
    ]
    for label, fn in fields:
        cells = "".join(
            f'<td style="color:{_PLAYER_COLOURS.get(pid,"#fff")}">{fn(summaries[pid])}</td>'
            for pid in pids
        )
        table_rows_html += f"<tr><td>{label}</td>{cells}</tr>\n"

    # ── insight bullets HTML ─────────────────────────────────────────────────
    bullets_per_player: Dict[int, str] = {pid: "" for pid in pids}
    bullets_global = ""
    for ins in insights:
        colour = _SEVERITY_COLOUR.get(ins.severity, "#90caf9")
        icon = {"positive": "✓", "warning": "⚠", "info": "ℹ"}.get(ins.severity, "•")
        li = f'<li style="border-left:3px solid {colour};padding-left:8px;margin:4px 0">{icon} {ins.text}</li>\n'
        if ins.player_id == 0:
            bullets_global += li
        elif ins.player_id in bullets_per_player:
            bullets_per_player[ins.player_id] += li

    # ── zone breakdown table ─────────────────────────────────────────────────
    zone_fields = [
        ("Far %",         "time_far_pct"),
        ("Near %",        "time_near_pct"),
        ("Front %",       "time_front_pct"),
        ("Back %",        "time_back_pct"),
        ("Left %",        "time_left_pct"),
        ("Right %",       "time_right_pct"),
        ("Far-Front %",   "far_front_pct"),
        ("Far-Back %",    "far_back_pct"),
        ("Near-Front %",  "near_front_pct"),
        ("Near-Back %",   "near_back_pct"),
    ]
    zone_rows_html = ""
    for label, attr in zone_fields:
        cells = "".join(
            f'<td style="color:{_PLAYER_COLOURS.get(pid,"#fff")}">{_safe(getattr(summaries[pid], attr, None))}</td>'
            for pid in pids
        )
        zone_rows_html += f"<tr><td>{label}</td>{cells}</tr>\n"

    # ── player header cells ──────────────────────────────────────────────────
    header_cells = "".join(
        f'<th style="color:{_PLAYER_COLOURS.get(pid,"#fff")}">Player {pid}</th>'
        for pid in pids
    )

    pid_colours_js = json.dumps(_PLAYER_COLOURS)
    speed_data_js   = json.dumps(speed_data)
    dc_times_js     = json.dumps(dc_times)
    shot_data_js    = json.dumps(shot_data)
    shot_labels_js  = json.dumps(shot_labels)
    meta_js         = json.dumps(session_meta, default=str)
    pids_js         = json.dumps(pids)

    video_name = session_meta.get("video_name", "session")
    session_id = session_meta.get("session_id", "")
    total_frames = session_meta.get("total_frames", 0)
    fps = session_meta.get("fps", 30)
    duration_s = total_frames / max(fps, 1) if total_frames else 0

    player_cards_html = ""
    for pid in pids:
        ps = summaries[pid]
        colour = _PLAYER_COLOURS.get(pid, "#fff")
        player_cards_html += f"""
<div class="player-card" style="border-top:3px solid {colour}">
  <h3 style="color:{colour}">Player {pid} — {ps.role_profile}</h3>
  <p>Shots: {ps.shots_total} &nbsp;|&nbsp;
     Reaction events: {ps.reaction_events} &nbsp;|&nbsp;
     Court crossings: {ps.crossing_events_count}</p>
  <div style="font-size:0.85em">
    <strong>Shot breakdown:</strong>
    {' | '.join(f'{k}: {v}' for k, v in ps.shot_type_breakdown.items()) or '—'}
  </div>
  <ul class="bullets">{bullets_per_player[pid] or '<li>No specific insights.</li>'}</ul>
  <canvas id="shot-radar-{pid}" width="300" height="300"
          style="display:block;margin:12px auto"></canvas>
</div>
"""

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>SkillPeak V3 — {video_name}</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4/dist/chart.umd.min.js"></script>
<style>
  :root{{--bg:#121212;--card:#1e1e1e;--text:#e0e0e0;--muted:#9e9e9e;--radius:8px}}
  *{{box-sizing:border-box;margin:0;padding:0}}
  body{{background:var(--bg);color:var(--text);font-family:system-ui,sans-serif;padding:20px}}
  h1,h2,h3{{font-weight:600;margin-bottom:8px}}
  h1{{font-size:1.6em;margin-bottom:4px}}
  .subtitle{{color:var(--muted);font-size:0.9em;margin-bottom:24px}}
  .grid-2{{display:grid;grid-template-columns:1fr 1fr;gap:16px}}
  .card{{background:var(--card);border-radius:var(--radius);padding:16px;margin-bottom:16px}}
  table{{width:100%;border-collapse:collapse;font-size:0.85em}}
  th,td{{padding:6px 10px;border-bottom:1px solid #333;text-align:left}}
  th{{color:var(--muted);font-weight:500}}
  .section{{margin-bottom:28px}}
  .player-card{{background:var(--card);border-radius:var(--radius);padding:16px;margin-bottom:12px}}
  .player-cards{{display:grid;grid-template-columns:repeat(auto-fit,minmax(300px,1fr));gap:14px}}
  ul.bullets{{margin-top:10px;padding-left:0;list-style:none}}
  .chart-wrap{{position:relative;height:260px}}
  .global-bullets{{background:var(--card);border-radius:var(--radius);padding:16px}}
  @media(max-width:640px){{.grid-2{{grid-template-columns:1fr}}}}
</style>
</head>
<body>
<h1>SkillPeak AI Coach — V3 Report</h1>
<div class="subtitle">
  Session: {session_id} &nbsp;|&nbsp;
  Video: {video_name} &nbsp;|&nbsp;
  Duration: {duration_s/60:.1f} min &nbsp;|&nbsp;
  Players: {', '.join(str(p) for p in pids)}
</div>

<!-- SUMMARY TABLE -->
<div class="section card">
  <h2>Session Summary</h2>
  <table>
    <thead><tr><th>Metric</th>{header_cells}</tr></thead>
    <tbody>
{table_rows_html}
    </tbody>
  </table>
</div>

<!-- SPEED CHART -->
<div class="section card">
  <h2>Speed Timeline</h2>
  <div class="chart-wrap"><canvas id="speedChart"></canvas></div>
</div>

<!-- ZONE BREAKDOWN -->
<div class="section card">
  <h2>Court Zone Breakdown</h2>
  <table>
    <thead><tr><th>Zone</th>{header_cells}</tr></thead>
    <tbody>
{zone_rows_html}
    </tbody>
  </table>
</div>

<!-- SHOT TYPE CHART -->
<div class="section card">
  <h2>Shot Type Distribution</h2>
  <div class="chart-wrap"><canvas id="shotChart"></canvas></div>
</div>

<!-- GLOBAL INSIGHTS -->
<div class="section">
  <h2>Team Insights</h2>
  <div class="global-bullets">
    <ul class="bullets">{bullets_global or '<li>No cross-player insights generated.</li>'}</ul>
  </div>
</div>

<!-- PER-PLAYER CARDS -->
<div class="section">
  <h2>Per-Player Analysis</h2>
  <div class="player-cards">
    {player_cards_html}
  </div>
</div>

<script>
const PIDS         = {pids_js};
const COLOURS      = {pid_colours_js};
const SPEED_DATA   = {speed_data_js};
const DC_TIMES     = {dc_times_js};
const SHOT_DATA    = {shot_data_js};
const SHOT_LABELS  = {shot_labels_js};
const META         = {meta_js};

// ── speed chart ──────────────────────────────────────────────────────────────
(function() {{
  const ctx = document.getElementById("speedChart").getContext("2d");
  const datasets = PIDS.map(pid => {{
    const d = SPEED_DATA[pid];
    if (!d) return null;
    // downsample to ~600 pts for chart performance
    const step = Math.max(1, Math.floor(d.t.length / 600));
    return {{
      label: "Player " + pid,
      data: d.t.filter((_,i)=>i%step===0).map((t,i)=>{{
        return {{x: t, y: d.v.filter((_,j)=>j%step===0)[i]}};
      }}),
      borderColor: COLOURS[pid] || "#fff",
      backgroundColor: "transparent",
      borderWidth: 1.5,
      pointRadius: 0,
      tension: 0.3,
    }};
  }}).filter(Boolean);

  const dcAnnotations = DC_TIMES.slice(0, 200).map(t => ({{
    type: "line",
    xMin: t, xMax: t,
    borderColor: "rgba(255,255,255,0.15)",
    borderWidth: 1,
  }}));

  new Chart(ctx, {{
    type: "line",
    data: {{ datasets }},
    options: {{
      animation: false,
      responsive: true,
      maintainAspectRatio: false,
      parsing: false,
      scales: {{
        x: {{ type:"linear", title:{{display:true,text:"Time (s)",color:"#9e9e9e"}}, ticks:{{color:"#9e9e9e"}} }},
        y: {{ title:{{display:true,text:"Speed (km/h)",color:"#9e9e9e"}}, ticks:{{color:"#9e9e9e"}}, min:0 }},
      }},
      plugins: {{
        legend: {{ labels:{{color:"#e0e0e0"}} }},
      }},
    }},
  }});
}})();

// ── shot type bar chart ───────────────────────────────────────────────────────
(function() {{
  const ctx = document.getElementById("shotChart").getContext("2d");
  const datasets = PIDS.map(pid => ({{
    label: "Player " + pid,
    data: SHOT_DATA[pid] || [],
    backgroundColor: COLOURS[pid] || "#fff",
  }}));
  new Chart(ctx, {{
    type: "bar",
    data: {{ labels: SHOT_LABELS, datasets }},
    options: {{
      animation: false,
      responsive: true,
      maintainAspectRatio: false,
      scales: {{
        x: {{ ticks:{{color:"#9e9e9e"}} }},
        y: {{ ticks:{{color:"#9e9e9e"}}, title:{{display:true,text:"Count",color:"#9e9e9e"}} }},
      }},
      plugins: {{ legend:{{ labels:{{color:"#e0e0e0"}} }} }},
    }},
  }});
}})();

// ── per-player shot radar charts ─────────────────────────────────────────────
PIDS.forEach(pid => {{
  const el = document.getElementById("shot-radar-" + pid);
  if (!el) return;
  const ctx = el.getContext("2d");
  const shots = SHOT_DATA[pid] || [];
  const total = shots.reduce((a,b)=>a+b, 0) || 1;
  new Chart(ctx, {{
    type: "radar",
    data: {{
      labels: SHOT_LABELS,
      datasets: [{{
        label: "Player " + pid,
        data: shots.map(v => (v/total*100).toFixed(1)),
        borderColor: COLOURS[pid] || "#fff",
        backgroundColor: (COLOURS[pid] || "#fff") + "33",
      }}],
    }},
    options: {{
      animation: false,
      responsive: false,
      scales: {{
        r: {{
          ticks: {{color:"#9e9e9e", backdropColor:"transparent"}},
          grid: {{color:"#333"}},
          pointLabels: {{color:"#9e9e9e", font:{{size:10}}}},
        }},
      }},
      plugins: {{ legend:{{ display:false }} }},
    }},
  }});
}});
</script>
</body>
</html>
"""

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as fh:
        fh.write(html)
    print(f"V3 — report.html written: {output_path}")
    return output_path

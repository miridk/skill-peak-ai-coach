import os, glob, json
from dataclasses import dataclass, asdict
import numpy as np
import pandas as pd

OUT_DIR = "metrics_out"
SERIES_DIR = os.path.join(OUT_DIR, "series")
EVENTS_DIR = os.path.join(OUT_DIR, "events")
os.makedirs(EVENTS_DIR, exist_ok=True)

MAX_EVENTS_PER_PLAYER = 10
CLUSTER_WINDOW_S = 2.5  # merge events within this many seconds
MIN_GAP_S = 4.0         # final spacing between kept events (soft)

# Tune thresholds
SPEED_PCTILE = 95       # burst threshold
OUT_OF_BASE_M = 1.15    # distance from "base" median position
OUT_OF_BASE_MIN_DUR_S = 1.2

@dataclass
class Event:
    player_id: int
    t_event: float
    t_pause: float
    type: str
    severity: float
    title: str
    body: str
    fix_steps: list
    evidence: dict

def _smooth(x: np.ndarray, w: int = 7) -> np.ndarray:
    if len(x) < w:
        return x
    kernel = np.ones(w) / w
    return np.convolve(x, kernel, mode="same")

def _cluster_events(events, window_s: float):
    if not events:
        return []
    events = sorted(events, key=lambda e: e.t_event)
    clustered = [events[0]]
    for e in events[1:]:
        last = clustered[-1]
        if (e.t_event - last.t_event) <= window_s and e.type == last.type:
            # merge by keeping the higher severity, and average timestamp slightly toward the stronger one
            if e.severity > last.severity:
                clustered[-1] = e
        else:
            clustered.append(e)
    return clustered

def _enforce_spacing(events, min_gap_s: float):
    kept = []
    for e in sorted(events, key=lambda e: (-e.severity, e.t_event)):
        if all(abs(e.t_event - k.t_event) >= min_gap_s for k in kept):
            kept.append(e)
    return sorted(kept, key=lambda e: e.t_event)

def _make_coach_text_for_speed_burst(speed, pctl, dt_hint=None):
    return (
        "Hurtigt burst (ofte reaktion / presset situation)",
        f"Du rammer {speed:.2f} m/s (≈P{pctl}). Fokus: første 2 skridt + stabil landing.",
        ["Split-step på modstanders kontakt", "Første skridt i korrekt retning", "Stabil landing → recovery"],
        {"speed_mps": float(speed), "speed_percentile": int(pctl), "dt_hint_s": dt_hint}
    )

def _make_coach_text_out_of_base(dist, dur):
    return (
        "Lang tid væk fra base",
        f"Du er {dist:.2f} m fra din base i ca. {dur:.1f}s. Risiko: sen på næste slag.",
        ["Efter eget slag: 1–2 recovery steps mod base", "Stop tidligere (undgå ekstra-skridt)", "Split-step når modstander slår"],
        {"dist_from_base_m": float(dist), "duration_s": float(dur)}
    )

def detect_events_for_player(player_id: int, series_csv: str) -> list[Event]:
    df = pd.read_csv(series_csv)
    df = df.sort_values("timestamp_s").dropna(subset=["timestamp_s","x_m","y_m","speed_mps"])
    if len(df) < 30:
        return []

    t = df["timestamp_s"].to_numpy()
    x = df["x_m"].to_numpy()
    y = df["y_m"].to_numpy()
    v = df["speed_mps"].to_numpy()

    # Smooth a little to reduce jitter
    v_s = _smooth(v, w=9)

    # --- Event 1: Speed bursts (top percentile)
    thr = np.percentile(v_s, SPEED_PCTILE)
    burst_idx = np.where(v_s >= thr)[0]

    burst_events = []
    for idx in burst_idx:
        sev = (v_s[idx] - thr) / max(thr, 1e-6)
        title, body, steps, ev = _make_coach_text_for_speed_burst(v_s[idx], SPEED_PCTILE)
        burst_events.append(Event(
            player_id=player_id,
            t_event=float(t[idx]),
            t_pause=float(t[idx] + 0.35),
            type="speed_burst",
            severity=float(sev),
            title=title,
            body=body,
            fix_steps=steps,
            evidence=ev
        ))

    # --- Event 2: Out-of-base (median position as simple base proxy)
    base_x = float(np.median(x))
    base_y = float(np.median(y))
    d_base = np.sqrt((x - base_x)**2 + (y - base_y)**2)

    away = d_base >= OUT_OF_BASE_M
    # find contiguous segments away from base
    out_events = []
    if np.any(away):
        starts = np.where((away[1:] == True) & (away[:-1] == False))[0] + 1
        if away[0]:
            starts = np.r_[0, starts]
        ends = np.where((away[1:] == False) & (away[:-1] == True))[0] + 1
        if away[-1]:
            ends = np.r_[ends, len(away)-1]

        for s, e in zip(starts, ends):
            t0, t1 = float(t[s]), float(t[e])
            dur = t1 - t0
            if dur < OUT_OF_BASE_MIN_DUR_S:
                continue
            # choose peak distance inside segment
            seg = np.arange(s, e+1)
            k = seg[int(np.argmax(d_base[seg]))]
            dist = float(d_base[k])
            sev = (dist - OUT_OF_BASE_M) + 0.25 * dur
            title, body, steps, ev = _make_coach_text_out_of_base(dist, dur)
            out_events.append(Event(
                player_id=player_id,
                t_event=float(t[k]),
                t_pause=float(t[k] + 0.35),
                type="out_of_base",
                severity=float(sev),
                title=title,
                body=body,
                fix_steps=steps,
                evidence=ev | {"base_x": base_x, "base_y": base_y}
            ))

    # Cluster + select
    all_events = burst_events + out_events
    all_events = _cluster_events(all_events, CLUSTER_WINDOW_S)

    # Keep top N by severity but keep spacing
    all_events = sorted(all_events, key=lambda e: e.severity, reverse=True)[:MAX_EVENTS_PER_PLAYER * 3]
    all_events = _enforce_spacing(all_events, MIN_GAP_S)
    all_events = all_events[:MAX_EVENTS_PER_PLAYER]

    # Final sort by time
    return sorted(all_events, key=lambda e: e.t_event)

def main():
    series_files = sorted(glob.glob(os.path.join(SERIES_DIR, "player_*_series.csv")))
    if not series_files:
        raise FileNotFoundError(f"No series CSV found in {SERIES_DIR}")

    all_out = []
    for fp in series_files:
        # extract player id from filename
        name = os.path.basename(fp)
        pid = int(name.split("_")[1])
        events = detect_events_for_player(pid, fp)

        out_path = os.path.join(EVENTS_DIR, f"events_player_{pid}.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump([asdict(e) for e in events], f, ensure_ascii=False, indent=2)
        print(f"✅ {out_path} ({len(events)} events)")
        all_out.extend([asdict(e) for e in events])

    all_path = os.path.join(EVENTS_DIR, "events_all.json")
    with open(all_path, "w", encoding="utf-8") as f:
        json.dump(all_out, f, ensure_ascii=False, indent=2)
    print(f"✅ {all_path} ({len(all_out)} total events)")

if __name__ == "__main__":
    main()
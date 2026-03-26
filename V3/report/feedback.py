"""
CoachFeedbackEngine — generates cross-player coaching bullets from PlayerSummary objects.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict
from report.metrics_v3 import PlayerSummary


@dataclass
class CoachingInsight:
    player_id: int          # which player this insight targets (0 = both)
    category: str           # "speed", "reaction", "positioning", "shot", "readiness"
    severity: str           # "positive", "warning", "info"
    text: str
    delta: Optional[float] = None   # magnitude of difference if comparative


class CoachFeedbackEngine:
    """
    Generates per-player and cross-player coaching bullets.
    All comparisons are cross-sectional (player vs player this session).
    """

    def generate(self, summaries: Dict[int, PlayerSummary]) -> List[CoachingInsight]:
        insights: List[CoachingInsight] = []

        pids = sorted(summaries.keys())

        for pid in pids:
            ps = summaries[pid]
            insights += self._speed_bullets(pid, ps)
            insights += self._readiness_bullets(pid, ps)
            insights += self._zone_bullets(pid, ps)
            insights += self._reaction_bullets(pid, ps)
            insights += self._shot_bullets(pid, ps)

        # cross-player comparisons (need at least 2 players)
        if len(pids) >= 2:
            insights += self._cross_speed(summaries, pids)
            insights += self._cross_reaction(summaries, pids)
            insights += self._cross_readiness(summaries, pids)
            insights += self._cross_positioning(summaries, pids)

        return insights

    # ── per-player bullets ─────────────────────────────────────────────────────

    def _speed_bullets(self, pid: int, ps: PlayerSummary) -> List[CoachingInsight]:
        out = []
        if ps.max_speed_mps == 0:
            return out
        top_kmh = ps.max_speed_mps * 3.6
        mean_kmh = ps.mean_speed_mps * 3.6
        if top_kmh >= 20.0:
            out.append(CoachingInsight(pid, "speed", "positive",
                f"Player {pid} reached {top_kmh:.1f} km/h — excellent court coverage speed."))
        elif top_kmh < 12.0:
            out.append(CoachingInsight(pid, "speed", "warning",
                f"Player {pid} top speed was only {top_kmh:.1f} km/h — consider working on first-step explosiveness."))

        if mean_kmh < 2.0 and ps.samples > 100:
            out.append(CoachingInsight(pid, "speed", "info",
                f"Player {pid} average speed {mean_kmh:.1f} km/h — spends significant time stationary; improve court movement habits."))
        return out

    def _readiness_bullets(self, pid: int, ps: PlayerSummary) -> List[CoachingInsight]:
        out = []
        if ps.ready_time_pct is not None:
            if ps.ready_time_pct >= 70.0:
                out.append(CoachingInsight(pid, "readiness", "positive",
                    f"Player {pid} maintained ready position {ps.ready_time_pct:.0f}% of the time — excellent base positioning."))
            elif ps.ready_time_pct < 40.0:
                out.append(CoachingInsight(pid, "readiness", "warning",
                    f"Player {pid} was in ready stance only {ps.ready_time_pct:.0f}% of the time — focus on recovering to base after each shot."))

        if ps.knee_angle_mean_deg is not None:
            if ps.knee_angle_mean_deg > 165.0:
                out.append(CoachingInsight(pid, "readiness", "warning",
                    f"Player {pid} average knee angle {ps.knee_angle_mean_deg:.0f}° — legs too straight; lower the centre of gravity for faster reactions."))
            elif ps.knee_angle_mean_deg < 130.0:
                out.append(CoachingInsight(pid, "readiness", "positive",
                    f"Player {pid} maintains a deep athletic stance (avg knee {ps.knee_angle_mean_deg:.0f}°)."))

        if ps.low_knee_time_pct is not None and ps.low_knee_time_pct >= 50.0:
            out.append(CoachingInsight(pid, "readiness", "positive",
                f"Player {pid} spends {ps.low_knee_time_pct:.0f}% of rally time with bent knees (≤150°) — good athletic posture."))
        return out

    def _zone_bullets(self, pid: int, ps: PlayerSummary) -> List[CoachingInsight]:
        out = []
        if ps.role_profile == "Net-heavy":
            out.append(CoachingInsight(pid, "positioning", "info",
                f"Player {pid} role: Net-heavy — strong net presence. Ensure partner covers the back court."))
        elif ps.role_profile == "Back-heavy":
            out.append(CoachingInsight(pid, "positioning", "info",
                f"Player {pid} role: Back-heavy — anchors the baseline. Be ready to push forward when partner is pulled wide."))
        elif ps.role_profile == "Rotating":
            out.append(CoachingInsight(pid, "positioning", "info",
                f"Player {pid} role: Rotating — good all-court movement, balancing net and back court."))

        if ps.crossing_events_count > 5:
            out.append(CoachingInsight(pid, "positioning", "warning",
                f"Player {pid} crossed the net line {ps.crossing_events_count} times — check court coverage assignments with partner."))
        return out

    def _reaction_bullets(self, pid: int, ps: PlayerSummary) -> List[CoachingInsight]:
        out = []
        if ps.mean_reaction_time_s is None or ps.reaction_events == 0:
            return out
        rt = ps.mean_reaction_time_s
        if rt <= 0.25:
            out.append(CoachingInsight(pid, "reaction", "positive",
                f"Player {pid} average reaction time {rt*1000:.0f} ms — elite-level anticipation."))
        elif rt <= 0.45:
            out.append(CoachingInsight(pid, "reaction", "info",
                f"Player {pid} average reaction time {rt*1000:.0f} ms — solid, room to improve anticipation."))
        else:
            out.append(CoachingInsight(pid, "reaction", "warning",
                f"Player {pid} average reaction time {rt*1000:.0f} ms — work on reading shuttle direction earlier."))
        return out

    def _shot_bullets(self, pid: int, ps: PlayerSummary) -> List[CoachingInsight]:
        out = []
        if ps.shots_total == 0:
            return out

        breakdown = ps.shot_type_breakdown
        total = ps.shots_total
        smash_pct = 100.0 * breakdown.get("smash", 0) / max(total, 1)
        overhead_pct = 100.0 * breakdown.get("overhead_clear", 0) / max(total, 1)
        drive_pct = 100.0 * breakdown.get("drive", 0) / max(total, 1)
        net_pct = 100.0 * breakdown.get("net_drop", 0) / max(total, 1)

        if smash_pct >= 30.0:
            out.append(CoachingInsight(pid, "shot", "info",
                f"Player {pid} uses smashes {smash_pct:.0f}% of shots — aggressive style."))
        if net_pct >= 30.0:
            out.append(CoachingInsight(pid, "shot", "info",
                f"Player {pid} plays frequent net drops ({net_pct:.0f}%) — good net touch, ensure back court coverage."))

        if ps.mean_elbow_angle_R is not None:
            if ps.mean_elbow_angle_R < 100.0:
                out.append(CoachingInsight(pid, "shot", "warning",
                    f"Player {pid} average elbow angle at contact {ps.mean_elbow_angle_R:.0f}° — arm not fully extending; may reduce power."))
            elif ps.mean_elbow_angle_R > 160.0:
                out.append(CoachingInsight(pid, "shot", "positive",
                    f"Player {pid} strong arm extension at contact ({ps.mean_elbow_angle_R:.0f}°) — maximising power transfer."))

        if ps.mean_hip_rotation is not None:
            if ps.mean_hip_rotation < 10.0:
                out.append(CoachingInsight(pid, "shot", "warning",
                    f"Player {pid} minimal hip rotation at contact ({ps.mean_hip_rotation:.0f}°) — add body rotation for more power."))
            elif ps.mean_hip_rotation >= 25.0:
                out.append(CoachingInsight(pid, "shot", "positive",
                    f"Player {pid} good hip rotation ({ps.mean_hip_rotation:.0f}°) — efficient power transfer through the body."))
        return out

    # ── cross-player comparisons ───────────────────────────────────────────────

    def _cross_speed(self, summaries: Dict[int, PlayerSummary],
                     pids: List[int]) -> List[CoachingInsight]:
        out = []
        # compare pairwise within a team (1v2, 3v4) and across teams
        pairs = [(1, 2), (3, 4)] if all(p in summaries for p in [1, 2, 3, 4]) else \
                [(pids[0], pids[1])]
        for a, b in pairs:
            if a not in summaries or b not in summaries:
                continue
            diff = summaries[a].mean_speed_mps - summaries[b].mean_speed_mps
            if abs(diff) > 0.5:
                faster, slower = (a, b) if diff > 0 else (b, a)
                out.append(CoachingInsight(0, "speed", "info",
                    f"Player {faster} covers more court (avg {abs(diff)*3.6:.1f} km/h faster than Player {slower}) — "
                    f"Player {slower} should focus on lateral movement drills.",
                    delta=abs(diff)))
        return out

    def _cross_reaction(self, summaries: Dict[int, PlayerSummary],
                        pids: List[int]) -> List[CoachingInsight]:
        out = []
        valid = [(pid, summaries[pid].mean_reaction_time_s)
                 for pid in pids if summaries[pid].mean_reaction_time_s is not None]
        if len(valid) < 2:
            return out
        valid.sort(key=lambda x: x[1])
        fastest_pid, fastest_rt = valid[0]
        slowest_pid, slowest_rt = valid[-1]
        delta = slowest_rt - fastest_rt
        if delta > 0.08:
            out.append(CoachingInsight(0, "reaction", "info",
                f"Player {fastest_pid} reacts {delta*1000:.0f} ms faster than Player {slowest_pid} on average — "
                f"Player {slowest_pid} should work on anticipation and shuttle reading.",
                delta=delta))
        return out

    def _cross_readiness(self, summaries: Dict[int, PlayerSummary],
                         pids: List[int]) -> List[CoachingInsight]:
        out = []
        valid = [(pid, summaries[pid].ready_time_pct)
                 for pid in pids if summaries[pid].ready_time_pct is not None]
        if len(valid) < 2:
            return out
        valid.sort(key=lambda x: x[1], reverse=True)
        best_pid, best_pct = valid[0]
        worst_pid, worst_pct = valid[-1]
        delta = best_pct - worst_pct
        if delta > 15.0:
            out.append(CoachingInsight(0, "readiness", "info",
                f"Player {best_pid} is in ready stance {delta:.0f}% more often than Player {worst_pid} — "
                f"Player {worst_pid} should focus on faster recovery to base position.",
                delta=delta))
        return out

    def _cross_positioning(self, summaries: Dict[int, PlayerSummary],
                           pids: List[int]) -> List[CoachingInsight]:
        out = []
        # Compare team spreads within each team
        far_players = [p for p in [1, 2] if p in summaries and summaries[p].mean_team_spread_m is not None]
        near_players = [p for p in [3, 4] if p in summaries and summaries[p].mean_team_spread_m is not None]

        for team_pids in [far_players, near_players]:
            if not team_pids:
                continue
            # mean_team_spread_m is the same value for both teammates; just report once
            spread = summaries[team_pids[0]].mean_team_spread_m
            if spread is None:
                continue
            team_label = "Far" if team_pids[0] in [1, 2] else "Near"
            if spread < 2.0:
                out.append(CoachingInsight(0, "positioning", "warning",
                    f"{team_label} team average spread {spread:.1f} m — players are bunching up; spread wider to cover more court."))
            elif spread > 5.0:
                out.append(CoachingInsight(0, "positioning", "warning",
                    f"{team_label} team average spread {spread:.1f} m — players are very spread out; tighten positioning to reduce gaps."))
            else:
                out.append(CoachingInsight(0, "positioning", "positive",
                    f"{team_label} team average spread {spread:.1f} m — good court coverage width."))
        return out

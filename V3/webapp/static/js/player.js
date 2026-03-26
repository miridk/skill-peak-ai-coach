/* player.js — Video event overlay system, ported from coach_player.html */

window.SP = window.SP || {};

/**
 * Attach coaching event overlays to the video element.
 *
 * Events are pause-and-show coaching tips at specific timestamps.
 * The events array comes from /api/session/<id>/events.
 *
 * @param {string} videoId   — id of the <video> element
 * @param {Array}  events    — from /api/session/<id>/events
 */
SP.initVideoEvents = function(videoId, events) {
  const video = document.getElementById(videoId);
  if (!video) return;

  // Filter to shot events with timestamps only
  const shotEvents = events
    .filter(e => e.type === "shot" && e.timestamp_s != null && !isNaN(e.timestamp_s))
    .sort((a, b) => a.timestamp_s - b.timestamp_s);

  if (!shotEvents.length) return;

  // Build pause-point array (deduplicated to nearest 0.1s)
  const pauseTimes = [];
  let lastT = -999;
  shotEvents.forEach(ev => {
    if (ev.timestamp_s - lastT >= 0.5) {
      pauseTimes.push({ t: ev.timestamp_s, ev });
      lastT = ev.timestamp_s;
    }
  });

  // Overlay element
  const overlay = _createOverlay(video);
  let currentPauseIdx = 0;
  let paused_by_us = false;

  video.addEventListener("timeupdate", () => {
    if (paused_by_us) return;
    while (
      currentPauseIdx < pauseTimes.length &&
      video.currentTime >= pauseTimes[currentPauseIdx].t
    ) {
      const { ev } = pauseTimes[currentPauseIdx];
      video.pause();
      paused_by_us = true;
      _showOverlay(overlay, ev);
      currentPauseIdx++;
      break;
    }
  });

  video.addEventListener("play", () => {
    if (paused_by_us) {
      paused_by_us = false;
      _hideOverlay(overlay);
    }
  });

  video.addEventListener("seeking", () => {
    // Reset index to match new position
    paused_by_us = false;
    _hideOverlay(overlay);
    currentPauseIdx = pauseTimes.findIndex(p => p.t > video.currentTime);
    if (currentPauseIdx < 0) currentPauseIdx = pauseTimes.length;
  });
};

function _createOverlay(video) {
  const parent = video.parentElement;
  parent.style.position = "relative";

  const div = document.createElement("div");
  div.style.cssText = [
    "position:absolute", "bottom:56px", "left:50%", "transform:translateX(-50%)",
    "background:rgba(0,0,0,0.85)", "color:#e0e0e0",
    "padding:10px 18px", "border-radius:8px",
    "font-size:0.9em", "max-width:420px", "text-align:center",
    "pointer-events:none", "display:none",
    "border:1px solid #444",
  ].join(";");
  parent.appendChild(div);
  return div;
}

function _showOverlay(div, ev) {
  const typeLabel = {
    smash: "Smash",
    overhead_clear: "Overhead Clear",
    drive: "Drive",
    net_drop: "Net Drop",
    unknown: "Shot",
  }[ev.shot_type] || "Shot";

  const colour = SP.PLAYER_COLOURS[ev.player_id] || "#fff";
  div.innerHTML = `
    <span style="color:${colour};font-weight:600">Player ${ev.player_id}</span>
    — <strong>${typeLabel}</strong>
    <br><small style="color:#9e9e9e">
      ${ev.shuttle_x_m != null ? 'at (' + ev.shuttle_x_m.toFixed(1) + 'm, ' + ev.shuttle_y_m.toFixed(1) + 'm)' : ''}
      &nbsp;Press play to continue
    </small>`;
  div.style.display = "block";
}

function _hideOverlay(div) {
  div.style.display = "none";
}

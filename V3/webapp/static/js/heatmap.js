/* heatmap.js — Live court minimap rendered on a canvas element */

window.SP = window.SP || {};

const COURT_W = 6.10;
const COURT_L = 13.40;

/**
 * Initialise the live minimap.
 * Loads all players' skeleton position series, then on each video timeupdate
 * redraws the minimap showing player positions at the current timestamp.
 *
 * @param {string} canvasId
 * @param {string} sessionId
 */
SP.initMinimap = async function(canvasId, sessionId) {
  const canvas = document.getElementById(canvasId);
  if (!canvas) return;
  const ctx = canvas.getContext("2d");

  // Load skeleton position data for all players
  const playerData = {};  // pid -> [{timestamp_s, x_m, y_m}]
  await Promise.all([1, 2, 3, 4].map(async pid => {
    const data = await fetch(`/api/session/${sessionId}/skeleton/${pid}`)
      .then(r => r.json()).catch(() => []);
    if (data.length) playerData[pid] = data;
  }));

  const W = canvas.width;
  const H = canvas.height;

  function mToCanvas(xm, ym) {
    return [
      (xm / COURT_W) * W,
      (ym / COURT_L) * H,
    ];
  }

  function drawCourt() {
    ctx.clearRect(0, 0, W, H);
    ctx.fillStyle = "#1e1e1e";
    ctx.fillRect(0, 0, W, H);

    ctx.strokeStyle = "#444";
    ctx.lineWidth = 1.5;
    ctx.strokeRect(2, 2, W - 4, H - 4);

    // Net line
    ctx.beginPath();
    ctx.moveTo(2, H / 2);
    ctx.lineTo(W - 2, H / 2);
    ctx.stroke();

    // Service lines at ~1.98m from net (≈ H/2 ± 1.98/13.4 * H)
    const sOff = (1.98 / COURT_L) * H;
    [H / 2 - sOff, H / 2 + sOff].forEach(y => {
      ctx.beginPath();
      ctx.moveTo(2, y); ctx.lineTo(W - 2, y);
      ctx.strokeStyle = "#2a2a2a"; ctx.stroke();
    });
    ctx.strokeStyle = "#444";
  }

  function getPositionAt(data, ts) {
    if (!data || !data.length) return null;
    // Binary-search closest timestamp
    let lo = 0, hi = data.length - 1;
    while (lo < hi) {
      const mid = (lo + hi) >> 1;
      if (data[mid].timestamp_s < ts) lo = mid + 1; else hi = mid;
    }
    return data[lo];
  }

  function drawPlayers(ts) {
    drawCourt();
    [1, 2, 3, 4].forEach(pid => {
      const d = playerData[pid];
      if (!d) return;
      const pos = getPositionAt(d, ts);
      if (!pos || pos.x_m == null) return;

      const [cx, cy] = mToCanvas(pos.x_m, pos.y_m);
      const colour = SP.PLAYER_COLOURS[pid] || "#fff";

      ctx.beginPath();
      ctx.arc(cx, cy, 7, 0, Math.PI * 2);
      ctx.fillStyle = colour + "cc";
      ctx.fill();
      ctx.strokeStyle = colour;
      ctx.lineWidth = 1.5;
      ctx.stroke();

      ctx.fillStyle = "#000";
      ctx.font = "bold 9px system-ui";
      ctx.textAlign = "center";
      ctx.textBaseline = "middle";
      ctx.fillText(String(pid), cx, cy);
    });
  }

  // Initial static render
  drawCourt();

  // Hook into the video element if present
  const video = document.getElementById("session-video");
  if (video) {
    video.addEventListener("timeupdate", () => drawPlayers(video.currentTime));
    video.addEventListener("seeked",     () => drawPlayers(video.currentTime));
  }

  // Fallback: if no video, just draw at t=0
  drawPlayers(0);
};

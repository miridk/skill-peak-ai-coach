/* charts.js — Chart.js wrappers for SkillPeak V3 */

window.SP = window.SP || {};

const SHOT_LABELS = ["smash", "overhead_clear", "drive", "net_drop", "unknown"];

/**
 * Build or update the speed timeline chart.
 * @param {string} canvasId
 * @param {string} sessionId
 * @param {number|null} filterPid  — if set, only show this player
 */
SP.buildSpeedChart = async function(canvasId, sessionId, filterPid = null) {
  const PIDS = filterPid ? [filterPid] : [1, 2, 3, 4];
  const ctx = document.getElementById(canvasId);
  if (!ctx) return;

  const datasets = [];

  for (const pid of PIDS) {
    const data = await fetch(`/api/session/${sessionId}/skeleton/${pid}`)
      .then(r => r.json()).catch(() => []);
    if (!data.length) continue;

    const step = Math.max(1, Math.floor(data.length / 600));
    const pts = data
      .filter((_, i) => i % step === 0)
      .map(r => ({ x: r.timestamp_s, y: r.speed_kmh }));

    datasets.push({
      label: "Player " + pid,
      data: pts,
      borderColor: SP.PLAYER_COLOURS[pid] || "#fff",
      backgroundColor: "transparent",
      borderWidth: 1.5,
      pointRadius: 0,
      tension: 0.3,
    });
  }

  new Chart(ctx, {
    type: "line",
    data: { datasets },
    options: {
      animation: false,
      responsive: true,
      maintainAspectRatio: false,
      parsing: false,
      scales: {
        x: {
          type: "linear",
          title: { display: true, text: "Time (s)", color: "#9e9e9e" },
          ticks: { color: "#9e9e9e" },
        },
        y: {
          min: 0,
          title: { display: true, text: "Speed (km/h)", color: "#9e9e9e" },
          ticks: { color: "#9e9e9e" },
        },
      },
      plugins: {
        legend: { labels: { color: "#e0e0e0" } },
        tooltip: {
          callbacks: {
            label: item => `Player ${item.dataset.label.split(" ")[1]}: ${item.parsed.y.toFixed(1)} km/h`,
          },
        },
      },
    },
  });
};

/**
 * Build the shot-type bar chart.
 * @param {string} canvasId
 * @param {object} summaries  — {pid: {shot_type_breakdown: {...}}}
 */
SP.buildShotChart = function(canvasId, summaries) {
  const ctx = document.getElementById(canvasId);
  if (!ctx) return;

  const datasets = Object.entries(summaries).map(([pid, ps]) => {
    const breakdown = ps.shot_type_breakdown || {};
    return {
      label: "Player " + pid,
      data: SHOT_LABELS.map(l => breakdown[l] || 0),
      backgroundColor: SP.PLAYER_COLOURS[parseInt(pid)] || "#fff",
    };
  });

  new Chart(ctx, {
    type: "bar",
    data: { labels: SHOT_LABELS, datasets },
    options: {
      animation: false,
      responsive: true,
      maintainAspectRatio: false,
      scales: {
        x: { ticks: { color: "#9e9e9e" } },
        y: {
          ticks: { color: "#9e9e9e" },
          title: { display: true, text: "Count", color: "#9e9e9e" },
        },
      },
      plugins: { legend: { labels: { color: "#e0e0e0" } } },
    },
  });
};

/**
 * Build per-player shot radar charts on the session page.
 * @param {object} summaries
 */
SP.buildRadarCharts = function(summaries) {
  Object.entries(summaries).forEach(([pidStr, ps]) => {
    const pid = parseInt(pidStr);
    const el = document.getElementById("radar-" + pid);
    if (!el) return;
    const breakdown = ps.shot_type_breakdown || {};
    const total = SHOT_LABELS.reduce((a, l) => a + (breakdown[l] || 0), 0) || 1;
    const data = SHOT_LABELS.map(l => (((breakdown[l] || 0) / total) * 100).toFixed(1));

    new Chart(el.getContext("2d"), {
      type: "radar",
      data: {
        labels: SHOT_LABELS,
        datasets: [{
          label: "Player " + pid,
          data,
          borderColor: SP.PLAYER_COLOURS[pid] || "#fff",
          backgroundColor: (SP.PLAYER_COLOURS[pid] || "#fff") + "33",
        }],
      },
      options: {
        animation: false,
        responsive: false,
        scales: {
          r: {
            ticks: { color: "#9e9e9e", backdropColor: "transparent" },
            grid: { color: "#333" },
            pointLabels: { color: "#9e9e9e", font: { size: 10 } },
          },
        },
        plugins: { legend: { display: false } },
      },
    });
  });
};

/**
 * Build a single player shot radar on the player detail page.
 * @param {string} canvasId
 * @param {number} pid
 * @param {object} breakdown  — {shot_type: count}
 */
SP.buildSingleRadar = function(canvasId, pid, breakdown) {
  const el = document.getElementById(canvasId);
  if (!el) return;
  const total = SHOT_LABELS.reduce((a, l) => a + (breakdown[l] || 0), 0) || 1;
  const data = SHOT_LABELS.map(l => (((breakdown[l] || 0) / total) * 100).toFixed(1));

  new Chart(el.getContext("2d"), {
    type: "radar",
    data: {
      labels: SHOT_LABELS,
      datasets: [{
        label: "Player " + pid,
        data,
        borderColor: SP.PLAYER_COLOURS[pid] || "#fff",
        backgroundColor: (SP.PLAYER_COLOURS[pid] || "#fff") + "33",
      }],
    },
    options: {
      animation: false,
      responsive: false,
      scales: {
        r: {
          ticks: { color: "#9e9e9e", backdropColor: "transparent" },
          grid: { color: "#333" },
          pointLabels: { color: "#9e9e9e", font: { size: 11 } },
        },
      },
      plugins: { legend: { display: false } },
    },
  });
};

/**
 * Render coaching insights into a <ul> element.
 * @param {Array} insights
 */
SP.renderInsights = function(insights) {
  const sevColour = { positive: "#69f0ae", warning: "#ff8a65", info: "#90caf9" };
  const sevIcon   = { positive: "✓", warning: "⚠", info: "ℹ" };

  const globalUl = document.getElementById("global-insights");

  insights.forEach(ins => {
    const c = sevColour[ins.severity] || "#90caf9";
    const i = sevIcon[ins.severity] || "•";
    const li = document.createElement("li");
    li.style.cssText = `border-left:3px solid ${c};padding-left:8px;margin:4px 0`;
    li.textContent = i + " " + ins.text;

    if (ins.player_id === 0 && globalUl) {
      globalUl.appendChild(li);
    } else {
      const ul = document.getElementById("insights-" + ins.player_id);
      if (ul) ul.appendChild(li.cloneNode(true));
    }
  });
};

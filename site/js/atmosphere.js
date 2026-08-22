// Snow color ramps, level lists and AGL ladders.

// --- snow depth colormap ----------------------------------------------------
// Transparent -> white (a dusting) -> pale blue -> deep blue (metres of snow).
// Alpha ramps in fast so even a couple of centimetres reads on the map.
export const DEPTH_MAX = 2.0; // metres at the top of the ramp

const DEPTH_STOPS = [
  // t, [r, g, b], alpha
  [0.0, [255, 255, 255], 0.0],
  [0.02, [255, 255, 255], 0.55],
  [0.1, [244, 249, 255], 0.78],
  [0.25, [219, 233, 248], 0.88],
  [0.5, [156, 195, 232], 0.92],
  [0.75, [107, 159, 216], 0.95],
  [1.0, [69, 119, 184], 0.97],
];

export function depthColor(t) {
  t = Math.min(Math.max(t, 0), 1);
  for (let i = 1; i < DEPTH_STOPS.length; i++) {
    if (t <= DEPTH_STOPS[i][0]) {
      const [t0, c0, a0] = DEPTH_STOPS[i - 1];
      const [t1, c1, a1] = DEPTH_STOPS[i];
      const f = (t - t0) / (t1 - t0);
      return [
        ...c0.map((c, k) => Math.round(c + (c1[k] - c) * f)),
        a0 + (a1 - a0) * f,
      ];
    }
  }
  const last = DEPTH_STOPS[DEPTH_STOPS.length - 1];
  return [...last[1], last[2]];
}

// 256x1 RGBA ramp texture; alpha in the texture so the cover shader gets the
// full colormap (including where snow fades to nothing) in one sample.
export function depthRampData(n = 256) {
  const data = new Uint8Array(n * 4);
  for (let i = 0; i < n; i++) {
    const [r, g, b, a] = depthColor(i / (n - 1));
    data.set([r, g, b, Math.round(a * 255)], i * 4);
  }
  return data;
}

// --- ladders ----------------------------------------------------------------
// Heights above ground (metres) the snowflakes actually fall through. Tight
// near the surface where the drift physics lives, reaching up to cloud base
// so flakes have real airtime before they land.
export const SNOW_LADDER = [0.1, 50, 150, 300, 600, 1000, 1600, 2400, 3000];

// Saltation layer for blowing-snow streaks: the bottom few metres only.
export const DRIFT_LADDER = [0.1, 0.6, 1.5, 3.0];

// Model levels the ladders interpolate from: everything the pipeline ships.
// (12 levels, surface to 700 hPa — exactly MAX_STACK.)
export const LADDER_SOURCE_IDS = [
  "10m", "80m", "1000", "975", "950", "925", "900", "875", "850", "800", "750", "700",
];

export function ladderSourceIndices(meta) {
  return LADDER_SOURCE_IDS
    .map((id) => meta.levels.find((l) => l.id === id))
    .filter(Boolean)
    .map((l) => l.index);
}

export function levelName(level) {
  if (level.kind === "height_agl") return `${level.value} m (surface)`;
  const km = (level.heightMeters / 1000).toFixed(1);
  return `${level.value} hPa (~${km} km)`;
}

// --- legend -----------------------------------------------------------------
export function drawDepthLegend(canvas) {
  const ctx = canvas.getContext("2d");
  const { width: w } = canvas;
  ctx.clearRect(0, 0, w, canvas.height);
  // checkerboard behind the ramp so the transparent end reads as transparent
  ctx.fillStyle = "#2a3242";
  ctx.fillRect(0, 0, w, 14);
  for (let x = 0; x < w; x += 8) {
    ctx.fillStyle = (x / 8) % 2 ? "#242b39" : "#2a3242";
    ctx.fillRect(x, 0, 8, 14);
  }
  for (let x = 0; x < w; x++) {
    const [r, g, b, a] = depthColor(x / (w - 1));
    ctx.fillStyle = `rgba(${r},${g},${b},${a})`;
    ctx.fillRect(x, 0, 1, 14);
  }
  ctx.fillStyle = "#a8b6ca";
  ctx.font = "9px sans-serif";
  for (const m of [0, 0.5, 1, 1.5, 2]) {
    const x = Math.min((m / DEPTH_MAX) * w, w - 18);
    ctx.fillText(m === 0 ? "0" : `${m}`, x, 24);
  }
  ctx.fillText("snow depth (m, drifted)", w - 108, 33);
}

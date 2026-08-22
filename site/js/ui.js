// DOM control wiring.

import { drawDepthLegend } from "./atmosphere.js";

export function initUI(map, layer, meta, setBasemap, stormOpts = {}) {
  const $ = (id) => document.getElementById(id);

  // --- header ---
  const init = new Date(meta.init_time);
  $("init-time").textContent = `init ${init.toISOString().slice(0, 13)}Z`;

  // --- historical storm picker ---
  const stormSel = $("storm-select");
  for (const s of stormOpts.storms ?? []) {
    const opt = document.createElement("option");
    opt.value = s.id;
    opt.textContent = s.label ?? s.id;
    stormSel.appendChild(opt);
  }
  stormSel.value = stormOpts.storm ?? "";
  stormSel.addEventListener("change", () => {
    // Switching datasets swaps every texture and the sim state: a clean
    // reload is simpler and safer than hot-swapping the FrameManager.
    const url = new URL(location.href);
    if (stormSel.value) url.searchParams.set("storm", stormSel.value);
    else url.searchParams.delete("storm");
    location.href = url.toString();
  });

  // --- layer toggles ---
  const bindToggle = (id, set) => {
    const el = $(id);
    el.addEventListener("change", () => set(el.checked));
    return el;
  };
  bindToggle("show-wind", (v) => { layer.windOn = v; }).checked = layer.windOn;
  bindToggle("show-flakes", (v) => { layer.flakesOn = v; }).checked = layer.flakesOn;
  bindToggle("show-streaks", (v) => { layer.streaksOn = v; }).checked = layer.streaksOn;
  bindToggle("show-cover", (v) => { layer.coverOn = v; }).checked = layer.coverOn;
  bindToggle("model-depth", (v) => { layer.modelDepthMode = v; }).checked = layer.modelDepthMode;

  // --- basemap ---
  const bm = $("basemap-select");
  const pistes = $("show-pistes");
  const applyBasemap = () => setBasemap(map, bm.value, pistes.checked);
  bm.addEventListener("change", applyBasemap);
  pistes.addEventListener("change", applyBasemap);

  // --- drift intensity ---
  const dg = $("drift-gain");
  dg.value = String(layer.qGain);
  dg.addEventListener("input", () => {
    layer.qGain = Number(dg.value);
    layer.dispGain = Math.min(Number(dg.value), 2);
    $("drift-val").textContent = Number(dg.value).toFixed(1);
  });

  // --- fresh snow density ---
  const ds = $("density-scale");
  ds.addEventListener("input", () => {
    layer.densityScale = Number(ds.value);
    $("density-val").textContent = `${Number(ds.value).toFixed(1)}×`;
  });

  // --- storm strength (simulation) ---
  const wg = $("wind-gain");
  const wgVal = $("wind-val");
  const wgHint = $("wind-hint");
  wg.addEventListener("input", () => {
    const v = Number(wg.value);
    layer.windGain = v;
    if (v === 1) {
      wgVal.textContent = "forecast";
      wgHint.textContent = "Scales the real wind so drift effects are easier to see. Not a forecast.";
      wgHint.classList.remove("sim");
    } else {
      wgVal.textContent = `${v.toFixed(1)}× simulated`;
      wgHint.textContent = `Simulated: showing ${v.toFixed(1)}× the forecast wind, not real conditions.`;
      wgHint.classList.add("sim");
    }
  });

  // --- particle count ---
  const pc = $("particles-select");
  pc.value = String(layer.flakeCount);
  pc.addEventListener("change", () => layer.setFlakeCount(Number(pc.value)));

  // --- terrain exaggeration ---
  const ex = $("exaggeration");
  ex.addEventListener("input", () => {
    const v = Number(ex.value);
    $("exag-val").textContent = v.toFixed(1);
    layer.exaggeration = v;
    layer.altScale = v || 1;
    map.setTerrain(v > 0 ? { source: "terrain-dem", exaggeration: v } : null);
  });

  // --- flake speed ---
  const sf = $("speed-factor");
  sf.addEventListener("input", () => {
    layer.speedFactor = Number(sf.value);
    $("speed-val").textContent = Number(sf.value).toFixed(1);
  });

  // --- restart storm ---
  $("restart-storm").addEventListener("click", () => {
    layer.resetSim();
    setTime(0);
    if (!playing) togglePlay();
  });

  // --- panel collapse ---
  const panel = document.getElementById("panel");
  $("panel-toggle").addEventListener("click", () => panel.classList.toggle("collapsed"));
  if (window.innerWidth < 700) panel.classList.add("collapsed");

  // --- time slider + playback ---
  const slider = $("time-slider");
  const label = $("time-label");
  const playBtn = $("play");
  const catchupEl = $("catchup");
  const maxLead = meta.frames[meta.frames.length - 1].lead_hours;
  slider.max = String(maxLead);
  let playing = false;
  let lastTs = 0;
  const HOURS_PER_SEC = 0.7;

  function setTime(t, fromSlider = false) {
    layer.time = Math.max(0, Math.min(maxLead, t));
    if (!fromSlider) slider.value = String(Math.round(layer.time));
    const valid = new Date(init.getTime() + layer.time * 3600e3);
    const opts = { month: "short", day: "numeric", hour: "numeric" };
    label.textContent = `+${Math.round(layer.time)} h · ${valid.toLocaleString(undefined, opts)}`;
  }

  slider.addEventListener("input", () => {
    pause();
    setTime(Number(slider.value), true);
  });

  function tick(ts) {
    if (!playing) return;
    const dt = lastTs ? (ts - lastTs) / 1000 : 0;
    lastTs = ts;
    // Don't outrun the accumulation sim while it is catching up.
    let t = layer.time + (layer.catchingUp ? 0 : dt * HOURS_PER_SEC);
    if (t > maxLead) t = 0; // loop (the layer resets + replays automatically)
    setTime(t);
    catchupEl.hidden = !layer.catchingUp;
    requestAnimationFrame(tick);
  }
  function pause() {
    playing = false;
    playBtn.textContent = "▶";
    catchupEl.hidden = true;
  }
  function togglePlay() {
    playing = !playing;
    playBtn.textContent = playing ? "❚❚" : "▶";
    if (playing) {
      lastTs = 0;
      requestAnimationFrame(tick);
    } else {
      catchupEl.hidden = true;
    }
  }
  playBtn.addEventListener("click", togglePlay);

  setTime(0);
  // The whole point is watching the storm build: play from the start.
  togglePlay();
  drawDepthLegend($("legend"));
}

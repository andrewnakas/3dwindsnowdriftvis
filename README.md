# 3D Snow Drift Viewer

Browser-based 3D visualization of **snow drifting and accumulation** over the
CONUS, driven by the NOAA HRRR 48-hour forecast. Sister project of
[3dWindViewer](https://github.com/andrewnakas/3dWindViewer) — it reuses that
project's HRRR pipeline and GPU wind engine, and adds a snow physics layer on
top.

Play the time slider and the storm builds in front of you: snowflakes fall out
of the forecast's precipitation field, land, and the wind immediately starts
working the fresh snow — scouring exposed windward slopes and ridge crests,
and loading lee slopes, gullies and sheltered bowls with drifts.

## What it shows

- **Falling snow** — GPU snowflake particles spawn with density proportional
  to the local HRRR snowfall rate, ride the 3D wind field (12 levels, surface
  to 700 hPa), fall at a terminal velocity set by flake type (dendrites ~0.8
  m/s; wet or mixed-phase precipitation faster), and land on the terrain.
- **Blowing snow** — surface-hugging saltation streaks that only appear where
  snow lies on the ground *and* the 10 m wind beats the transport threshold.
- **Drifted snow cover** — a persistent accumulation simulation on a
  1350×795 grid. Each step:
  1. accumulates model snowfall, converting SWE to depth with a
     temperature-dependent fresh-snow density (LaChapelle / Hedstrom–Pomeroy:
     ~50 kg/m³ in deep cold, ~250 kg/m³ near freezing);
  2. redistributes it with a mass-conserving donor-cell drift pass: transport
     capacity goes with u*³ excess over an age/density-dependent threshold
     (Pomeroy & Gray), amplified on windward slopes and convex crests
     (Liston–Elder) and collapsed in Winstral-Sx sheltered lee zones —
     deposition emerges wherever capacity decreases downwind, which is
     SnowTran-3D's core mechanism;
  3. melts snow where the 2 m temperature is above freezing.

  At render time the coarse sim depth is modulated by a drift-susceptibility
  factor computed against the ~1.4 km terrain texture, so drifts read at 3×
  the sim resolution.
- **Model depth mode** — HRRR's own snow-depth field, no drift, for A/B
  comparison.

Scrubbing backward (or the loop wrapping around) clears the pack and fast-
forwards from hour 0 — a deliberate storm time-lapse, since drift is
path-dependent and cannot be evaluated at an instant.

## Basemap

Winter-styled by default: [OpenTopoMap](https://opentopomap.org) shaded topo
with the [OpenSnowMap](https://www.opensnowmap.org) ski-piste overlay
(toggleable), Esri World Imagery as the alternate base, and 3D terrain from
AWS Terrain Tiles — all on MapLibre GL.

## Architecture

```
pipeline/   Python: dynamical.org HRRR (icechunk/zarr) -> PNG texture atlases
  config.py         single source of truth (grid, levels, atlas layout, scales)
  build_frames.py   per-lead wind levels + snow scalars, cumulative SWE
  encode.py         8-bit quantization, meta.json contract
  make_test_data.py synthetic ridge-field storm for offline dev (no network)
site/       no build step: CDN MapLibre + ES modules + raw WebGL2
  js/snowSystem.js  custom layer: flakes + streaks + cover sim + draped mesh
  js/snowShaders.js GLSL for all three subsystems
  js/shaders.js     wind sampling + terrain physics (from 3dWindViewer)
```

Each frame atlas (2250×795) carries 12 wind level tiles (R=u, G=v, B=omega,
A=valid), a 16-bit terrain tile, and two snow scalar tiles: snowfall this
hour / cumulative SWE / 2 m temperature, and model depth / percent frozen /
precipitation rate. Hourly snowfall SWE is derived as precipitation rate ×
frozen fraction. A GitHub Action rebuilds `site/data/` four times a day and
deploys the whole site to GitHub Pages; no data is committed.

## Development

```bash
# offline: synthetic westerly storm crossing ridge lines near Bozeman
pip install numpy pillow scipy pyproj
PYTHONPATH=pipeline python pipeline/make_test_data.py --out site/data

# real data (needs network + full requirements.txt)
pip install -r requirements.txt
PYTHONPATH=pipeline python pipeline/build_frames.py --out site/data --leads "0 3 6" --workers 2

cd site && python -m http.server 8123   # -> http://localhost:8123
```

`node tools/screenshot.mjs` drives the site headlessly and captures
screenshots (requires playwright). Debug query flags: `?drift=0`, `?flakes=0`,
`?cover=0`, `?tp=0` (terrain physics off). `window.__snow.debugReadCover()`
reads back the accumulation state for sanity checks.

With the synthetic storm, a steady westerly must scour the west faces and
load the east (lee) faces — the included physics check measures exactly that
(lee ≈ 0.28 m vs windward ≈ 0.17 m after 8 simulated hours).

## Credits

- Forecast data: [NOAA HRRR via dynamical.org](https://dynamical.org/catalog/noaa-hrrr-forecast-48-hour-virtual/)
- Basemap © [OpenTopoMap](https://opentopomap.org) (CC-BY-SA), © OpenStreetMap contributors, SRTM
- Ski runs © [OpenSnowMap.org](https://www.opensnowmap.org)
- Imagery © Esri, Maxar, Earthstar Geographics
- Terrain tiles: AWS Terrain Tiles / Mapzen (terrarium)
- Drift physics after Liston & Sturm (SnowTran-3D), Liston & Elder (MicroMet),
  Pomeroy & Gray (saltation), Winstral et al. (Sx shelter index)

MIT License.

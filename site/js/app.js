// Bootstrap: map + winter basemaps + terrain + snow system + UI.

import { SnowSystem } from "./snowSystem.js";
import { SOURCES, BASE_LAYERS, LABEL_LAYER, setBasemap } from "./basemaps.js";
import { initUI } from "./ui.js";

// The `?c=1` is deliberate. S3 only attaches Access-Control-Allow-Origin when
// the request carries an Origin header, so a cached non-CORS copy of the same
// URL — fetched earlier by anything else — gets reused for the CORS request and
// Safari blocks it, leaving the map with no terrain at all. A distinct query
// string keeps the DEM in its own cache entry only ever populated via CORS.
const TERRAIN_TILES = "https://s3.amazonaws.com/elevation-tiles-prod/terrarium/{z}/{x}/{y}.png?c=1";

function fail(msg) {
  const el = document.getElementById("error");
  el.textContent = msg;
  el.hidden = false;
}

async function main() {
  // ?storm=<id> loads a prebuilt historical event from data/storms/<id>/
  // instead of the live forecast. The picker options come from the storm
  // index the deploy assembles; missing index just means no storms shipped.
  const params = new URLSearchParams(location.search);
  const storm = (params.get("storm") ?? "").replace(/[^a-z0-9-]/gi, "");
  const basePath = storm ? `data/storms/${storm}/` : "data/";

  let storms = [];
  try {
    const r = await fetch(`data/storms/index.json?t=${Date.now()}`);
    if (r.ok) storms = await r.json();
  } catch { /* no historical storms shipped */ }

  let meta;
  try {
    const r = await fetch(`${basePath}meta.json?t=${Date.now()}`); // always-fresh meta
    if (!r.ok) throw new Error(`HTTP ${r.status}`);
    meta = await r.json();
  } catch (e) {
    fail(storm
      ? `Storm "${storm}" is not available on this deployment.`
      : "Snow data not available yet (data/meta.json missing). " +
        "The GitHub Action may still be running its first build.");
    throw e;
  }

  const isMobile = /Mobi|Android|iPhone|iPad/i.test(navigator.userAgent) || window.innerWidth < 700;
  let map;
  try {
    map = new maplibregl.Map({
      container: "map",
      style: {
        version: 8,
        sources: SOURCES,
        layers: BASE_LAYERS,
      },
      center: [-111.04, 45.68], // Bozeman, MT
      zoom: isMobile ? 8.8 : 9.6,
      pitch: 65,
      bearing: -15,
      maxPitch: 80,
      antialias: !isMobile, // MSAA is heavy on mobile GPUs
    });
  } catch (e) {
    fail(`Couldn't start the map (WebGL2 unavailable?): ${e.message}`);
    throw e;
  }
  map.addControl(new maplibregl.NavigationControl({ visualizePitch: true }), "top-right");
  map.addControl(
    new maplibregl.GeolocateControl({
      positionOptions: { enableHighAccuracy: true },
      trackUserLocation: true,
      showUserLocation: true,
      fitBoundsOptions: { maxZoom: 10, pitch: 65 },
    }),
    "top-right"
  );
  window.__map = map; // debugging hook

  map.on("load", () => {
    map.addSource("terrain-dem", {
      type: "raster-dem",
      tiles: [TERRAIN_TILES],
      tileSize: 256,
      maxzoom: 12,
      encoding: "terrarium",
      attribution: "Terrain: AWS Terrain Tiles / Mapzen",
    });
    map.setTerrain({ source: "terrain-dem", exaggeration: 1 });
    setBasemap(map, "winter", true);

    const layer = new SnowSystem(map, meta, {
      exaggeration: 1,
      mobile: isMobile,
      basePath,
      terrainPhysics: params.get("tp") !== "0",
      onReady: () => {
        initUI(map, layer, meta, setBasemap, { storms, storm });
        // Labels go on AFTER the snow layers so places stay readable.
        map.addLayer(LABEL_LAYER);
      },
    });
    if (params.get("drift") === "0") layer.qGain = 0;
    if (params.get("flakes") === "0") layer.flakesOn = false;
    if (params.get("cover") === "0") layer.coverOn = false;
    if (params.get("wind") === "0") layer.windOn = false;
    window.__snow = layer; // debugging hook
    if (sessionStorage.getItem("lowmem")) {
      layer.flakeCount = 16384;
      layer.streakCount = 8192;
    }
    try {
      map.addLayer(layer);
    } catch (e) {
      fail(`This browser can't run the snow layer: ${e.message}`);
      throw e;
    }
  });

  map.on("error", (e) => console.warn("map error:", e?.error?.message ?? e));
  map.getCanvas().addEventListener("webglcontextlost", () => {
    fail("Graphics memory ran out — reloading with lighter settings…");
    sessionStorage.setItem("lowmem", "1");
    setTimeout(() => location.reload(), 1500);
  });
}

main();

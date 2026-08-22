// Winter-styled raster basemaps. All sources and layers are declared in the
// style up front; switching flips layout.visibility so the terrain source and
// the custom GL layers survive the swap (setStyle would tear them down).

// MapLibre raster sources have no {s} subdomain templating — enumerate them.
export const SOURCES = {
  opentopo: {
    type: "raster",
    tiles: [
      "https://a.tile.opentopomap.org/{z}/{x}/{y}.png",
      "https://b.tile.opentopomap.org/{z}/{x}/{y}.png",
      "https://c.tile.opentopomap.org/{z}/{x}/{y}.png",
    ],
    tileSize: 256,
    maxzoom: 17,
    attribution:
      'Map © <a href="https://opentopomap.org">OpenTopoMap</a> (CC-BY-SA), © OpenStreetMap contributors, SRTM',
  },
  pistes: {
    type: "raster",
    tiles: ["https://tiles.opensnowmap.org/pistes/{z}/{x}/{y}.png"],
    tileSize: 256,
    maxzoom: 18,
    attribution: 'Ski runs © <a href="https://www.opensnowmap.org">OpenSnowMap.org</a>',
  },
  imagery: {
    type: "raster",
    tiles: [
      "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
    ],
    tileSize: 256,
    maxzoom: 18,
    attribution: "Imagery © Esri, Maxar, Earthstar Geographics",
  },
  labels: {
    type: "raster",
    tiles: [
      "https://server.arcgisonline.com/ArcGIS/rest/services/Reference/World_Boundaries_and_Places/MapServer/tile/{z}/{y}/{x}",
    ],
    tileSize: 256,
    maxzoom: 18,
  },
};

export const BASE_LAYERS = [
  { id: "bg", type: "background", paint: { "background-color": "#0b0e14" } },
  { id: "opentopo", type: "raster", source: "opentopo" },
  { id: "imagery", type: "raster", source: "imagery", layout: { visibility: "none" } },
  { id: "pistes", type: "raster", source: "pistes" },
];

// Labels are added separately AFTER the snow layers so places stay readable
// through the snow cover.
export const LABEL_LAYER = {
  id: "labels", type: "raster", source: "labels", paint: { "raster-opacity": 0.85 },
};

// name -> which base layers are visible
const MODES = {
  winter: { opentopo: true, imagery: false },
  satellite: { opentopo: false, imagery: true },
};

export function setBasemap(map, mode, pistesOn = true) {
  const vis = MODES[mode] ?? MODES.winter;
  for (const [id, on] of Object.entries(vis)) {
    map.setLayoutProperty(id, "visibility", on ? "visible" : "none");
  }
  map.setLayoutProperty("pistes", "visibility", pistesOn ? "visible" : "none");
}

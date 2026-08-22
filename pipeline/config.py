"""Single source of truth for the data pipeline.

The contract between pipeline and frontend is meta.json + the atlas PNG
layout defined here. Change tile size / level list here only.

Compared to the parent 3dWindViewer this build carries far fewer wind
levels — snow drift lives in the lowest kilometre or two of the
atmosphere — and spends the freed atlas slots on snow scalar fields.
"""

DATASET_ID = "noaa-hrrr-forecast-48-hour-virtual"

# Lower-troposphere pressure levels only: falling snow and the saltation
# layer need the wind from the surface to ~3 km, not the jet stream.
PRESSURE_LEVELS = [1000, 975, 950, 925, 900, 875, 850, 800, 750, 700]

# Atlas tile order: index 0 = 10m, 1 = 80m, then pressure levels surface->top.
# Each entry: (id, kind, value)
LEVELS = [("10m", "height_agl", 10), ("80m", "height_agl", 80)] + [
    (str(p), "pressure", p) for p in PRESSURE_LEVELS
]  # 12 levels total

# Target regular lat/lon grid (covers the HRRR LCC domain; cells outside the
# native grid get alpha=0 in the atlas). Bounds chosen from the dataset's
# 2D latitude/longitude coords, rounded outward slightly.
WEST, EAST = -134.1, -60.9
SOUTH, NORTH = 21.1, 52.7
TILE_W, TILE_H = 450, 265  # lon x lat samples per level tile (~12 km effective)

ATLAS_COLS, ATLAS_ROWS = 5, 3  # 15 slots: 12 levels + terrain + 2 snow tiles
TERRAIN_TILE_INDEX = 12  # surface elevation, 16-bit packed in R/G
# Snow scalar tiles. Channels are per-tile, quantized over the FIXED ranges in
# SNOW_SCALES below (not data-derived) so legends stay stable across builds.
#   tile A: R = snowfall this hour (mm SWE), G = cumulative SWE since init (mm),
#           B = 2 m temperature (degC), A = valid
#   tile B: R = model snow depth (m), G = percent frozen precip (0-1),
#           B = total precip rate (mm/h), A = valid
SNOW_TILE_A = 13
SNOW_TILE_B = 14
ATLAS_W = ATLAS_COLS * TILE_W   # 2250
ATLAS_H = ATLAS_ROWS * TILE_H   # 795
assert len(LEVELS) <= TERRAIN_TILE_INDEX, "terrain tile would overwrite a level"
assert SNOW_TILE_B < ATLAS_COLS * ATLAS_ROWS, "snow tiles fall off the atlas"
assert ATLAS_W <= 4096 and ATLAS_H <= 4096, "atlas exceeds safe WebGL texture size"

# Fixed quantization ranges for the snow scalar channels, {name: (min, max)}.
# 8-bit over these spans: 0.06 mm/h, 0.6 mm, 0.25 degC, 2 cm, 0.4%, 0.12 mm/h.
SNOW_SCALES = {
    "snowHr": (0.0, 15.0),    # mm SWE accumulated this hour
    "cumSwe": (0.0, 150.0),   # mm SWE since init
    "t2m": (-40.0, 25.0),     # degC
    "depth": (0.0, 5.0),      # m, model snow depth
    "pfrozen": (0.0, 1.0),    # fraction
    "precip": (0.0, 30.0),    # mm/h total precipitation rate
}

# Standalone high-resolution terrain texture (its own PNG, not an atlas tile).
# The atlas is deliberately coarse — 12.8 km cells are plenty for a smooth
# wind field — but drift physics reads the SLOPE and SHELTER of the ground,
# and the cover has to drape on the same surface MapLibre draws. Both demand
# far more resolution, hence a separate texture at the largest safe size.
TERRAIN_HI_W, TERRAIN_HI_H = 4096, 2560
assert TERRAIN_HI_W <= 4096 and TERRAIN_HI_H <= 4096, "terrain exceeds safe texture size"

# Curvature length scale (eta): roughly half the wavelength of the terrain
# features that should drive ridge scour / bowl trapping. ~2 hi-res cells.
CURV_LENGTH_M = 2800.0

# HRRR native grid / projection (verified against the dataset's spatial_ref).
LCC_PROJ = (
    "+proj=lcc +lat_1=38.5 +lat_2=38.5 +lat_0=38.5 +lon_0=-97.5 "
    "+x_0=0 +y_0=0 +R=6371229 +units=m +no_defs"
)
REF_LON = -97.5
ROTCON = 0.6225146  # sin(38.5 deg): grid->earth wind rotation constant

LEAD_HOURS = list(range(49))  # 0..48

# Quantization: per-level min/max over all frames, padded by this fraction.
SCALE_PAD = 0.05


def height_meters(level_id: str, kind: str, value: float) -> float:
    """Approximate geometric altitude for a level (standard atmosphere for
    pressure levels; AGL height used directly for 10m/80m)."""
    if kind == "height_agl":
        return float(value)
    return round(44330.0 * (1.0 - (value / 1013.25) ** 0.1903), 0)

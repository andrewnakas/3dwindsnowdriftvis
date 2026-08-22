"""High-resolution terrain texture for the shader's terrain-flow physics.

This is a separate PNG, not an atlas tile. The wind atlas is deliberately
coarse, but terrain physics reads the *slope* of the ground, and regridding
flattens slopes: the 12.8 km atlas tile keeps under half the slope HRRR
actually resolves. Sampling terrain on its own finer grid recovers it without
inflating the per-frame atlas, and the texture is time-invariant so it loads
once instead of riding along in every frame.

Channels:
  R, G = surface elevation, 16-bit big-endian over (hMin, hMax)
  B    = Liston-Elder curvature, normalized to [-0.5, 0.5] then biased to
         [0, 1] (128 = flat). Precomputed because the normalization is a
         domain-wide maximum the shader cannot know.
  A    = validity (inside the HRRR domain)
"""

import logging

import numpy as np
from PIL import Image

from config import (
    CURV_LENGTH_M,
    EAST,
    NORTH,
    SOUTH,
    TERRAIN_HI_H,
    TERRAIN_HI_W,
    WEST,
)
from reproject import build_index_map, regrid

log = logging.getLogger("terrain")


def cell_size_meters(nx=TERRAIN_HI_W, ny=TERRAIN_HI_H, lat_ref=45.0):
    """Approximate target cell size in meters at a reference latitude."""
    dx = (EAST - WEST) / (nx - 1) * 111320.0 * np.cos(np.radians(lat_ref))
    dy = (NORTH - SOUTH) / (ny - 1) * 110540.0
    return float(dx), float(dy)


def curvature(elev, cell_m, length_scale=CURV_LENGTH_M):
    """Liston-Elder terrain curvature (MicroMet), positive on convex ground.

    Averages the cardinal and diagonal second differences at a fixed physical
    length scale, so the result reflects terrain shape rather than grid
    spacing. Diagonal neighbours are sqrt(2) further away and are weighted
    accordingly.
    """
    inc = max(1, int(round(length_scale / cell_m)))
    z = np.nan_to_num(elev, nan=float(np.nanmean(elev)))
    sh = lambda dr, dc: np.roll(np.roll(z, dr, axis=0), dc, axis=1)  # noqa: E731

    cardinal = (
        4.0 * z - sh(0, inc) - sh(0, -inc) - sh(inc, 0) - sh(-inc, 0)
    ) / (16.0 * inc * cell_m)
    diagonal = (
        4.0 * z - sh(inc, inc) - sh(-inc, -inc) - sh(inc, -inc) - sh(-inc, inc)
    ) / (np.sqrt(2.0) * 16.0 * inc * cell_m)
    curv = cardinal + diagonal

    # Edges wrapped around; they are outside the HRRR domain anyway.
    curv[:inc, :] = curv[-inc:, :] = 0.0
    curv[:, :inc] = curv[:, -inc:] = 0.0
    return curv


def build_terrain_fields(native_height, x_coords, y_coords, use_dem=True):
    """Build the hi-res terrain fields: elevation, curvature, validity.

    Prefers the real terrarium DEM — the same surface MapLibre renders, so
    particles sit on the terrain the viewer can actually see. HRRR's own
    orography is a fallback: it is the surface the weather model ran on, but
    it smooths summits by 700-1600 m, which puts particles inside peaks and
    high above valleys.

    Returns (elev, curv_norm, valid, curv_ref).
    """
    imap = build_index_map(x_coords, y_coords, TERRAIN_HI_W, TERRAIN_HI_H)
    hrrr = regrid(native_height, imap)
    valid = ~np.isnan(hrrr)   # HRRR defines the domain either way

    dx, dy = cell_size_meters()
    elev = None
    if use_dem:
        try:
            from dem import fetch_dem, resample_to_grid

            mosaic, lon_axis, lat_axis = fetch_dem()
            peak, mean = resample_to_grid(
                mosaic, lon_axis, lat_axis, TERRAIN_HI_W, TERRAIN_HI_H
            )
            # Use the MEAN, not the peak. Max-pooling was meant to stop
            # downsampling erasing summits, but it makes every cell report the
            # highest point inside it — over rough ground that is a surface
            # sitting up to ~190 m above the real terrain, so the wind rides an
            # inflated skin and reads as floating above the mountains. The mean
            # is the ground the map actually draws; it is also what the slope
            # and curvature terms need.
            elev = mean
            curv = curvature(mean, 0.5 * (dx + dy))
        except Exception as e:  # noqa: BLE001 - fall back to model orography
            log.warning("DEM fetch failed (%s); falling back to HRRR orography", e)
            elev = None

    if elev is None:
        elev = hrrr
        curv = curvature(elev, 0.5 * (dx + dy))

    # Bathymetry is irrelevant to surface wind and would spend half the 16-bit
    # elevation range on the sea floor; wind over water blows at sea level.
    # The upper clamp drops the handful of corrupt DEM samples (a few pixels
    # read 6-8 km); nothing in the domain is above Whitney at 4421 m, and
    # letting them through would stretch the encoding range for everyone.
    elev = np.clip(elev, 0.0, 4600.0)
    elev = np.where(valid, np.nan_to_num(elev), np.nan)
    curv[~valid] = 0.0

    # Normalize against a high percentile of *mountainous* curvature rather
    # than the domain max. The max is a single freak cell — dividing by it
    # would squash every real ridge to a few percent of the available range,
    # and the whole effect would vanish. Clipping the rare extremes instead
    # keeps ordinary ridges expressive.
    ref = np.abs(curv[valid])
    ref = ref[ref > 0]
    curv_ref = float(np.percentile(ref, 99.5)) if ref.size else 0.0
    curv_ref = max(curv_ref, 1e-9)
    curv_norm = np.clip(0.5 * curv / curv_ref, -0.5, 0.5)
    return elev, curv_norm, valid, curv_ref


def write_terrain_png(out_path, elev, curv_norm, valid):
    """Encode the terrain texture and return its meta block."""
    finite = elev[valid]
    h_min = float(np.floor(finite.min() / 10.0) * 10.0 - 10.0)
    h_max = float(np.ceil(finite.max() / 10.0) * 10.0 + 10.0)

    q = np.round(
        np.clip((np.nan_to_num(elev) - h_min) / (h_max - h_min), 0.0, 1.0) * 65535.0
    ).astype(np.uint32)

    img = np.zeros((elev.shape[0], elev.shape[1], 4), dtype=np.uint8)
    img[:, :, 0] = np.where(valid, q >> 8, 0).astype(np.uint8)
    img[:, :, 1] = np.where(valid, q & 0xFF, 0).astype(np.uint8)
    img[:, :, 2] = np.where(
        valid, np.round((curv_norm + 0.5) * 255.0), 128
    ).astype(np.uint8)
    img[:, :, 3] = np.where(valid, 255, 0).astype(np.uint8)

    Image.fromarray(img, "RGBA").save(out_path, optimize=True)
    return {
        "width": int(elev.shape[1]),
        "height": int(elev.shape[0]),
        "hMin": h_min,
        "hMax": h_max,
        "curvLength": CURV_LENGTH_M,
    }

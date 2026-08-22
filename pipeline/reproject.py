"""LCC -> regular lat/lon regridding and grid->earth wind rotation."""

import numpy as np
from pyproj import Transformer
from scipy.ndimage import map_coordinates, uniform_filter

from config import EAST, LCC_PROJ, NORTH, REF_LON, ROTCON, SOUTH, TILE_H, TILE_W, WEST


def target_lonlat(nx=TILE_W, ny=TILE_H):
    """1D lon (nx) and lat (ny) axes of the output grid.

    Row 0 of the output tile is the NORTH edge (image convention: top row
    first), matching how the PNG is written and sampled in the shader.
    """
    lon = np.linspace(WEST, EAST, nx)
    lat = np.linspace(NORTH, SOUTH, ny)
    return lon, lat


def build_index_map(x_coords, y_coords, nx=TILE_W, ny=TILE_H):
    """Fractional (row, col) indices into the native HRRR grid for every
    target-grid cell, plus a validity mask and the rotation angle field.

    x_coords/y_coords: 1D native projected coordinate arrays from the dataset.
    nx/ny: output grid size; defaults to the wind atlas tile.
    """
    lon, lat = target_lonlat(nx, ny)
    lon2d, lat2d = np.meshgrid(lon, lat)

    tf = Transformer.from_crs("EPSG:4326", LCC_PROJ, always_xy=True)
    px, py = tf.transform(lon2d, lat2d)

    dx = float(x_coords[1] - x_coords[0])
    dy = float(y_coords[1] - y_coords[0])
    col = (px - float(x_coords[0])) / dx
    row = (py - float(y_coords[0])) / dy

    nrow, ncol = len(y_coords), len(x_coords)
    valid = (col >= 0) & (col <= ncol - 1) & (row >= 0) & (row <= nrow - 1)

    # Grid->earth rotation angle at each target cell (radians). Depends only
    # on longitude for LCC with a single reference meridian.
    lond = ((lon2d - REF_LON + 180.0) % 360.0) - 180.0
    alpha = np.deg2rad(ROTCON * lond)

    return {
        "row": row.astype(np.float64),
        "col": col.astype(np.float64),
        "valid": valid,
        "cos_a": np.cos(alpha).astype(np.float32),
        "sin_a": np.sin(alpha).astype(np.float32),
    }


def regrid(field, index_map, presmooth=0):
    """Bilinear sample a native 2D field onto the target grid (NaN outside).

    presmooth: box-average the native field over this many cells first. Point
    sampling a grid several times finer than the target aliases — a ridge crest
    is hit or missed depending on where the target lattice happens to land, so
    the result has neither the right peaks nor the right mean. Set this to the
    native-cells-per-target-cell ratio when downsampling terrain.
    """
    src = np.ascontiguousarray(field, dtype=np.float32)
    if presmooth > 1:
        src = uniform_filter(src, size=presmooth, mode="nearest")
    out = map_coordinates(
        src,
        [index_map["row"], index_map["col"]],
        order=1,
        mode="constant",
        cval=np.nan,
    )
    out[~index_map["valid"]] = np.nan
    return out


def rotate_winds(u, v, index_map):
    """Rotate grid-relative HRRR winds to earth-relative (NCEP formula)."""
    ca, sa = index_map["cos_a"], index_map["sin_a"]
    ue = ca * u + sa * v
    ve = -sa * u + ca * v
    return ue, ve

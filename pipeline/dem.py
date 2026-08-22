"""Real-DEM terrain for the flow physics, from the same tiles the map draws.

The pipeline's other terrain source is HRRR's own orography, which is the
surface the weather model ran on. That is the right choice for masking
below-ground levels, but it is badly wrong for placing particles: HRRR
smooths Bridger Peak from 2865 m to 2124 m, and Emigrant Peak by 1.6 km.
MapLibre renders the real terrarium DEM, so particles built on HRRR's
orography float high over the valleys and sink into the peaks.

Fetching the same DEM the map uses keeps the two in agreement.

Downsampling uses a max filter, not an average: for wind hitting a ridge,
the summit elevation is the physically meaningful number, and averaging is
exactly what erased those peaks in the first place. Slopes come from the
mean-pooled field, which keeps gradients representative of the whole cell.
"""

import io
import logging
import math
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from PIL import Image
from scipy.ndimage import maximum_filter, uniform_filter, zoom

from config import EAST, NORTH, SOUTH, WEST

log = logging.getLogger("dem")

TILE_URL = "https://s3.amazonaws.com/elevation-tiles-prod/terrarium/{z}/{x}/{y}.png"
TILE_PX = 256
# z=8 is ~432 m/px at mid-latitudes: fine enough to keep summits after pooling,
# and ~2400 tiles, which fetches in a couple of minutes.
DEM_ZOOM = 8


def _deg2tile(lon, lat, z):
    n = 2.0**z
    x = (lon + 180.0) / 360.0 * n
    lat_r = math.radians(lat)
    y = (1.0 - math.log(math.tan(lat_r) + 1.0 / math.cos(lat_r)) / math.pi) / 2.0 * n
    return x, y


def _fetch(z, x, y, attempts=3):
    url = TILE_URL.format(z=z, x=x, y=y)
    for i in range(attempts):
        try:
            with urllib.request.urlopen(url, timeout=30) as r:
                img = Image.open(io.BytesIO(r.read())).convert("RGB")
            return np.asarray(img).astype(np.float32)
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return None  # ocean / outside coverage
            if i == attempts - 1:
                raise
        except Exception:
            if i == attempts - 1:
                raise
    return None


def fetch_dem(zoom=DEM_ZOOM, workers=16):
    """Terrarium DEM over the domain as (elev, lon_axis, lat_axis) in meters."""
    x0, y0 = _deg2tile(WEST, NORTH, zoom)
    x1, y1 = _deg2tile(EAST, SOUTH, zoom)
    tx0, tx1 = int(math.floor(x0)), int(math.ceil(x1))
    ty0, ty1 = int(math.floor(y0)), int(math.ceil(y1))
    nx, ny = tx1 - tx0, ty1 - ty0
    log.info("fetching %d x %d = %d DEM tiles at z=%d", nx, ny, nx * ny, zoom)

    mosaic = np.zeros((ny * TILE_PX, nx * TILE_PX), dtype=np.float32)
    jobs = [(ix, iy) for iy in range(ny) for ix in range(nx)]

    def work(job):
        ix, iy = job
        t = _fetch(zoom, tx0 + ix, ty0 + iy)
        return ix, iy, t

    done = 0
    with ThreadPoolExecutor(max_workers=workers) as ex:
        for ix, iy, t in ex.map(work, jobs):
            done += 1
            if done % 400 == 0:
                log.info("  %d/%d tiles", done, len(jobs))
            if t is None:
                continue
            # terrarium: (R*256 + G + B/256) - 32768
            e = (t[:, :, 0] * 256.0 + t[:, :, 1] + t[:, :, 2] / 256.0) - 32768.0
            mosaic[iy * TILE_PX:(iy + 1) * TILE_PX, ix * TILE_PX:(ix + 1) * TILE_PX] = e

    # Web-mercator pixel axes of the mosaic -> lon/lat of each pixel centre
    n = 2.0**zoom
    px_lon = (np.arange(nx * TILE_PX) + 0.5) / TILE_PX + tx0
    px_lat = (np.arange(ny * TILE_PX) + 0.5) / TILE_PX + ty0
    lon_axis = px_lon / n * 360.0 - 180.0
    lat_axis = np.degrees(np.arctan(np.sinh(np.pi * (1.0 - 2.0 * px_lat / n))))
    return mosaic, lon_axis, lat_axis


def resample_to_grid(mosaic, lon_axis, lat_axis, nx, ny):
    """Resample the mercator mosaic onto the target lat/lon grid.

    Returns (peak, mean): peak keeps summit elevations through a max filter,
    mean is the smoothed field slopes are computed from.
    """
    src_h, src_w = mosaic.shape
    lon_t = np.linspace(WEST, EAST, nx)
    lat_t = np.linspace(NORTH, SOUTH, ny)
    col = np.interp(lon_t, lon_axis, np.arange(src_w))
    row = np.interp(-lat_t, -lat_axis, np.arange(src_h))

    # The pool must span a whole target cell, or a summit sitting a few pixels
    # off the sample point is missed entirely — that is how a 3240 m peak read
    # as 1497 m. Mercator rows compress toward the equator, so the vertical
    # span of a target cell varies by latitude; size the pool from the widest
    # spacing so no cell under-samples.
    pool_x = max(1, int(math.ceil(np.max(np.diff(col)))))
    pool_y = max(1, int(math.ceil(np.max(np.diff(row)))))
    log.info("DEM pooling %d x %d source px per target cell", pool_y, pool_x)
    peak_src = maximum_filter(mosaic, size=(pool_y, pool_x), mode="nearest")
    mean_src = uniform_filter(mosaic, size=(pool_y, pool_x), mode="nearest")

    cc, rr = np.meshgrid(col, row)
    from scipy.ndimage import map_coordinates

    peak = map_coordinates(peak_src, [rr, cc], order=1, mode="nearest")
    mean = map_coordinates(mean_src, [rr, cc], order=1, mode="nearest")
    return peak.astype(np.float32), mean.astype(np.float32)

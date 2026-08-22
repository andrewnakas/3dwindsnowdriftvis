"""Synthetic storm for offline frontend development — no network, no dataset.

Writes a complete site/data/: a ridge-field terrain texture, 9 hourly frames
of a uniform westerly gale carrying a Gaussian snowfall blob east across the
ridges near Bozeman, and a matching meta.json. Every shader path is exercised
deterministically: with a steady westerly, drift MUST scour the west (windward)
faces and load the east (lee) faces within a few simulated hours.

Usage:
  PYTHONPATH=pipeline python pipeline/make_test_data.py --out site/data
"""

import argparse
import json
from pathlib import Path

import numpy as np

from config import LEVELS, NORTH, SOUTH, EAST, WEST, TILE_H, TILE_W, height_meters
from encode import write_output
from terrain import curvature, write_terrain_png

# Smaller than the production 4096x2560 texture: the synthetic ridges are
# smooth enough that 1024 cells across resolves them, and the repo stays light.
HI_W, HI_H = 1024, 640

BOZEMAN = (-111.04, 45.68)
WIND_U = 14.0  # m/s, uniform westerly at every level


def lonlat_grids(w, h):
    lon = np.linspace(WEST, EAST, w)[None, :].repeat(h, axis=0)
    lat = np.linspace(NORTH, SOUTH, h)[:, None].repeat(w, axis=1)  # row 0 = north
    return lon, lat


def synthetic_terrain(w, h):
    """Two north-south sinusoidal ridgelines near Bozeman, ~40 km wavelength,
    fading into 500 m plains — a westerly hits them square-on."""
    lon, lat = lonlat_grids(w, h)
    env = np.exp(
        -(((lon - BOZEMAN[0]) / 3.0) ** 2 + ((lat - BOZEMAN[1]) / 2.0) ** 2)
    )
    ridges = np.maximum(0.0, np.sin(2.0 * np.pi * (lon - BOZEMAN[0]) / 0.5)) ** 1.5
    return 500.0 + 1800.0 * env * ridges


def storm_rate(lon, lat, t_hours):
    """Gaussian snowfall blob (mm/h SWE) crossing the ridges west to east."""
    cx = BOZEMAN[0] - 2.5 + 0.5 * t_hours
    r2 = ((lon - cx) / 1.4) ** 2 + ((lat - BOZEMAN[1]) / 1.0) ** 2
    return 6.0 * np.exp(-r2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="site/data")
    ap.add_argument("--leads", type=int, default=9, help="number of hourly leads")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    # --- hi-res terrain texture --------------------------------------------
    elev_hi = synthetic_terrain(HI_W, HI_H)
    dx = (EAST - WEST) / (HI_W - 1) * 111320.0 * np.cos(np.radians(45.0))
    dy = (NORTH - SOUTH) / (HI_H - 1) * 110540.0
    curv = curvature(elev_hi, 0.5 * (dx + dy))
    ref = np.percentile(np.abs(curv[curv != 0]), 99.5)
    curv_norm = np.clip(0.5 * curv / max(ref, 1e-9), -0.5, 0.5)
    valid_hi = np.ones_like(elev_hi, dtype=bool)
    terrain_hi_meta = write_terrain_png(out / "terrain.png", elev_hi, curv_norm, valid_hi)
    terrain_hi_meta["file"] = "terrain.png"
    # write_terrain_png reports the config texture size; this one is smaller.
    terrain_hi_meta["width"], terrain_hi_meta["height"] = HI_W, HI_H

    # --- frames -------------------------------------------------------------
    nlev = len(LEVELS)
    lon, lat = lonlat_grids(TILE_W, TILE_H)
    terrain_lo = synthetic_terrain(TILE_W, TILE_H)

    heights = [height_meters(lid, kind, value) for lid, kind, value in LEVELS]
    scales = [
        {"uMin": -20.0, "uMax": 20.0, "vMin": -20.0, "vMax": 20.0,
         "wMin": -1.0, "wMax": 1.0}
        for _ in range(nlev)
    ]

    frames_by_lead = {}
    snow_by_lead = {}
    cum = np.zeros((TILE_H, TILE_W), dtype=np.float32)
    for t in range(args.leads):
        frame = np.zeros((nlev, TILE_H, TILE_W, 3), dtype=np.float16)
        frame[:, :, :, 0] = WIND_U
        valid = np.ones((nlev, TILE_H, TILE_W), dtype=bool)
        frames_by_lead[t] = (frame, valid)

        rate = storm_rate(lon, lat, float(t)).astype(np.float32)
        rate[rate < 0.05] = 0.0
        if t > 0:
            cum = cum + rate  # 1 h per lead
        snow_by_lead[t] = {
            "snowHr": rate if t > 0 else np.zeros_like(rate),
            "cumSwe": cum.copy(),
            "t2m": np.full_like(rate, -8.0),
            "depth": (cum / 100.0).astype(np.float32),  # ~10:1 fresh snow ratio
            "pfrozen": np.ones_like(rate),
            "precip": rate,
        }

    meta = write_output(
        out, frames_by_lead, scales, "2026-01-15T00:00:00Z", heights,
        terrain_lo, terrain_hi_meta, snow_by_lead,
    )
    print(f"wrote {len(meta['frames'])} synthetic frames + terrain to {out}")
    print(json.dumps({k: meta[k] for k in ("bounds", "atlas", "snow")}, indent=2)[:600])


if __name__ == "__main__":
    main()

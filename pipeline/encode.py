"""Quantize regridded wind + snow fields into texture-atlas PNGs + meta.json.

Atlas channels per wind-level tile:
  R = u (east wind), G = v (north wind), B = omega (vertical velocity, Pa/s),
  A = valid mask (inside HRRR domain AND level above ground).
Per-level min/max scaling for each channel lives in meta.json.

Two extra tiles carry snow scalar fields (see config.SNOW_TILE_A/B), each
quantized over the FIXED ranges in config.SNOW_SCALES so the frontend's
depth legend and density model stay stable across builds.
"""

import json
import math
from pathlib import Path

import numpy as np
from PIL import Image

from config import (
    ATLAS_COLS,
    ATLAS_H,
    ATLAS_ROWS,
    ATLAS_W,
    DATASET_ID,
    EAST,
    LEVELS,
    NORTH,
    SCALE_PAD,
    SNOW_SCALES,
    SNOW_TILE_A,
    SNOW_TILE_B,
    SOUTH,
    TERRAIN_TILE_INDEX,
    TILE_H,
    TILE_W,
    WEST,
)

GRAVITY = 9.80665
RHO0, SCALE_H = 1.225, 8500.0  # standard-atmosphere density profile

# Which snow plane lands in which tile/channel. Order matters: R, G, B.
SNOW_TILE_LAYOUT = {
    SNOW_TILE_A: ("snowHr", "cumSwe", "t2m"),
    SNOW_TILE_B: ("depth", "pfrozen", "precip"),
}


def w_factor(height_m: float) -> float:
    """m/s of upward motion per Pa/s of omega at this altitude: w = -omega/(rho g)."""
    rho = RHO0 * math.exp(-height_m / SCALE_H)
    return -1.0 / (rho * GRAVITY)


def compute_scales(frames):
    """Per-level min/max of u, v, omega over all frames.

    frames: list of (nlev, H, W, 3) arrays.
    """
    stack = np.stack(frames)  # (nframes, nlev, H, W, 3)
    scales = []
    for i in range(stack.shape[1]):
        entry = {}
        for name, k, min_pad in (("u", 0, 0.5), ("v", 1, 0.5), ("w", 2, 0.05)):
            vals = stack[:, i, :, :, k]
            lo, hi = float(np.nanmin(vals)), float(np.nanmax(vals))
            pad = max((hi - lo) * SCALE_PAD, min_pad)
            entry[f"{name}Min"] = lo - pad
            entry[f"{name}Max"] = hi + pad
        scales.append(entry)
    return scales


def _tile_view(atlas, index):
    r0 = (index // ATLAS_COLS) * TILE_H
    c0 = (index % ATLAS_COLS) * TILE_W
    return atlas[r0 : r0 + TILE_H, c0 : c0 + TILE_W]


def encode_terrain_tile(atlas, terrain, t_range):
    """Pack surface elevation (m) 16-bit into R (hi) / G (lo) of the spare tile.
    Linear decode (r*255*256 + g*255)/65535 commutes with bilinear filtering."""
    lo_m, hi_m = t_range
    ok = ~np.isnan(terrain)
    v = np.round(np.clip((np.nan_to_num(terrain) - lo_m) / (hi_m - lo_m), 0, 1) * 65535).astype(np.uint32)
    tile = _tile_view(atlas, TERRAIN_TILE_INDEX)
    tile[:, :, 0] = np.where(ok, v >> 8, 0).astype(np.uint8)
    tile[:, :, 1] = np.where(ok, v & 0xFF, 0).astype(np.uint8)
    tile[:, :, 3] = np.where(ok, 255, 0).astype(np.uint8)


def encode_snow_tiles(atlas, snow):
    """snow: {name: (TILE_H, TILE_W) float array} for every name in
    SNOW_SCALES. NaN cells (outside the domain) get alpha=0."""
    for index, names in SNOW_TILE_LAYOUT.items():
        tile = _tile_view(atlas, index)
        ok = np.ones((TILE_H, TILE_W), dtype=bool)
        for plane_name in names:
            ok &= ~np.isnan(snow[plane_name])
        for ch, plane_name in enumerate(names):
            lo, hi = SNOW_SCALES[plane_name]
            q = np.clip((np.nan_to_num(snow[plane_name]).astype(np.float64) - lo) / (hi - lo), 0, 1)
            tile[:, :, ch] = np.where(ok, np.round(q * 255), 0).astype(np.uint8)
        tile[:, :, 3] = np.where(ok, 255, 0).astype(np.uint8)


def encode_frame(frame, valid, scales, terrain=None, t_range=None, snow=None):
    """frame: (nlev, TILE_H, TILE_W, 3); valid: (nlev, TILE_H, TILE_W) bool;
    snow: {name: (TILE_H, TILE_W)} scalar planes for this lead."""
    atlas = np.zeros((ATLAS_H, ATLAS_W, 4), dtype=np.uint8)
    if terrain is not None:
        encode_terrain_tile(atlas, terrain, t_range)
    if snow is not None:
        encode_snow_tiles(atlas, snow)
    for i in range(frame.shape[0]):
        s = scales[i]
        ok = valid[i] & ~np.isnan(frame[i]).any(axis=-1)
        tile = _tile_view(atlas, i)
        for ch, (lo, hi) in enumerate(
            ((s["uMin"], s["uMax"]), (s["vMin"], s["vMax"]), (s["wMin"], s["wMax"]))
        ):
            q = np.clip((frame[i, :, :, ch].astype(np.float64) - lo) / (hi - lo), 0, 1)
            tile[:, :, ch] = np.where(ok, np.round(q * 255), 0).astype(np.uint8)
        tile[:, :, 3] = np.where(ok, 255, 0).astype(np.uint8)
    return atlas


def snow_meta_block():
    """The meta.json 'snow' block: where each scalar lives and how to decode it."""
    chans = "rgb"
    out = {}
    for key, (index, names) in zip(("tileA", "tileB"), SNOW_TILE_LAYOUT.items()):
        out[key] = {
            "index": index,
            "channels": {
                name: {"ch": chans[ch], "min": SNOW_SCALES[name][0], "max": SNOW_SCALES[name][1]}
                for ch, name in enumerate(names)
            },
        }
    return out


def write_output(
    out_dir, frames_by_lead, scales, init_time_iso, heights, terrain,
    terrain_hi=None, snow_by_lead=None,
):
    """frames_by_lead: {lead: (frame, valid)}; heights: per-level meters ASL;
    terrain: (TILE_H, TILE_W) surface elevation in meters;
    terrain_hi: optional meta block for the standalone hi-res terrain PNG;
    snow_by_lead: {lead: {name: (TILE_H, TILE_W)}} scalar snow planes."""
    out = Path(out_dir)
    (out / "frames").mkdir(parents=True, exist_ok=True)

    t_range = (
        float(np.floor(np.nanmin(terrain) / 10) * 10 - 10),
        float(np.ceil(np.nanmax(terrain) / 10) * 10 + 10),
    )
    frame_entries = []
    for lead, (frame, valid) in sorted(frames_by_lead.items()):
        name = f"frames/f{lead:02d}.png"
        snow = snow_by_lead.get(lead) if snow_by_lead else None
        atlas = encode_frame(frame, valid, scales, terrain, t_range, snow)
        Image.fromarray(atlas, "RGBA").save(out / name, optimize=True)
        frame_entries.append({"lead_hours": lead, "file": name})

    meta = {
        "dataset": DATASET_ID,
        "init_time": init_time_iso,
        "bounds": {"west": WEST, "south": SOUTH, "east": EAST, "north": NORTH},
        "tile": {"width": TILE_W, "height": TILE_H},
        "atlas": {"cols": ATLAS_COLS, "rows": ATLAS_ROWS},
        "terrain": {"index": TERRAIN_TILE_INDEX, "hMin": t_range[0], "hMax": t_range[1]},
        **({"terrainHi": terrain_hi} if terrain_hi else {}),
        "snow": snow_meta_block(),
        "frames": frame_entries,
        "levels": [
            {
                "index": i,
                "id": lid,
                "kind": kind,
                "value": value,
                "heightMeters": round(float(heights[i]), 1),
                "wFactor": 0.0 if kind == "height_agl" else round(w_factor(heights[i]), 5),
                **{k: round(v, 3) for k, v in scales[i].items()},
            }
            for i, (lid, kind, value) in enumerate(LEVELS)
        ],
    }
    (out / "meta.json").write_text(json.dumps(meta))
    return meta

"""Build web-ready wind + snow frames from the dynamical.org virtual HRRR dataset.

Usage:
  python pipeline/build_frames.py --out site/data [--leads "0 6 12"] [--workers 4]

Alongside the wind levels, each frame atlas carries two snow scalar tiles
(see config.py): snowfall this hour, cumulative SWE since init, 2 m
temperature, model snow depth, percent-frozen precipitation, and total
precipitation. Hourly snowfall SWE comes straight from HRRR's
snowfall_water_equivalent_surface (kg m-2 == mm of water accumulated since
the previous hourly step); total_precipitation_surface ships in the same
per-step units. Both are unit-checked at read time.
"""

import argparse
import logging
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np

from config import LEAD_HOURS, LEVELS, PRESSURE_LEVELS, TILE_H, TILE_W, height_meters
from encode import compute_scales, write_output
from hrrr_source import open_datasets, pick_init
from reproject import build_index_map, regrid, rotate_winds
from terrain import build_terrain_fields, write_terrain_png

log = logging.getLogger("build_frames")

SFC_VARS = [("wind_u_10m", "wind_v_10m"), ("wind_u_80m", "wind_v_80m")]

# Set TERRAIN_FALLBACK_OK=1 to publish without the hi-res terrain texture (the
# drift physics then runs on the coarse atlas tile). Off by default so a
# broken terrain build fails loudly instead of quietly shipping degraded.
ALLOW_TERRAIN_FALLBACK = os.environ.get("TERRAIN_FALLBACK_OK") == "1"


def amount_to_mm(da, name):
    """Convert a per-step precipitation amount to mm of water, keyed off its
    units attribute. Fails loudly on units it can't place — a silent factor
    error here corrupts every derived snow field. The dataset's steps are
    hourly, so mm per step is numerically mm/h."""
    units = str(da.attrs.get("units", "")).strip().lower()
    vals = da.values
    if units in ("kg m-2", "kg m**-2", "kg/m^2", "mm"):
        return vals
    if units == "m":
        return vals * 1000.0
    raise ValueError(f"{name}: unrecognized amount units {units!r}")


def to_celsius(da):
    vals = da.values
    units = str(da.attrs.get("units", "")).strip().lower()
    if units in ("k", "kelvin") or np.nanmean(vals) > 150.0:
        return vals - 273.15
    return vals


def build_frame(sfc, prs, init, lead, index_map, attempts=3):
    """Read + regrid + rotate all levels + snow scalars for one lead hour.

    Returns (frame, valid, snow): frame (nlev, TILE_H, TILE_W, 3) float16
    [u, v, omega], an above-ground/in-domain mask, and a dict of regridded
    snow planes (snow_hr NOT yet accumulated — main() does the prefix sum).
    """
    last_err = None
    for attempt in range(attempts):
        try:
            return _build_frame(sfc, prs, init, lead, index_map)
        except KeyError:
            # A missing variable is permanent — retrying burns minutes per
            # lead on an error that a code or dataset-schema change caused.
            raise
        except Exception as e:  # noqa: BLE001 - network reads; retry then fail
            last_err = e
            log.warning("lead %d attempt %d failed: %s", lead, attempt + 1, e)
            time.sleep(5 * (attempt + 1))
    raise RuntimeError(f"lead {lead} failed after {attempts} attempts") from last_err


def _build_frame(sfc, prs, init, lead, index_map):
    nlev = len(LEVELS)
    frame = np.zeros((nlev, TILE_H, TILE_W, 3), dtype=np.float16)
    valid = np.zeros((nlev, TILE_H, TILE_W), dtype=bool)
    domain = index_map["valid"]

    sfc_t = sfc.sel(init_time=init).isel(lead_time=lead)
    psfc = regrid(sfc_t["pressure_surface"].values, index_map)  # Pa

    for i, (uname, vname) in enumerate(SFC_VARS):
        u = regrid(sfc_t[uname].values, index_map)
        v = regrid(sfc_t[vname].values, index_map)
        frame[i, :, :, 0], frame[i, :, :, 1] = rotate_winds(u, v, index_map)
        valid[i] = domain  # AGL levels are above ground by definition

    prs_t = prs.sel(init_time=init).isel(lead_time=lead)
    u_slab = prs_t.wind_u.values  # (y, x, nplev)
    v_slab = prs_t.wind_v.values
    w_slab = prs_t.vertical_velocity.values  # omega, Pa/s
    plev_axis = list(prs.pressure_level.values)
    for j, p in enumerate(PRESSURE_LEVELS):
        k = plev_axis.index(p)
        u = regrid(u_slab[:, :, k], index_map)
        v = regrid(v_slab[:, :, k], index_map)
        idx = 2 + j
        frame[idx, :, :, 0], frame[idx, :, :, 1] = rotate_winds(u, v, index_map)
        frame[idx, :, :, 2] = regrid(w_slab[:, :, k], index_map)
        # level is above ground where its pressure is below surface pressure
        valid[idx] = domain & (p * 100.0 <= np.nan_to_num(psfc, nan=0.0))

    if np.isnan(frame[:, TILE_H // 2, TILE_W // 2, :2]).any():
        raise ValueError(f"lead {lead}: NaN wind at domain center")

    # --- snow scalar planes -------------------------------------------------
    snow_amt = regrid(
        amount_to_mm(sfc_t["snowfall_water_equivalent_surface"],
                     "snowfall_water_equivalent_surface"),
        index_map,
    )
    snow_amt = np.clip(np.nan_to_num(snow_amt), 0.0, None)
    precip = regrid(
        amount_to_mm(sfc_t["total_precipitation_surface"],
                     "total_precipitation_surface"),
        index_map,
    )
    precip = np.clip(np.nan_to_num(precip), 0.0, None)
    # CPOFP uses -50 as its "no precipitation" sentinel; clip it to 0.
    pfrozen = regrid(sfc_t["percent_frozen_precipitation_surface"].values, index_map)
    pfrozen = np.clip(np.nan_to_num(pfrozen), 0.0, 100.0) / 100.0
    t2m = regrid(to_celsius(sfc_t["temperature_2m"]), index_map)
    depth = regrid(sfc_t["snow_thickness_surface"].values, index_map)
    depth = np.clip(np.nan_to_num(depth), 0.0, None)

    # Lead 0 is the analysis: its "since the previous step" accumulation has
    # no step before it, so nothing has fallen yet in this forecast's story.
    snow_hr = snow_amt if lead > 0 else np.zeros_like(snow_amt)

    snow = {
        "snow_hr": snow_hr.astype(np.float32),
        "t2m": t2m.astype(np.float32),
        "depth": depth.astype(np.float32),
        "pfrozen": pfrozen.astype(np.float32),
        "precip": precip.astype(np.float32),
        "domain": domain,
    }
    return frame, valid, snow


def read_heights(prs, init):
    """Domain-mean geopotential height per level (m ASL), read once at lead 0.
    Falls back to standard atmosphere per level on failure."""
    out = [10.0, 80.0]
    try:
        gh = prs.sel(init_time=init).isel(lead_time=0).geopotential_height.values
        plev_axis = list(prs.pressure_level.values)
        for p in PRESSURE_LEVELS:
            out.append(float(np.nanmean(gh[:, :, plev_axis.index(p)])))
    except Exception as e:  # noqa: BLE001
        log.warning("geopotential height read failed (%s); using std atmosphere", e)
        out = [height_meters(lid, kind, value) for lid, kind, value in LEVELS]
    return out


def accumulate_swe(snow_by_lead):
    """Prefix-sum hourly snowfall into cumulative SWE since init, in lead
    order. Leads may be sparse (test builds): each gap is filled by holding
    the newer lead's hourly rate across the gap."""
    cum = None
    prev_lead = None
    for lead in sorted(snow_by_lead):
        s = snow_by_lead[lead]
        if cum is None:
            cum = np.zeros_like(s["snow_hr"])
        else:
            gap_h = float(lead - prev_lead)
            cum = cum + s["snow_hr"] * gap_h
        # NaN outside the domain so the encoder masks it out.
        s["cumSwe"] = np.where(s["domain"], cum, np.nan).astype(np.float32)
        prev_lead = lead
        # Rename to the meta.json channel names and mask off-domain cells.
        s["snowHr"] = np.where(s["domain"], s.pop("snow_hr"), np.nan)
        for name in ("t2m", "depth", "pfrozen", "precip"):
            s[name] = np.where(s["domain"], s[name], np.nan)
        del s["domain"]
        # Sanity: accumulation must never decrease and rates must be finite.
        assert np.nanmin(s["snowHr"]) >= 0.0, f"lead {lead}: negative snowfall"
        assert np.nanmax(s["snowHr"]) < 200.0, f"lead {lead}: absurd snowfall rate"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--leads", default="", help='e.g. "0 1 2"; default all 0-48')
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--init", default="auto", help="auto or ISO time")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    leads = [int(x) for x in args.leads.split()] if args.leads.strip() else LEAD_HOURS

    t0 = time.time()
    sfc, prs = open_datasets()
    init = pick_init(prs, args.init)
    init_iso = np.datetime_as_string(init, unit="s") + "Z"
    log.info("init=%s leads=%s", init_iso, leads)

    index_map = build_index_map(prs.x.values, prs.y.values)
    heights = read_heights(prs, init)

    orography = sfc.sel(init_time=init).isel(lead_time=0).geopotential_height_surface.values
    # The atlas tile is ~4x coarser than native, so area-average first — point
    # sampling would hit or miss ridge crests depending on lattice alignment.
    terrain = regrid(orography, index_map, presmooth=4)

    terrain_hi_meta = None
    try:
        # write_output creates this later, but the terrain texture is written
        # first and a clean checkout has no site/data yet.
        Path(args.out).mkdir(parents=True, exist_ok=True)
        elev, curv, valid_hi, curv_max = build_terrain_fields(
            orography, prs.x.values, prs.y.values
        )
        terrain_hi_meta = write_terrain_png(
            Path(args.out) / "terrain.png", elev, curv, valid_hi
        )
        terrain_hi_meta["file"] = "terrain.png"
        log.info(
            "hi-res terrain %dx%d, elev %.0f..%.0f m, curv max %.2e 1/m",
            terrain_hi_meta["width"], terrain_hi_meta["height"],
            terrain_hi_meta["hMin"], terrain_hi_meta["hMax"], curv_max,
        )
    except Exception:  # noqa: BLE001
        # Only a DEM fetch problem is worth degrading for — anything else is a
        # bug, and swallowing it ships a build that looks green while silently
        # dropping the drift physics, which is the whole point of this site.
        log.exception("hi-res terrain build failed; frontend will use the atlas tile")
        if not ALLOW_TERRAIN_FALLBACK:
            raise

    frames_by_lead = {}
    snow_by_lead = {}
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(build_frame, sfc, prs, init, t, index_map): t for t in leads}
        for fut in as_completed(futs):
            lead = futs[fut]
            frame, valid, snow = fut.result()
            frames_by_lead[lead] = (frame, valid)
            snow_by_lead[lead] = snow
            log.info("lead %02d done (%d/%d)", lead, len(frames_by_lead), len(leads))

    # Cumulative SWE needs the leads in order, so it happens after the pool.
    accumulate_swe(snow_by_lead)
    log.info(
        "domain-max hourly snowfall %.1f mm SWE, final cumulative %.1f mm, t2m %.0f..%.0f C",
        max(float(np.nanmax(s["snowHr"])) for s in snow_by_lead.values()),
        float(np.nanmax(snow_by_lead[max(snow_by_lead)]["cumSwe"])),
        float(np.nanmin(snow_by_lead[max(snow_by_lead)]["t2m"])),
        float(np.nanmax(snow_by_lead[max(snow_by_lead)]["t2m"])),
    )

    scales = compute_scales([f for f, _ in frames_by_lead.values()])
    meta = write_output(
        args.out, frames_by_lead, scales, init_iso, heights, terrain,
        terrain_hi_meta, snow_by_lead,
    )

    log.info(
        "wrote %d frames to %s in %.1f min (init %s)",
        len(meta["frames"]), args.out, (time.time() - t0) / 60, init_iso,
    )


if __name__ == "__main__":
    sys.exit(main())

"""Open the dynamical.org virtual HRRR dataset and pick a usable init time."""

import logging

import dynamical_catalog
import numpy as np

from config import DATASET_ID

log = logging.getLogger(__name__)


def open_datasets():
    """Return (surface_ds, pressure_ds), both lazy xarray Datasets."""
    sfc = dynamical_catalog.open(DATASET_ID, chunks=None)
    prs = dynamical_catalog.open(DATASET_ID, chunks=None, group="pressure_level")
    return sfc, prs


def pick_init(prs, requested: str | None = None, max_steps_back: int = 4):
    """Choose an init_time whose final lead hour is actually readable.

    The virtual dataset lists an init as soon as ingest starts; the last lead
    hours may not exist yet. Probe lead 48 and step back 6h until a complete
    init is found.
    """
    if requested and requested != "auto":
        init = np.datetime64(requested)
        _probe(prs, init)
        return init

    inits = prs.init_time.values
    for i in range(1, max_steps_back + 2):
        init = inits[-i]
        try:
            _probe(prs, init)
            log.info("using init %s", init)
            return init
        except Exception as e:  # noqa: BLE001 - any read failure means incomplete
            log.warning("init %s not complete (%s); stepping back 6h", init, e)
    raise RuntimeError(f"no complete init found in last {max_steps_back + 1} cycles")


def _probe(prs, init):
    sl = (
        prs.wind_u.sel(init_time=init, pressure_level=500)
        .isel(lead_time=-1, x=slice(890, 910), y=slice(520, 540))
        .values
    )
    if np.isnan(sl).all():
        raise ValueError("probe slice is all-NaN")

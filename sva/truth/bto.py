from functools import cache
from pathlib import Path

import numpy as np
import xarray as xr


@cache
def _load_bto_data() -> xr.DataArray:
    """Loads BTO data into an xarray.
    These are diffraction patterns over ~0 to 25 AA.
    The termperature range is 150 to 445.
    """
    path = Path(__file__).parent / "bto_data.nc"
    return xr.open_dataarray(path)


def truth_bto(temperature) -> np.ndarray:
    """
    Returns interpolated X-ray diffraction pattern for BTO at
    temperature between 150 and 445.

    Parameters
    ----------
    temperature : Union[float, list, np.ndarray]
        temperature points to estimate patterns for
    """
    da = _load_bto_data()
    return da.interp(temperature=temperature).data

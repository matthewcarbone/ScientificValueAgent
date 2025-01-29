from functools import cache
from pathlib import Path

import numpy as np
import xarray as xr
from attrs import define, field

from ..base import Experiment, ExperimentProperties

root = Path(__file__).parent.resolve()


@cache
def load_data():
    return xr.open_dataset(root / "ds_AlLiFe_square_29Jan2025_14-31-22.nc")


@define
class WaferAlLiFe(Experiment):
    properties = field(
        factory=lambda: ExperimentProperties(
            n_input_dim=2,
            n_output_dim=650,
            domain=np.array([[-20.0, 20.0], [-20.0, 20.0]]).T,
        )
    )

    def _truth(self, X):
        data = load_data()
        return np.array(
            [
                data["iq"].interp({"x": point[0], "y": point[1]}).data
                for point in X
            ]
        )

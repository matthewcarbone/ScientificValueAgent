import torch

torch.set_default_dtype(torch.float64)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

from sva.models.gp.gp import (
    EasyFixedNoiseGP,
    EasyMultiTaskGP,
    EasySingleTaskGP,
    fit_EasyGP_mll,
)

__all__ = [
    "EasySingleTaskGP",
    "EasyMultiTaskGP",
    "EasyFixedNoiseGP",
    "fit_EasyGP_mll",
]

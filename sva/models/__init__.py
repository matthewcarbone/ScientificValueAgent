import torch

torch.set_default_dtype(torch.float64)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

from sva.models.gp.gp import EasyFixedNoiseGP, EasyMultiTaskGP, EasySingleTaskGP

__all__ = [
    "EasySingleTaskGP",
    "EasyMultiTaskGP",
    "EasyFixedNoiseGP",
]

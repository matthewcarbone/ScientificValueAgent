from .ggce.ggce import Peierls
from .negatedackley2d.negatedackley2d import NegatedAckley2d
from .negatedrosenbrock2d.negatedrosenbrock2d import NegatedRosenbrock2d
from .rastrigin2d.rastrigin2d import Rastrigin2d
from .simple2d.simple2d import Simple2d
from .simplesigmoid.simplesigmoid import SimpleSigmoid

__all__ = [
    "Simple2d",
    "NegatedRosenbrock2d",
    "Rastrigin2d",
    "NegatedAckley2d",
    "SimpleSigmoid",
    "Peierls",
]

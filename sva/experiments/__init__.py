from .gp.gp import GPDream, GPDreamKNN
from .negatedackley2d.negatedackley2d import NegatedAckley2d
from .negatedrosenbrock2d.negatedrosenbrock2d import NegatedRosenbrock2d
from .quadraticprior.quadraticprior import QuadraticPrior
from .rastrigin2d.rastrigin2d import Rastrigin2d
from .simple2d.simple2d import Simple2d
from .simple5phase.simple5phase import Simple5Phase
from .simplesigmoid.simplesigmoid import SimpleSigmoid
from .sine2phase.sine2phase import Sine2Phase

__all__ = [
    "Simple2d",
    "NegatedRosenbrock2d",
    "Rastrigin2d",
    "NegatedAckley2d",
    "SimpleSigmoid",
    "Sine2Phase",
    "GPDream",
    "GPDreamKNN",
    "QuadraticPrior",
    "Simple5Phase",
]

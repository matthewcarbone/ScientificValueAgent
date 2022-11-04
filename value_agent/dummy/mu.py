import numpy as np

from value_agent.utils import sigmoid


def mu_XAS(E, a=1.0, b=1.0, c=1.0, d=1.0, e=1.0, f=1.0):
    """Dummy spectrum creator from Torrisi et al.
    
    Parameters
    ----------
    E : numpy.ndarray
        Energy grid.

    Returns
    -------
    numpy.ndarray
        The spectrum on the provided grid.
    """

    t1 = 1.0 / (1.0 + np.exp(-3.0 * a * (E + b)))
    t2 = 1.0 / (d * np.sqrt(2.0 * np.pi)) * np.exp(-(E-c)**2 / 2.0 / d**2)
    t3 = -1.0 / (d * np.sqrt(2.0 * np.pi)) * np.exp(-(E-e)**2 / 2.0 / f**2)
    return t1 + t2 + t3


def two_phase_raster_linear_XAS(coords, E=np.linspace(-5, 10, 100)):

    def param_a(x, y, a0=2, af=1, scale=20, offset=0.5):
        val = np.mean([x, y])
        sig = 1.0 / (1.0 + np.exp(-(val - offset) * scale))
        return a0 + sig * (af - a0)

    def param_b(x, y, scale=0.05):
        return 0.6 * np.random.normal(loc=1.0, scale=scale)

    def param_c(x, y, c0=3, cf=7, scale=20, offset=0.5):
        val = np.mean([x, y])
        sig = 1.0 / (1.0 + np.exp(-(val - offset) * scale))
        return c0 + sig * (cf - c0)

    def param_d(x, y, scale=0.05):
        return np.random.normal(loc=1.0, scale=scale)

    def param_e(x, y, e0=6, ef=4, scale=20, offset=0.5):
        val = np.mean([x, y])
        sig = 1.0 / (1.0 + np.exp(-(val - offset) * scale))
        return e0 + sig * (ef - e0)

    def param_f(x, y, scale=0.05):
        return np.random.normal(loc=1.0, scale=scale)

    return np.array([
        mu_XAS(
            E,
            a=param_a(x, y),
            b=param_b(x, y),
            c=param_c(x, y),
            d=param_d(x, y),
            e=param_e(x, y),
            f=param_f(x, y)
        )
        for (x, y) in coords
    ])


def two_phase_raster_sine_XAS(coords, E=np.linspace(-5, 10, 100)):

    def param_a(x, y, a0=2, af=1, scale=50, offset=0.5):
        val = (y - np.sin(2.0 * np.pi * x) / 2.0 - 0.5)**2 + offset
        sig = 1.0 / (1.0 + np.exp(-(val - offset) * scale))
        return a0 + sig * (af - a0)

    def param_b(x, y, scale=0.05):
        return 0.6 * np.random.normal(loc=1.0, scale=scale)

    def param_c(x, y, c0=3, cf=7, scale=50, offset=0.5):
        val = (y - np.sin(2.0 * np.pi * x) / 2.0 - 0.5)**2 + offset
        sig = 1.0 / (1.0 + np.exp(-(val - offset) * scale))
        return c0 + sig * (cf - c0)

    def param_d(x, y, scale=0.05):
        return np.random.normal(loc=1.0, scale=scale)

    def param_e(x, y, e0=6, ef=4, scale=50, offset=0.5):
        val = (y - np.sin(2.0 * np.pi * x) / 2.0 - 0.5)**2 + offset
        sig = 1.0 / (1.0 + np.exp(-(val - offset) * scale))
        return e0 + sig * (ef - e0)

    def param_f(x, y, scale=0.05):
        return np.random.normal(loc=1.0, scale=scale)

    return np.array([
        mu_XAS(
            E,
            a=param_a(x, y),
            b=param_b(x, y),
            c=param_c(x, y),
            d=param_d(x, y),
            e=param_e(x, y),
            f=param_f(x, y)
        )
        for (x, y) in coords
    ])


def mu_Gaussians(p, E=np.linspace(-1, 1, 100), x0=0.5, sd=0.05):
    """Returns a dummy "spectrum" which is just two Gaussian functions. The
    proportion of the two functions is goverened by ``p``.
    
    Parameters
    ----------
    p : float
        The proportion of the first phase.
    E : numpy.ndarray
        Energy grid.
    
    Returns
    -------
    numpy.ndarray
        The spectrum on the provided grid.
    """

    p2 = 1.0 - p
    return p * np.exp(-(x0 - E)**2 / sd) + p2 * np.exp(-(x0 + E)**2 / sd)


def phase_1_linear_on_2d_raster(x, y, x0=0.5, a=30.0):
    return sigmoid((x + y) / 2.0, x0, a)


def linear_on_2d_raster_observations(X):
    phase_1 = [phase_1_linear_on_2d_raster(*c) for c in X]
    return np.array([mu_Gaussians(p) for p in phase_1])


def _sine(x):
    return 0.25 * np.sin(2.0 * np.pi * x)


def phase_1_sine_on_2d_raster(x, y, x0=0.5, a=30.0):
    """Takes the y-distance between a sigmoid function and the provided point.
    """
    
    distance = y - _sine(x)
    return sigmoid(distance, x0, a)


def sine_on_2d_raster_observations(X):
    phase_1 = [phase_1_sine_on_2d_raster(*c) for c in X]
    return np.array([mu_Gaussians(p) for p in phase_1])

from pathlib import Path

import numpy as np
from attrs import define, field
from attrs.validators import ge, instance_of, optional

from sva.logger import logger
from sva.monty.json import MSONable
from sva.utils import Timer, seed_everything


@define
class CampaignData(MSONable):
    """Container for sampled data during experiments. This is a serializable
    abstraction over the data used during the experiments. This includes
    updating it and saving it to disk. Note that data contained here must
    always be two-dimensional (x.ndim == 2)."""

    X: np.ndarray = field(default=None)
    Y: np.ndarray = field(default=None)
    metadata = field(factory=list)

    @property
    def N(self):
        if self.X is None:
            return 0
        return self.X.shape[0]

    @property
    def is_initialized(self):
        if self.X is None and self.Y is None:
            return False
        return True

    def update_X(self, X):
        """Updates the current input data with new inputs.

        Parameters
        ----------
        X : np.ndarray
            Two-dimensional data to update the X values with.
        """

        assert X.ndim == 2
        if self.X is not None:
            self.X = np.concatenate([self.X, X], axis=0)
        else:
            self.X = X

    def update_Y(self, Y):
        """Updates the current output data with new outputs.

        Parameters
        ----------
        Y : np.array
            Two-dimensional data to update the Y values with.
        """

        if self.Y is not None:
            self.Y = np.concatenate([self.Y, Y], axis=0)
        else:
            self.Y = Y

    def update_metadata(self, new_metadata):
        self.metadata.extend(new_metadata)

    def update(self, X, Y, metadata=None):
        """Helper method for updating the data attribute with new data.

        Parameters
        ----------
        X, Y : numpy.ndarray
            The input and output data to update with.
        metadata : list, optional
            Should be a list of Any or None.
        """

        assert X.shape[0] == Y.shape[0]
        if metadata is not None:
            assert X.shape[0] == len(metadata)
        else:
            metadata = [None] * X.shape[0]

        self.update_X(X)
        self.update_Y(Y)
        self.update_metadata(metadata)

    def prime(self, experiment, protocol, **kwargs):
        """Initializes the data via some provided protocol.

        Current options are "random", "LatinHypercube" and "dense". In
        addition, there is the "cold_start" option, which does nothing. This
        will force the campaign model to use the unconditioned prior.

        Parameters
        ----------
        experiment : sva.experiments.Experiment
            Callable experiment.
        protocol : str, optional
            The method for using to initialize the data.
        kwargs
            To pass to the particular method.
        """

        if protocol == "cold_start":
            return  # do nothing!

        if protocol == "random":
            X = experiment.get_random_coordinates(**kwargs)
        elif protocol == "LatinHypercube":
            X = experiment.get_latin_hypercube_coordinates(**kwargs)
        elif protocol == "dense":
            X = experiment.get_dense_coordinates(**kwargs)
        else:
            raise NotImplementedError(f"Unknown provided protocol {protocol}")

        d = {"experiment": experiment.name, "policy": protocol}
        self.update(X, experiment(X), metadata=[d] * X.shape[0])

    def __eq__(self, exp):
        # XOR for when things aren't initialized
        if (exp.X is None) ^ (self.X is None):
            return False
        if (exp.Y is None) ^ (self.Y is None):
            return False

        if exp.X is not None and self.X is not None:
            if exp.X.shape != self.X.shape:
                return False
            if not np.all(exp.X == self.X):
                return False
        if exp.Y is not None and self.Y is not None:
            if exp.Y.shape != self.Y.shape:
                return False
            if not np.all(exp.Y == self.Y):
                return False

        return True


@define(kw_only=True)
class Campaign(MSONable):
    """Core executor for running an experiment.

    Parameters
    ----------
    seed : int
        Random number generator seed to ensure reproducibility. The Campaign
        ensures the seed is passed through to all generators that require it
        such that the entire campaign is reproducible.
    """

    seed = field(validator=[instance_of(int), ge(0)])
    experiment = field()
    policy = field()
    save_dir = field(default=None, validator=optional(instance_of(str)))
    data = field(factory=lambda: CampaignData())

    def _log_experiment_prime_data(self, experiment):
        logger.debug(
            f"Experiment: {experiment.name} primed with {self.data.N} data "
            "points"
        )

    @property
    def name(self):
        return f"{self.experiment.name}_{self.policy.name}_{self.seed}"

    def _run(self):
        seed_everything(self.seed)
        logger.debug(f"Seeded {self.name} with seed={self.seed}")

        # The first step is always to prime the data with points selected
        # by some procedure, usually random or Latin Hypercube
        self.data.prime(self.experiment, **self.policy.prime_kwargs)
        self._log_experiment_prime_data(self.experiment)

        # Then the simulation is run
        terminate = False
        step = 0
        while self.data.N < self.policy.n_max and not terminate:
            # The policy returns a state at every step
            # The data is also updated in-place
            with Timer() as timer:
                state = self.policy.step(self.experiment, self.data)
            logger.debug(f"Step {step:03} done in {timer.dt:.02f} s")

            # Determine early-stopping criteria
            terminate = state.terminate

            step += 1

    def run(self):
        with Timer() as timer:
            self._run()
        logger.success(f"[{timer.dt:.02f} s] {self.name}")
        if self.save_dir is not None:
            name = Path(self.save_dir) / f"{self.name}.json"
            self.save(name, json_kwargs={"indent": 4, "sort_keys": True})
            logger.debug(f"{self.name} saved to {name}")

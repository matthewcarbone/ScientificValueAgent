import numpy as np
from monty.json import MSONable

from attrs import field, validators, define

from sva.experiments.base import ExperimentMixin

def assert_inherits_from_ExperimentMixin(klass):
    if not issubclass(klass, ExperimentMixin):
        raise ValueError("Experiment must inherit from ExperimentMixin")

@define
class DualModalityExperiment(MSONable):
    """A multifidelity experient that takes two experiments and runs them in
    tandem.

    Attributes
    ----------
    low, high : Experiment
    """

    low = field()
    @low.validator
    def low_validator(self, attribute, value):
        assert_inherits_from_ExperimentMixin(value)
 
    high = field()
    @high.validator
    def high_validator(self, attribute, value):
        assert_inherits_from_ExperimentMixin(value)

    ratio = field(default=[10, 1], type=list)
    @ratio.validator
    def validate_ratio(self, attribute, value):
        assert all([isinstance(ii, int) for ii in value])

    def __call__(self, x):
        return np.array([self.low(x), self.high(x)])

    def run_gp_experiment(
        self,
        max_experiments,
        svf=None,
        acquisition_function="UCB",
        acquisition_function_kwargs={"beta": 10.0},
        optimize_acqf_kwargs={"q": 1, "num_restarts": 20, "raw_samples": 100},
        pbar=True,
    ):
        # if len(self.data.history) > 0: 
        #     start = self.data.history[-1]["iteration"] + 1
        # else:
        #     start = 0

        # First, check to see if the data is initialized
        if self.low.X is None:
            raise ValueError("You must initialize starting data first for the low-fidelity experiment")

        # Run the experiment
        iteration = 0
        current_counter = 0
        ratio_index = 0
        while iteration < max_experiments: 
            # Get the data
            X = self.data.X
            Y = self.data.Y

            if X.shape[0] > max_experiments:
                break

            # Simple fitting of a Gaussian process
            # using some pretty simple default values for things, which we
            # can always change later
            target = Y if not svf else svf(X, Y)
            if target.ndim > 1 and target.shape[1] > 1:
                raise ValueError("Can only predict on a scalar target")
            if target.ndim == 1:
                target = target.reshape(-1, 1)
            gp = EasySingleTaskGP.from_default(X, target)

            # Should be able to change how the gp is fit here
            gp.fit_mll()

            # Ask the model what to do next
            if acquisition_function in ["EI", "qEI"]:
                acquisition_function_kwargs["best_f"] = Y.max()
            state = ask(
                gp.model,
                acquisition_function,
                bounds=self.properties.experimental_domain,
                acquisition_function_kwargs=acquisition_function_kwargs,
                optimize_acqf_kwargs=optimize_acqf_kwargs,
            )

            # Update the internal data store with the next points
            X2 = state["next_points"]
            self.update_data(X2)

            # Append the history with everything we want to keep
            # Note that the complete state of the GP is saved in the
            # acquisition function model
            self.data.history.append(
                {
                    "iteration": ii,
                    "next_points": state["next_points"],
                    "value": state["value"],
                    "acquisition_function": deepcopy(state["acquisition_function"]),
                    "easy_gp": deepcopy(gp),
                }
            )


import json
import pickle
from collections import defaultdict
from copy import deepcopy
from pathlib import Path

import numpy as np
from attrs import define, field, validators
from joblib import Parallel, delayed
from monty.json import MSONable

from sva.utils import seed_everything

from . import DynamicExperiment
from .base import ExperimentMixin


@define
class PolicyPerformanceEvaluator(MSONable):
    experiment = field()

    @experiment.validator
    def valid_experiment(self, _, value):
        if not issubclass(value.__class__, ExperimentMixin):
            raise ValueError("Provided experiment must inherit ExperimentMixin")

    history = field(factory=list)

    checkpoint_dir = field(
        default=None,
        validator=validators.optional(validators.instance_of((Path, str))),
    )

    def __attrs_post_init__(self):
        if self.checkpoint_dir is not None:
            self.checkpoint_dir = Path(self.checkpoint_dir)
            self.checkpoint_dir.mkdir(exist_ok=True, parents=True)

    @staticmethod
    def _run_job(job):
        """Helper method for multiprocessing in the main function."""

        ckpt_name = job["checkpoint_name"]
        if ckpt_name is not None:
            if Path(ckpt_name).exists():
                return pickle.load(open(ckpt_name, "rb"))

        seed = job["job_seed"]
        if seed is not None:
            seed_everything(seed)
        experiment = job["experiment"]
        n = job["n_steps"]
        acqf = job["acquisition_function"]
        acqf_kwargs = job["acquisition_function_kwargs"]
        optimize_acqf_kwargs = job["optimize_acqf_kwargs"]
        experiment.run(
            n,
            acqf,
            acqf_kwargs,
            optimize_acqf_kwargs=optimize_acqf_kwargs,
            optimize_gp=True,
            pbar=False,
        )
        job["experiment"] = experiment
        if ckpt_name is not None:
            protocol = pickle.HIGHEST_PROTOCOL
            pickle.dump(job, open(ckpt_name, "wb"), protocol=protocol)

        # Required for multiprocessing
        return job

    def _get_name_from_job(self, job):
        acqf_str = str(job["acquisition_function"])
        acqf_kwargs_str = str(job["acquisition_function_kwargs"])
        job_seed_str = str(job["job_seed"])
        return f"{acqf_str}-{acqf_kwargs_str}-{job_seed_str}.pkl"

    def process_results(self, results=None):
        """Processes the saved history (or provided results) into a
        statistical analysis. Results are grouped by the combination of
        acquisition function signature and its keyword arguments (all json-
        serialized).

        Parameters
        ----------
        results : list
            The results to process. If None, uses the results saved in the
            history. Generally, leaving this as None is probably the correct
            way to go.

        Returns
        -------
        dict
        """

        if results is None:
            results = self.history

        tmp0 = defaultdict(list)
        for result in results:
            # Get some string representation of the combination of the
            # acquisition function and its keyword arguments for grouping
            # the results together
            key = json.dumps(
                [
                    result["acquisition_function"],
                    result["acquisition_function_kwargs"],
                ]
            )
            tmp0[key].append(result)

        to_return = {}
        for key, policy_results in tmp0.items():
            tmp2 = []
            for job in policy_results:
                # For each job, get the ground truth "dreamed" experiment
                exp = job["experiment"]

                # Find the x coordinate corresponding to the maximum y-value
                # in the experiment
                x_star = exp.metadata["optima"]["next_points"].numpy()

                # Take that x coordinate and find its corresponding value.
                # Note that we should not take the result directly from the
                # optima "value" key, since this is likely scaled by some
                # output scaler. The following line gets the unscaled value
                # directly.
                y_star, _ = exp(x_star)
                y_star = y_star.squeeze()

                tmp = []

                # For each step in that experiment's history
                for step in exp.history:
                    # Find the x coordinate of the next point, which will
                    # depend on the acquisition function of the experiment
                    x_step_star = step["optimize_gp"]["next_points"].numpy()

                    # Get the corresponding y value
                    y_step_star, _ = exp(x_step_star)
                    y_step_star = y_step_star.squeeze()

                    # Calculate the normalized opportunity cost of this value
                    cost = np.abs(y_star - y_step_star) / np.abs(y_star)
                    tmp.append(cost)
                tmp2.append(tmp)

            to_return[key] = np.array(tmp2)
        return to_return

    def run(
        self,
        n_steps,
        n_dreams,
        acquisition_functions,
        optimize_acqf_kwargs=None,
        n_jobs=12,
        seed=123,
    ):
        """Runs a policy performance evaluation on the experiment provided
        at initialization. Results are saved in the history attribute.

        Parameters
        ----------
        n_steps : int
            the number of steps to take in each simulated experiments (the
            number of experiments to run/new data points to sample).
        n_dreams : int
            the number of samples from the gp fit on the original data to run
            the simulations over.
        acquisition_functions : list
            A list of acquisition functions or signatures to test during the
            campaign. Each entry must contain keys "acqf" and "acqf_kwargs".
        n_jobs : int
            number of parallel multiprocessing jobs to use at a time.
        seed : int, optional
            the seed for the campaign. this is required since multiprocessing
            does not pass the generator state to each of the downstream tasks.
            this is one of the few instances where utils.seed_everything will
            be called internally.
        """

        jobs = []
        existing_names = [job["checkpoint_name"] for job in self.history]
        experiment = deepcopy(self.experiment)

        for _, acqf_value in enumerate(acquisition_functions):
            acqf = acqf_value["acqf"]
            acqf_kwargs = acqf_value["acqf_kwargs"]
            if acqf_kwargs is not None and len(acqf_kwargs) == 0:
                acqf_kwargs = None
            for dream_index in range(n_dreams):
                # Get the current seed and seed the current dreamed experiment
                # if the seed is provided
                job_seed = None
                if seed is not None:
                    job_seed = seed + dream_index
                    seed_everything(job_seed)

                # Get the dreamed experiment
                dream_experiment = DynamicExperiment.from_data(
                    deepcopy(experiment.data.X),
                    deepcopy(experiment.data.Y),
                    experiment.properties.experimental_domain,
                )

                # Setup the job payload
                job = {
                    "job_seed": job_seed,
                    "dream_index": dream_index,
                    "experiment": dream_experiment,
                    "acquisition_function": acqf,
                    "acquisition_function_kwargs": acqf_kwargs,
                    "optimize_acqf_kwargs": optimize_acqf_kwargs,
                    "n_steps": n_steps,
                }
                ckpt_name = None
                if self.checkpoint_dir is not None:
                    ckpt_name = self._get_name_from_job(job)
                    ckpt_name = self.checkpoint_dir / Path(ckpt_name)
                job["checkpoint_name"] = ckpt_name

                # We don't want to repeat already completed jobs
                if job["checkpoint_name"] not in existing_names:
                    jobs.append(job)

        results = Parallel(n_jobs=n_jobs)(
            delayed(self._run_job)(job) for job in jobs
        )
        self.history.extend(results)

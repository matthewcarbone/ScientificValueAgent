import pickle
from pathlib import Path

from attrs import define, field, validators
from joblib import Parallel, delayed
from monty.json import MSONable

from sva.utils import seed_everything

from . import get_dreamed_experiment
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

        ii = job["dream_index"]
        print(f"done with {str(acqf)}, {str(acqf_kwargs)}, {ii}")

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

    def run(
        self,
        n_steps,
        n_dreams,
        acquisition_functions,
        acquisition_function_kwargs,
        optimize_acqf_kwargs=None,
        n_jobs=12,
        seed=123,
    ):
        """
        parameters
        ----------
        n_steps : int
            the number of steps to take in each simulated experiments (the number
            of experiments to run/new data points to sample).
        n_dreams : int
            the number of samples from the gp fit on the original data to run the
            simulations over.
        acquisition_functions : list
            a list of acquisition functions or signatures to test during the
            campaign.
        acquisition_function_kwargs : list
            a list of keyword arguments to pass to the acquisition functions.
        n_jobs : int
            number of parallel multiprocessing jobs to use at a time.
        seed : int, optional
            the seed for the campaign. this is required since multiprocessing
            does not pass the generator state to each of the downstream tasks.
            this is one of the few instances where utils.seed_everything will
            be called internally.
        """

        assert len(acquisition_function_kwargs) == len(acquisition_functions)

        jobs = []
        existing_names = [job["checkpoint_name"] for job in self.history]
        for ii, (acqf, acqf_kwargs) in enumerate(
            zip(acquisition_functions, acquisition_function_kwargs)
        ):
            for dream_index in range(n_dreams):
                dream_experiment = get_dreamed_experiment(
                    self.experiment.data.X,
                    self.experiment.data.Y,
                    self.experiment.properties.experimental_domain,
                )
                job_seed = None
                if seed is not None:
                    job_seed = seed + ii * n_dreams + dream_index

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

from sva import experiments
from sva.truth import sine2phase
from sva.value import default_asymmetric_value_function


def test_sine2phase():
    data = experiments.Data.from_initial_conditions(
        sine2phase.truth_sine2phase,
        value=default_asymmetric_value_function,
        seed=123,
        how="random",
        truth_kwargs=dict(),
        value_kwargs=dict(),
        xmin=0.0,
        xmax=1.0,
        points_per_dimension=3,
        ndim=2,
        n_raster=None,
    )
    exp = experiments.Experiment(
        data=data,
        acqf_signature="botorch.acquisition:UpperConfidenceBound",
        acqf_kwargs=dict(beta=10),
        optimize_acqf_kwargs={"q": 1, "num_restarts": 5, "raw_samples": 20},
        experiment_seed=123,
    )
    exp.run(
        max_n_dat=200,
        pbar=False,
        return_self=False,
        print_at_end=False,
    )
    assert 0.5 < sine2phase.points_in_10_percent_range(exp.data.X) < 0.6

from attrs import define, field

from sva.models.gp import EasySingleTaskGP, get_train_protocol

from .base import ExperimentData, ExperimentMixin, ExperimentProperties
from .campaign import CampaignBaseMixin


@define
class DynamicExperiment(ExperimentMixin, CampaignBaseMixin):
    gp = field(default=None)
    properties = field(default=None)

    @classmethod
    def from_data(
        cls,
        X,
        Y,
        domain,
        train_protocol="fit_mll",
        dream_ppd=20,
        optimize_gp=None,
    ):
        """Creates an Experiment object from data alone. This is done via the
        following steps.

        1. A standard single task GP is fit to the data.
        2. A sample is drawn from that GP.
        3. That sample itself is fit by another GP.
        4. The mean of this GP is now the experiment in question. Running
        experiment(x) will produce the mean of this function as the prediction.

        Parameters
        ----------
        X, Y : np.ndarray
            The input and output data of the experiment. Since this will be
            approximated with a single task GP, Y must be one-dimensional.
        domain : np.ndarray
            The experimental domain of the problem. Must be of shape (2, d),
            where d is the dimensionality of the input.
        train_protocol : dict or str
            The protocol and its keyword arguments. Must be a method defined on
            the EasyGP. For example: {"method": "fit_Adam", "kwargs": None}. If
            only a string is provided, attemps that method with no keyword args.
            This is the training protocol used to fit the dreamed GP.
        dream_ppd : int
            The number of points-per-dimension used in the dreamed GP.
        optimize_gp : dict, optional
            Keyword arguments to pass to BoTorch's acquisition function
            optimizers.

        Returns
        -------
        DynamicExperiment
        """

        train_method, train_kwargs = get_train_protocol(train_protocol)

        gp = EasySingleTaskGP.from_default(X, Y)
        getattr(gp, train_method)(**train_kwargs)

        dreamed_gp = gp.dream(ppd=dream_ppd, domain=domain)
        n_input_dim = dreamed_gp.model.train_inputs[0].shape[1]

        properties = ExperimentProperties(
            n_input_dim=n_input_dim,
            n_output_dim=1,
            valid_domain=None,
            experimental_domain=domain,
        )
        data = ExperimentData(X=X, Y=Y)

        exp = cls(gp=dreamed_gp, properties=properties, data=data)

        if optimize_gp is None:
            optimize_gp_kwargs = {"num_restarts": 150, "raw_samples": 150}
        exp.metadata["optima"] = exp.gp.optimize(
            domain=domain, **optimize_gp_kwargs
        )
        return exp

    def _truth(self, x):
        mu, _ = self.gp.predict(x)
        return mu.reshape(-1, 1)

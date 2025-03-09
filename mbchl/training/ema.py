# Copyright (c) 2019 Samuel G. Fadel
# MIT License
# https://github.com/fadel/pytorch_ema

# Copyright (c) 2024 Philippe Gonzalez
# Apache License Version 2.0

import os
import warnings

import numpy as np
import torch

from ..utils import Registry

EMARegistry = Registry("ema")


class BaseEMA:
    """Base class for exponential moving average (EMA) objects."""

    @property
    def params(self):
        """Get the non-averaged model parameters."""
        return self.model.parameters()

    def update(self, ema_params, beta):
        """Update the averaged parameters.

        Parameters
        ----------
        ema_params : list[torch.Tensor]
            Averaged parameters.
        beta : float
            EMA decay factor or momentum.

        """
        with torch.no_grad():
            for param, ema_param in zip(self.params, ema_params):
                ema_param += (1 - beta) * (param - ema_param)

    def store(self):
        """Store a temporary copy of the non-averaged model parameters."""
        self.stored_params = [p.clone() for p in self.params]

    def restore(self):
        """Copy a previously stored copy of parameters back to the model."""
        if self.stored_params is None:
            raise RuntimeError("no stored parameters")
        for param, stored_param in zip(self.params, self.stored_params):
            param.data.copy_(stored_param.data)
        self.stored_params = None

    def apply(self, ema_params):
        """Copy the averaged parameters to the model.

        Parameters
        ----------
        ema_params : list[torch.Tensor]
            Averaged parameters.

        """
        for param, ema_param in zip(self.params, ema_params):
            param.data.copy_(ema_param.data)

    def state_dict(self):
        """Get the state dict of the EMA object."""
        return {attr: getattr(self, attr) for attr in self._state_dict_attrs}

    def load_state_dict(self, state_dict):
        """Load the state dict of the EMA object."""
        assert set(state_dict) == set(self._state_dict_attrs)
        for attr, value in state_dict.items():
            setattr(self, attr, value)


@EMARegistry.register("classic")
class EMA(BaseEMA):
    r"""Traditional exponential moving average (EMA).

    The update rule is

    .. math::

        \theta_{t} = \beta \theta_{t-1} + (1 - \beta) \theta_{t}

    where :math:`\theta_{t}` are the averaged parameters at time :math:`t`.

    Parameters
    ----------
    model : torch.nn.Module
        Model to apply EMA to.
    beta : float, optional
        EMA decay factor or momentum.

    """

    def __init__(self, model, beta=0.999):
        assert 0.0 < beta < 1.0
        self.model = model
        self.beta = beta
        self.ema_params = [p.clone().detach() for p in model.parameters()]
        self.stored_params = None
        self._state_dict_attrs = ["ema_params"]

    def update(self):
        """Update the averaged parameters."""
        super().update(self.ema_params, self.beta)

    def apply(self):
        """Copy the averaged parameters to the model."""
        super().apply(self.ema_params)


@EMARegistry.register("karras")
class EMAKarras(BaseEMA):
    """Post-hoc exponential moving average (EMA).

    Proposed in [1].

    .. [1] T. Karras, M. Aittala, J. Lehtinen, J. Hellsten, T. Aila and S. Laine,
       "Analyzing and Improving the Training Dynamics of Diffusion Models", in Proc.
       CVPR, 2024.

    Parameters
    ----------
    model : torch.nn.Module
        Model to apply EMA to.
    sigma_rels : list[float], optional
        Standard deviation of the profiles to track. Each value can be seen as the
        "width" of the peak of the profile relative to training time. Between 0 and 1.
    sigma_rel_apply : float, optional
        Standard deviation of the profile to apply to the model when calling
        :meth:`apply`. Useful during validation. Must be in ``sigma_rels``. If ``None``,
        the first value in ``sigma_rels`` is used.

    """

    def __init__(self, model, sigma_rels=[0.05, 0.1], sigma_rel_apply=None):
        assert all(0.0 < sigma_rel < 1.0 for sigma_rel in sigma_rels)
        self.model = model
        self.sigma_rels = sigma_rels
        self.sigma_rel_apply = None
        self.ema_params = {
            sigma_rel: [p.clone().detach() for p in model.parameters()]
            for sigma_rel in sigma_rels
        }
        self.stored_params = None
        self._num_updates = 0
        self._gammas = {
            sigma_rel: self._sigma_rel_to_gamma(sigma_rel) for sigma_rel in sigma_rels
        }
        self._state_dict_attrs = ["ema_params", "_num_updates", "_gammas"]

    def update(self):
        """Update the averaged parameters."""
        self._num_updates += 1
        for sigma_rel in self.sigma_rels:
            ema_params = self.ema_params[sigma_rel]
            gamma = self._gammas[sigma_rel]
            beta = (1 - 1 / self._num_updates) ** (gamma + 1)
            super().update(ema_params, beta)

    def apply(self):
        """Copy the averaged parameters to the model.

        This copies the parameters that were averaged using the ``sigma_rel_apply``
        profile. To copy parameters synthesized using arbitrary post-hoc profiles, use
        the :meth:`post_hoc_ema` method.

        """
        if self.sigma_rel_apply is None:
            warnings.warn("no sigma_rel_apply provided, using first value")
            sigma_rel = self.sigma_rels[0]
        else:
            sigma_rel = self.sigma_rel_apply
        ema_params = self.ema_params[sigma_rel]
        super().apply(ema_params)

    @staticmethod
    def _sigma_rel_to_gamma(sigma_rel):
        """Algorithm 2 in [1].

        .. [1] T. Karras, M. Aittala, J. Lehtinen, J. Hellsten, T. Aila and S. Laine,
           "Analyzing and Improving the Training Dynamics of Diffusion Models", in Proc.
           CVPR, 2024.
        """
        t = sigma_rel**-2
        gamma = np.roots([1, 7, 16 - t, 12 - t]).real.max()
        return gamma.item()

    @staticmethod
    def _solve_weights(t_i, gamma_i, t_r, gamma_r):
        """Algorithm 3 in [1].

        .. [1] T. Karras, M. Aittala, J. Lehtinen, J. Hellsten, T. Aila and S. Laine,
           "Analyzing and Improving the Training Dynamics of Diffusion Models", in Proc.
           CVPR, 2024.
        """

        def p_dot_p(t_a, gamma_a, t_b, gamma_b):
            t_ratio = t_a / t_b
            t_exp = np.where(t_a < t_b, gamma_b, -gamma_a)
            t_max = np.maximum(t_a, t_b)
            num = (gamma_a + 1) * (gamma_b + 1) * t_ratio**t_exp
            den = (gamma_a + gamma_b + 1) * t_max
            return num / den

        rv = lambda x: np.float64(x).reshape(-1, 1)  # noqa: E731
        cv = lambda x: np.float64(x).reshape(1, -1)  # noqa: E731
        A = p_dot_p(rv(t_i), rv(gamma_i), cv(t_i), cv(gamma_i))
        B = p_dot_p(rv(t_i), rv(gamma_i), cv(t_r), cv(gamma_r))
        X = np.linalg.solve(A, B)
        return X

    def post_hoc_ema(
        self,
        ckpts_or_ckpt_dir,
        sigma_rel_r,
        t_r=None,
        ext=".ckpt",
        state_dict_key=None,
        apply=True,
        map_location=None,
    ):
        """Apply arbitrary profiles to model parameters from checkpoints.

        Parameters
        ----------
        ckpts_or_ckpt_dir : str or list[str]
            Path to the checkpoint directory or list of checkpoint paths.
        sigma_rel_r : float or list[float]
            Standard deviation of the profiles to synthesize. Between 0 and 1.
        t_r : int or list[int], optional
            Target update step for each profile. If ``None``, the latest update step
            is used.
        ext : str, optional
            Checkpoint file ext.
        state_dict_key : str, optional
            Key to access the EMA state dict within each checkpoint file. If ``None``,
            the entire state dict is assumed to be the EMA state.
        apply : bool, optional
            Whether to apply the profile to the model. If ``True``, then ``sigma_rel_r``
            must be a single value, since the profile to apply to the model would be
            ambiguous otherwise.
        map_location : callable or torch.device or str or dict, optional
            Where to load tensors. Passed to ``torch.load``.

        Returns
        -------
        list[list[torch.Tensor]] or list[torch.Tensor]
            Averaged parameters for each profile. Same length as ``sigma_rel_r`` if
            ``sigma_rel_r`` is a list, otherwise a single list of averaged parameters.

        """
        if isinstance(ckpts_or_ckpt_dir, str):
            ckpts = [
                os.path.join(ckpts_or_ckpt_dir, f)
                for f in os.listdir(ckpts_or_ckpt_dir)
                if f.endswith(ext)
            ]
            if not ckpts:
                raise ValueError(f"no {ext} file in {ckpts_or_ckpt_dir}")
        else:
            ckpts = ckpts_or_ckpt_dir

        sigma_rel_r_was_list = isinstance(sigma_rel_r, list)
        t_r_was_list = isinstance(t_r, list)

        if not sigma_rel_r_was_list:
            if t_r_was_list:
                sigma_rel_r = [sigma_rel_r] * len(t_r)
            else:
                sigma_rel_r = [sigma_rel_r]
        if not all(isinstance(s, float) for s in sigma_rel_r):
            raise TypeError("sigma_rel_r must be a float or a list of floats")
        if not all(0.0 < s < 1.0 for s in sigma_rel_r):
            raise ValueError("sigma_rel_r values must be strictly in [0, 1]")

        if t_r is not None:
            if not t_r_was_list:
                if sigma_rel_r_was_list:
                    t_r = [t_r] * len(sigma_rel_r)
                else:
                    t_r = [t_r]
            if not all(isinstance(t, int) for t in t_r):
                raise TypeError("t_r must be an int or a list of ints")
            if len(t_r) != len(sigma_rel_r):
                raise ValueError("gamma_r and t_r must have the same length")

        if apply and len(sigma_rel_r) > 1:
            raise ValueError("cannot apply multiple EMA profiles to the model")

        ema_params = []
        t_i = []
        gamma_i = []
        _loaded_num_updates = set()

        for ckpt in ckpts:
            state_dict = torch.load(ckpt, map_location=map_location, weights_only=True)

            if state_dict_key is not None:
                if state_dict_key in state_dict:
                    state_dict = state_dict[state_dict_key]
                else:
                    raise ValueError(f"no '{state_dict_key}' key in {ckpt}")

            if state_dict["_num_updates"] in _loaded_num_updates:
                warnings.warn(
                    "Found multiple checkpoints saved at the same number of updates. "
                    "Only one will be used to prevent unstable post-hoc EMA "
                    "reconstruction."
                )
                continue
            _loaded_num_updates.add(state_dict["_num_updates"])

            for sigma_rel in self.sigma_rels:
                if sigma_rel not in state_dict["ema_params"]:
                    raise ValueError(
                        "no averaged parameters for " f"sigma_rel={sigma_rel} in {ckpt}"
                    )

                ema = state_dict["ema_params"][sigma_rel]
                ema_params.append(ema)
                t_i.append(state_dict["_num_updates"])
                gamma_i.append(state_dict["_gammas"][sigma_rel])

        if t_r is None:
            t_r = [max(t_i)] * len(sigma_rel_r)

        gamma_r = [self._sigma_rel_to_gamma(s) for s in sigma_rel_r]
        X = self._solve_weights(t_i, gamma_i, t_r, gamma_r)

        with torch.no_grad():
            ema_params = [
                [
                    sum(x.item() * param for x, param in zip(X[:, i], params))
                    for params in zip(*ema_params)
                ]
                for i in range(X.shape[1])
            ]

        if apply:
            assert len(ema_params) == 1
            super().apply(ema_params[0])

        if not sigma_rel_r_was_list and not t_r_was_list:
            ema_params = ema_params[0]

        return ema_params

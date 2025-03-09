import math
from typing import override

import numpy as np
import torch
from scipy.special import expi
from scipy.stats import norm

from ...utils import Registry

SDERegistry = Registry("sde")


class _BaseSDE:
    def probability_flow(self, x, y, score, t):
        """Compute the derivative using the  probability flow equation."""
        return self.f(x, y, t) - 0.5 * self.g(t) ** 2 * score

    def reverse_step(self, x, y, score, t, dt):
        """Take one step in reverse time."""
        noise = self.g(t) * (-dt) ** 0.5 * torch.randn(x.shape, device=x.device)
        return (self.f(x, y, t) - self.g(t) ** 2 * score) * dt + noise

    def prior(self, y):
        """Sample from the prior distribution."""
        t = torch.tensor(1, device=y.device)
        sigma = self.s(t) * self.sigma(t)
        return y + sigma * torch.randn_like(y)

    def s(self, t):
        """Compute the scaling factor."""
        raise NotImplementedError

    def sigma(self, t):
        """Compute the standard deviation."""
        raise NotImplementedError

    def f(self, x, y, t):
        """Compute the drift coefficient."""
        raise NotImplementedError

    def g(self, t):
        """Compute the diffusion coefficient."""
        raise NotImplementedError

    def sigma_inv(self, t):
        """Compute the inverse function of the standard deviation."""
        raise NotImplementedError


class _BaseOUVESDE(_BaseSDE):
    def __init__(
        self,
        stiffness=1.5,
        sigma_min=0.05,
        sigma_max=0.5,
    ):
        self.stiffness = stiffness
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self._sigma_p = sigma_max / sigma_min
        self._log_sigma_p = math.log(sigma_max / sigma_min)

    def s(self, t):
        return (-self.stiffness * t).exp()

    def f(self, x, y, t):
        return self.stiffness * (y - x)


@SDERegistry.register("richter-ouve")
class RichterOUVESDE(_BaseOUVESDE):
    """Ornstein-Uhlenbeck Variance Exploding SDE.

    Proposed in [1].

    .. [1] S. Welker, J. Richter and T. Gerkmann, "Speech Enhancement with Score-Based
       Generative Models in the Complex STFT Domain", in Proc. INTERSPEECH, 2022.
    """

    @override
    def sigma(self, t):
        return (
            self.sigma_min
            * (
                ((self._sigma_p**t / self.s(t)) ** 2 - 1)
                / (1 + self.stiffness / self._log_sigma_p)
            )
            ** 0.5
        )

    @override
    def g(self, t):
        return self.sigma_min * self._sigma_p**t * (2 * self._log_sigma_p) ** 0.5

    @override
    def sigma_inv(self, sigma):
        return (
            0.5
            * (
                1
                + (1 + self.stiffness / self._log_sigma_p)
                * (sigma / self.sigma_min) ** 2
            ).log()
            / (self.stiffness + self._log_sigma_p)
        )


@SDERegistry.register("brever-ouve")
class BreverOUVESDE(_BaseOUVESDE):
    """Alternative Ornstein-Uhlenbeck Variance Exploding SDE formulation.

    Proposed in [1].

    .. [1] P. Gonzalez, Z.-H. Tan, J. Østergaard, J. Jensen, T. S. Alstrøm and T. May,
       "Investigating the Design Space of Diffusion Models for Speech Enhancement", in
       IEEE/ACM Trans. Audio, Speech, Lang. Process., 2024.
    """

    @override
    def sigma(self, t):
        return self.sigma_min * (self._sigma_p ** (2 * t) - 1) ** 0.5

    @override
    def g(self, t):
        return (
            self.s(t)
            * self.sigma_min
            * self._sigma_p**t
            * (2 * self._log_sigma_p) ** 0.5
        )

    @override
    def sigma_inv(self, sigma):
        return 0.5 * ((sigma / self.sigma_min) ** 2 + 1).log() / self._log_sigma_p


class _BaseVPSDE(_BaseSDE):
    def s(self, t):
        return (-self.stiffness * t).exp() / (1 + self.sigma(t) ** 2) ** 0.5

    def f(self, x, y, t):
        return (self.stiffness + 0.5 * self.beta(t)) * (y - x)

    def g(self, t):
        return (-self.stiffness * t).exp() * self.beta(t) ** 0.5


@SDERegistry.register("brever-ouvp")
class BreverOUVPSDE(_BaseVPSDE):
    """Ornstein-Uhlenbeck Variance Preserving SDE.

    Proposed in [1].

    .. [1] P. Gonzalez, Z.-H. Tan, J. Østergaard, J. Jensen, T. S. Alstrøm and T. May,
       "Investigating the Design Space of Diffusion Models for Speech Enhancement", in
       IEEE/ACM Trans. Audio, Speech, Lang. Process., 2024.
    """

    def __init__(
        self,
        stiffness=1.5,
        beta_min=0.01,
        beta_max=1.0,
    ):
        self.stiffness = stiffness
        self.beta_min = beta_min
        self.beta_max = beta_max
        self._beta_d = beta_max - beta_min

    def beta(self, t):
        """Compute the beta coefficient."""
        return self.beta_min + self._beta_d * t

    @override
    def sigma(self, t):
        return ((0.5 * self._beta_d * t**2 + self.beta_min * t).exp() - 1) ** 0.5

    @override
    def sigma_inv(self, sigma):
        return (
            (self.beta_min**2 + 2 * self._beta_d * (sigma**2 + 1).log()) ** 0.5
            - self.beta_min
        ) / self._beta_d


@SDERegistry.register("brever-oucosine")
class BreverOUCosineSDE(_BaseVPSDE):
    """Ornstein-Uhlenbeck shifted-cosine SDE.

    A shifted-cosine schedule with an additional Ornstein-Uhlenbeck drift term.
    """

    def __init__(
        self,
        stiffness=0.0,
        lambda_min=-12.0,
        lambda_max=float("inf"),
        shift=0.0,
        beta_clamp=10.0,
    ):
        self.stiffness = stiffness
        self.shift = shift
        self.lambda_min = lambda_min + shift
        self.lambda_max = lambda_max + shift
        self.t_min = self.lambda_inv(self.lambda_min)
        self.t_max = self.lambda_inv(self.lambda_max)
        self.t_d = self.t_min - self.t_max
        self.beta_clamp = beta_clamp

    def lambda_(self, t):
        """Compute the signal-to-noise ratio."""
        return -2 * (math.pi * t / 2).tan().log() + self.shift

    def lambda_inv(self, lambda_):
        """Compute the inverse function of the signal-to-noise ratio."""
        if isinstance(lambda_, torch.Tensor):
            return 2 / math.pi * ((-lambda_ + self.shift) / 2).exp().atan()
        else:
            return 2 / math.pi * math.atan(math.exp((-lambda_ + self.shift) / 2))

    def lambda_tilde(self, t):
        """Compute the truncated signal-to-noise ratio."""
        return self.lambda_(self.t_max + self.t_d * t)

    def lambda_tilde_inv(self, lambda_):
        """Compute the inverse function of the truncated signal-to-noise ratio."""
        return (self.lambda_inv(lambda_) - self.t_max) / self.t_d

    def beta(self, t):
        """Compute the beta coefficient."""
        pi_t_half = math.pi * (self.t_max + self.t_d * t) / 2
        return (
            math.pi
            * self.t_d
            / pi_t_half.cos() ** 2
            * pi_t_half.tan()
            / (math.exp(self.shift) + pi_t_half.tan() ** 2)
        ).clamp(max=self.beta_clamp)

    @override
    def sigma(self, t):
        return (-self.lambda_tilde(t) / 2).exp()

    @override
    def sigma_inv(self, sigma):
        return self.lambda_tilde_inv(-2 * sigma.log())


class _BaseBBSDE(_BaseSDE):
    def clamp(self, t):
        return t * self.t_max

    def s(self, t):
        return 1 - self.clamp(t)

    def f(self, x, y, t):
        return (y - x) / (1 - self.clamp(t))


@SDERegistry.register("bbed")
class BBEDSDE(_BaseBBSDE):
    """Brownian Bridge with Exponential Diffusion coefficient SDE.

    Proposed in [1].

    .. [1] B. Lay, S. Welker, J. Richter and T. Gerkmann, "Reducing the Prior Mismatch
       of Stochastic Differential Equations for Diffusion-Based Speech Enhancement", in
       Proc. INTERSPEECH, 2023.
    """

    def __init__(self, scaling=0.1, k=10.0, t_max=0.999):
        self.scaling = scaling
        self.t_max = t_max
        self.k = k
        self._k2 = k**2
        self._logk2 = 2 * math.log(k)

    @override
    def g(self, t):
        t = self.clamp(t)
        return self.scaling * self.k**t

    @override
    def sigma(self, t):
        t = self.clamp(t)
        return (
            self.scaling
            * (
                self._k2
                * self._logk2
                * (expi((t.cpu() - 1) * self._logk2).to(t.device) - expi(-self._logk2))
                - self._k2**t / (t - 1)
                - 1
            )
            ** 0.5
        )


@SDERegistry.register("bbcd")
class BBCD(_BaseBBSDE):
    """Brownian Bridge with Constant Diffusion coefficient."""

    def __init__(self, scaling=0.1, t_max=0.999):
        self.scaling = scaling
        self.t_max = t_max

    @override
    def g(self, t):
        return self.scaling

    @override
    def sigma(self, t):
        t = self.clamp(t)
        return self.scaling * (t / (1 - t)) ** 0.5

    @override
    def sigma_inv(self, sigma):
        return sigma**2 / (self.scaling**2 + sigma**2) / self.t_max


@SDERegistry.register("bbls")
class BBLS(_BaseBBSDE):
    """Brownian Bridge with Linear Standard deviation."""

    def __init__(self, scaling=0.1, t_max=0.999):
        self.scaling = scaling
        self.t_max = t_max

    @override
    def g(self, t):
        t = self.clamp(t)
        return self.scaling * (1 - t) * (2 * t) ** 0.5

    @override
    def sigma(self, t):
        t = self.clamp(t)
        return self.scaling * t

    @override
    def sigma_inv(self, sigma):
        return sigma / (self.scaling * self.t_max)


@SDERegistry.register("edm-training")
class EDMTrainingSDE(_BaseSDE):
    """Normal log-sigma distribution.

    Proposed in [1].

    .. [1] T. Karras, M. Aittala, T. Aila and S. Laine, "Elucidating the Design Space of
       Diffusion-Based Generative Models", in Proc. NeurIPS, 2022.
    """

    def __init__(self, P_mean=-1.2, P_std=1.2, t_max=0.999):
        self.loc = -2 * P_mean
        self.scale = 2 * P_std
        self.t_max = t_max

    def lambda_(self, t):
        """Compute the signal-to-noise ratio."""
        t = t * self.t_max
        out = norm.ppf(1 - t.cpu(), loc=self.loc, scale=self.scale)
        out = torch.as_tensor(out).to(t.device, t.dtype)
        return out

    def lambda_inv(self, lambda_):
        """Compute the inverse function of the signal-to-noise ratio."""
        out = 1 - norm.cdf(lambda_.cpu(), loc=self.loc, scale=self.scale)
        out = torch.as_tensor(out).to(lambda_.device, lambda_.dtype)
        return out / self.t_max

    def lambda_prime(self, t):
        """Compute the derivative of the signal-to-noise ratio."""
        t = t * self.t_max
        out = -1 / norm.pdf(self.lambda_(t).cpu(), loc=self.loc, scale=self.scale)
        out = torch.as_tensor(out).to(t.device, t.dtype)
        return out

    @override
    def s(self, t):
        return torch.ones_like(t)

    @override
    def f(self, x, y, t):
        return torch.zeros_like(x)

    @override
    def g(self, t):
        with np.errstate(divide="ignore"):
            out = (-self.lambda_prime(t) * (-self.lambda_(t)).exp()) ** 0.5
        out[t == 0] = 0
        return out

    @override
    def sigma(self, t):
        return (-self.lambda_(t) / 2).exp()

    @override
    def sigma_inv(self, sigma):
        return self.lambda_inv(-2 * sigma.log())


@SDERegistry.register("edm-sampling")
class EDMSamplingSDE(_BaseSDE):
    """Polynomial warp schedule.

    Proposed in [1].

    .. [1] T. Karras, M. Aittala, T. Aila and S. Laine, "Elucidating the Design Space of
       Diffusion-Based Generative Models", in Proc. NeurIPS, 2022.
    """

    def __init__(self, sigma_min=0.002, sigma_max=80, rho=7):
        self.rho = rho
        self._a = sigma_min ** (1 / rho)
        self._b = sigma_max ** (1 / rho) - sigma_min ** (1 / rho)

    @override
    def s(self, t):
        return torch.ones_like(t)

    @override
    def f(self, x, y, t):
        return torch.zeros_like(x)

    @override
    def g(self, t):
        return (
            2 * self.rho * self._b * (self._a + t * self._b) ** (2 * self.rho - 1)
        ) ** 0.5

    @override
    def sigma(self, t):
        return (self._a + t * self._b) ** self.rho

    @override
    def sigma_inv(self, sigma):
        return (sigma ** (1 / self.rho) - self._a) / self._b

import math

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchcomp.core import compressor_core

from ..utils import Registry, linear_interpolation
from .filters import FilterbankRegistry, IIRFilterbank, butter

IHCRegistry = Registry("rectification")
AdaptationRegistry = Registry("adaptation")
IntegrationRegistry = Registry("integration")

class AuditoryModel(nn.Module):
    """Auditory model."""

    def __init__(
        self,
        fs=20000,
        filterbank="gammatone",
        filterbank_kw=None,
        ihc="hwrlp",
        ihc_kw=None,
        adaptation="log",
        adaptation_kw=None,
        integration="none",
        integration_kw=None,
        modulation="mfbtd",
        modulation_kw=None,
        output_scale=1.0,
    ):
        super().__init__()

        filterbank_kw = filterbank_kw or {}
        ihc_kw = ihc_kw or {}
        adaptation_kw = adaptation_kw or {}
        integration_kw = integration_kw or {}
        modulation_kw = modulation_kw or {}
        for kw, name in zip(
            [
                filterbank_kw,
                ihc_kw,
                adaptation_kw,
                integration_kw,
                modulation_kw,
            ],
            [
                "filterbank_kw",
                "ihc_kw",
                "adaptation_kw",
                "integration_kw",
                "modulation_kw",
            ],
        ):
            if "fs" not in kw:
                kw["fs"] = fs
            elif fs != kw["fs"]:
                raise ValueError(f"fs argument does not match {name}['fs']")

        self.filterbank = FilterbankRegistry.get(filterbank)(**filterbank_kw)
        self.ihc = IHCRegistry.get(ihc)(**ihc_kw)
        self.adaptation = AdaptationRegistry.get(adaptation)(**adaptation_kw)
        self.frame_average = IntegrationRegistry.get(integration)(**integration_kw)
        self.modulation = FilterbankRegistry.get(modulation)(**modulation_kw)
        self.modulation_name = modulation
        self.output_scale = output_scale

    def forward(self, x, audiogram=None):
        """Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Input signal. Shape ``(batch_size, ..., time)``.
        audiogram : torch.Tensor
            Audiogram. Shape ``(batch_size, n_thresholds, 2)``. First column is
            frequency in Hz, second column is hearing loss in dB.

        Returns
        -------
        torch.Tensor
            Auditory model output.

        """
        if audiogram is None:
            ohc_loss, ihc_loss = None, None
        else:
            if audiogram.shape[-1] != 2:
                raise ValueError(
                    "audiogram dimension along last axis must be 2, "
                    f"got {audiogram.shape}"
                )
            if x.ndim == 1 and audiogram.ndim != 2:
                raise ValueError(
                    "audiogram must be 2D for 1D input, "
                    f"got {audiogram.shape} audiogram and {x.shape} input"
                )
            if x.ndim > 1 and (audiogram.ndim != 3 or audiogram.shape[0] != x.shape[0]):
                raise ValueError(
                    "audiogram must be 3D with same batch size as input for batched "
                    f"inputs, got {audiogram.shape} audiogram and {x.shape} input"
                )
            ohc_loss, ihc_loss = audiogram_to_ohc_ihc_loss(
                audiogram, freqs=self.filterbank.fc
            )
        x = self.filterbank(x, ohc_loss=ohc_loss)  # (batch_size, time) (32, 64000)
        x = self.ihc(x)  # (batch_size, n_filters, time) (32, 31, 64000)
        if ihc_loss is not None:
            ihc_loss = ihc_loss.unsqueeze(-1)  # (batch_size, n_filters, 1)
            while ihc_loss.ndim < x.ndim:
                ihc_loss = ihc_loss.unsqueeze(1)  # (batch_size, ..., n_filters, 1)
            x = x * ihc_loss
        x = self.adaptation(x)  # (batch_size, n_filters, time) (32, 31, 64000)
        if self.frame_average is not None:
            x = self.frame_average(x)  # (batch_size, n_filters, time) (32, 31, 500) and
        if self.modulation_name == 'lphp_lfilter':  # Haven't determined the 'lphp' modulation yet
                x, x_speech_mod, x_env_mod = self.modulation(x)
                return x * self.output_scale, x_speech_mod * self.output_scale, x_env_mod * self.output_scale
        else:
            x = self.modulation(x)
            return x * self.output_scale


@IHCRegistry.register("hwrlp")
class HalfwaveRectificationLowpassIHC(nn.Module):
    """Half-wave rectification followed by low-pass filtering."""

    def __init__(
        self,
        fc=1000,
        fs=20000,
        order=2,
        dtype=torch.float32,
    ):
        super().__init__()
        b, a = butter(order, 2 * fc / fs)
        self.lp = IIRFilterbank(b, a, dtype=dtype)

    def forward(self, x):
        """Forward pass."""
        x = x.relu()
        x = self.lp(x)
        x = x.relu()
        return x


@IHCRegistry.register("hwr")
class HalfwaveRectificationIHC(nn.Module):
    """Half-wave rectification."""

    def __init__(self, *args, **kwargs):
        super().__init__()

    def forward(self, x):
        """Forward pass."""
        x = x.relu()
        return x


@IHCRegistry.register("none")
class NoIHC(nn.Module):
    """Identity inner hair cell."""

    def __init__(self, *args, **kwargs):
        super().__init__()

    def forward(self, x):
        """Return input unchanged."""
        return x


@AdaptationRegistry.register("log")
class LogAdaptation(nn.Module):
    """Instantaneous logarithmic adaptation."""

    def __init__(self, thr=1e-5, **kwargs):
        super().__init__()
        self.thr = thr

    def forward(self, x):
        """Forward pass."""
        assert x.ge(0).all()
        return x.div(self.thr).log1p()


@AdaptationRegistry.register("logdrc")
class LogDynamicRangeCompressionAdaptation(nn.Module):
    """Logarithmic dynamic range compression adaptation."""

    def __init__(self, fs, tau=0.1, thr=1e-5):
        super().__init__()
        self.alpha = 1 - math.exp(-1 / (fs * tau))
        self.thr = thr

    def forward(self, x):
        """Forward pass."""
        assert x.ge(0).all()
        inshape = x.shape
        x = x.reshape(-1, x.shape[-1])
        x_thr = x.clamp(self.thr).div(self.thr)
        f = x_thr.log() / x_thr.sub(1)
        f = f.nan_to_num(nan=1.0)
        alpha = x.new_full((x.shape[0],), self.alpha)
        zi = x.new_ones(x.shape[0])
        g = compressor_core(f, zi, alpha, alpha)
        return x.mul(g).reshape(inshape) / self.thr


@AdaptationRegistry.register("none")
class NoAdaptation(nn.Module):
    """Identity adaptation."""

    def __init__(self, *args, **kwargs):
        super().__init__()

    def forward(self, x):
        """Return input unchanged."""
        return x

@AdaptationRegistry.register("pcen")
class PCENAdaptation_EMA(nn.Module):
    """Per-Channel Energy Normalization (PCEN) adaptation with Exponential Moving Average of the input signal.

    Compatible with (batch_size, n_filters, time) shaped inputs.
    """

    def __init__(self, 
                 fs=16000,               # 采样率，不太用得上但保持统一
                 alpha=0.98, 
                 delta=2.0, 
                 r=0.5, 
                 eps=1e-6, 
                 smoothing_coef=0.025, 
                 learnable=True):
        """
        Initialize PCEN adaptation module.

        Parameters
        ----------
        fs : float
            Sampling rate (Hz).
        alpha : float
            Strength of normalization (default 0.98).
        delta : float
            Bias term to avoid division issues (default 2.0).
        r : float
            Root compression factor (default 0.5).
        eps : float
            Small constant to prevent division by zero (default 1e-6).
        smoothing_coef : float
            Smoothing coefficient for EMA (default 0.025).
        learnable : bool
            Whether alpha, delta, r are learnable parameters.
        """
        super().__init__()
        self.eps = float(eps)
        self.smoothing_coef = smoothing_coef

        if learnable:
            self.alpha = nn.Parameter(torch.tensor(alpha))
            self.delta = nn.Parameter(torch.tensor(delta))
            self.r = nn.Parameter(torch.tensor(r))
        else:
            self.register_buffer('alpha', torch.tensor(alpha))
            self.register_buffer('delta', torch.tensor(delta))
            self.register_buffer('r', torch.tensor(r))

    def forward(self, x):
        """
        Forward pass of PCEN adaptation.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor, shape (batch_size, n_filters, time).

        Returns
        -------
        torch.Tensor
            PCEN adapted output, same shape as input.
        """
        B, C, T = x.shape
        x = x.clamp(min=0.0)

        # Initialize M(t) for each (B, C)
        M = torch.zeros(B, C, device=x.device)

        outputs = []

        for t in range(T):
            # Exponential Moving Average update
            M = (1 - self.smoothing_coef) * M + self.smoothing_coef * x[:, :, t]
            # PCEN normalization and compression
            norm = x[:, :, t] / (self.eps + M).pow(self.alpha)
            norm = (norm + self.delta).pow(self.r) - self.delta.pow(self.r)
            outputs.append(norm.unsqueeze(-1))  # (B, C, 1)

        y = torch.cat(outputs, dim=-1)  # (B, C, T)
        return y



@AdaptationRegistry.register("pcen_res")
class PCENAdaptation_Res(nn.Module):
    """Per-Channel Energy Normalization (PCEN) adaptation with recursive exponential smoothing of the input signal.

    Efficient cumulative smoothing version.
    """

    def __init__(self, 
                 fs=16000,
                 alpha=0.98,
                 delta=2.0,
                 r=0.5,
                 eps=1e-6,
                 smoothing_coef=0.025,
                 learnable=True):
        super().__init__()
        self.eps = float(eps)

        if learnable:
            self.alpha = nn.Parameter(torch.tensor(alpha))
            self.delta = nn.Parameter(torch.tensor(delta))
            self.r = nn.Parameter(torch.tensor(r))
            self.smoothing_coef = nn.Parameter(torch.tensor(smoothing_coef))
        else:
            self.register_buffer('alpha', torch.tensor(alpha))
            self.register_buffer('delta', torch.tensor(delta))
            self.register_buffer('r', torch.tensor(r))
            self.register_buffer('smoothing_coef', torch.tensor(smoothing_coef))

    def forward(self, x):
        """
        Parameters

        ----------
        x : torch.Tensor
            Input tensor, shape (batch_size, n_filters, time).

        Returns
        -------
        torch.Tensor
            PCEN adapted output, same shape as input.
        """
        B, C, T = x.shape
        x = x.clamp(min=0.0)

        # # Cumulative smoothing (fast)
        # M = []
        # M_t = x[:, :, 0]  # Initialize with first frame
        # M.append(M_t.unsqueeze(-1))

        # for t in range(1, T):
        #     M_t = (1 - self.smoothing_coef) * M_t + self.smoothing_coef * x[:, :, t]
        #     M.append(M_t.unsqueeze(-1))

        # M = torch.cat(M, dim=-1)  # (B, C, T)
        # Precompute decay powers


        smoothing = self.smoothing_coef.clamp(min=1e-5)
        one_minus_s = 1.0 - smoothing
        decay_factors = torch.pow(one_minus_s, torch.arange(T, device=x.device)).view(1, 1, T)

        # Compute cumulative sum for efficient EMA
        x_cumsum = torch.cumsum(x * decay_factors, dim=-1)

        # Normalize by cumulative weights
        weights_cumsum = torch.cumsum(decay_factors.expand(B, C, T), dim=-1)

        # Compute smoothed M (EMA approximation)
        M = x_cumsum / (weights_cumsum + self.eps)

        # PCEN normalization and compression
        norm = x / (self.eps + M).pow(self.alpha)
        y = (norm + self.delta).pow(self.r) - self.delta.pow(self.r)

        return y


@AdaptationRegistry.register("pcen_leaf")
class PCENLeafAdaptation(nn.Module):
    """

    LEAF-style Per-Channel Energy Normalization (sPCEN) adaptation.

    Compatible with (batch_size, n_filters, time) shaped inputs.
    Each filter channel has its own learnable smoothing, alpha, delta, and r parameters if enabled.
    """

    def __init__(self,
                 fs=16000,             # 采样率，不直接用，但保持风格一致
                 alpha=0.98,            # 初始归一化指数
                 delta=2.0,             # 偏置项
                 r=0.5,                 # 根压缩指数
                 eps=1e-6,              # 防止除零
                 smoothing_coef=0.025,  # 初始平滑系数
                 learnable=True,        # alpha, delta, r是否学习
                 learn_smoothing=True,  # 平滑系数是否学习
                 per_channel_smoothing=True,  # 是否每个channel独立
                 n_filters=None):       # 必须指定filter数量用于per-channel参数
        super().__init__()

        assert n_filters is not None, "n_filters must be specified for per-channel PCENLeaf!"

        self.eps = float(eps)
        self.n_filters = n_filters

        # Alpha
        if learnable:
            self.alpha = nn.Parameter(torch.full((n_filters,), alpha))
        else:
            self.register_buffer('alpha', torch.full((n_filters,), alpha))

        # Delta
        if learnable:
            self.delta = nn.Parameter(torch.full((n_filters,), delta))
        else:
            self.register_buffer('delta', torch.full((n_filters,), delta))

        # r
        if learnable:
            self.r = nn.Parameter(torch.full((n_filters,), r))
        else:
            self.register_buffer('r', torch.full((n_filters,), r))

        # smoothing coefficient
        if per_channel_smoothing:
            if learn_smoothing:
                self.smoothing_coef = nn.Parameter(torch.full((n_filters,), smoothing_coef))
            else:
                self.register_buffer('smoothing_coef', torch.full((n_filters,), smoothing_coef))
        else:
            if learn_smoothing:
                self.smoothing_coef = nn.Parameter(torch.tensor(smoothing_coef))
            else:
                self.register_buffer('smoothing_coef', torch.tensor(smoothing_coef))

        self.per_channel_smoothing = per_channel_smoothing

    def forward(self, x):
        """
        x: Tensor of shape (batch_size, n_filters, time)
        Returns:
            y: PCEN adapted output, same shape as input
        """
        B, C, T = x.shape
        assert C == self.n_filters, f"Expected {self.n_filters} filters, but got {C}"
        x = x.clamp(min=0.0)

        # Expand parameters for broadcasting: shape (1, C, 1)
        alpha = self.alpha.view(1, C, 1)
        delta = self.delta.view(1, C, 1)
        r = self.r.view(1, C, 1)

        if self.per_channel_smoothing:
            smoothing = self.smoothing_coef.view(1, C, 1).clamp(min=1e-5, max=1.0)
        else:
            smoothing = self.smoothing_coef.view(1, 1, 1).clamp(min=1e-5, max=1.0)

        one_minus_s = 1.0 - smoothing

        # Compute decay factors for cumulative smoothing
        time_idx = torch.arange(T, device=x.device).float().view(1, 1, T)
        decay_factors = one_minus_s.pow(time_idx)  # shape: (1, C, T) or (1, 1, T)

        # Weighted cumulative moving average (vectorized)
        x_decay = x * decay_factors
        x_cumsum = torch.cumsum(x_decay, dim=-1)
        weights = decay_factors.expand(B, C, T)
        weights_cumsum = torch.cumsum(weights, dim=-1)
        M = x_cumsum / (weights_cumsum + self.eps)

        # PCEN compression
        norm = x / (self.eps + M).pow(alpha)
        y = (norm + delta).pow(r) - delta.pow(r)

        return y





# @AdaptationRegistry.register("dual_tc")
# class DualTimeConstantAdaptation(nn.Module):
#     """
#     Dual-Time-Constant (Dual-TC) adaptation module.

#     Inspired by Zilany auditory nerve model fast adaptation.
#     Compatible with (batch_size, n_filters, time) input shape.
#     """

#     def __init__(self,
#                  fs=16000,
#                  tau_fast_ms=2.0,
#                  tau_slow_ms=60.0,
#                  learnable=True,
#                  eps=1e-6):
#         """
#         Initialize the Dual-TC adaptation module.

#         Parameters
#         ----------
#         fs : float
#             Sampling rate (Hz).
#         tau_fast_ms : float
#             Fast adaptation time constant in milliseconds.
#         tau_slow_ms : float
#             Slow adaptation time constant in milliseconds.
#         learnable : bool
#             Whether to make time constants and mixing weight learnable.
#         eps : float
#             Small value to avoid division by zero.
#         """
#         super().__init__()
#         self.fs = fs
#         self.eps = float(eps)  # Ensure eps is a float

#         # Compute smoothing coefficients from time constants
#         init_mu_fast = 1 - math.exp(-1 / (fs * tau_fast_ms / 1000))
#         init_mu_slow = 1 - math.exp(-1 / (fs * tau_slow_ms / 1000))

#         if learnable:
#             self.mu_fast = nn.Parameter(torch.tensor(init_mu_fast, dtype=torch.float32))
#             self.mu_slow = nn.Parameter(torch.tensor(init_mu_slow, dtype=torch.float32))
#             self.mix_weight = nn.Parameter(torch.tensor(0.5))  # initial 0.5 mixing
#         else:
#             self.register_buffer('mu_fast', torch.tensor(init_mu_fast, dtype=torch.float32))
#             self.register_buffer('mu_slow', torch.tensor(init_mu_slow, dtype=torch.float32))
#             self.register_buffer('mix_weight', torch.tensor(0.5))

#     def forward(self, x):
#         """
#         Forward pass of the Dual-TC adaptation.

#         Parameters
#         ----------
#         x : torch.Tensor
#             Input tensor, shape (batch_size, n_filters, time).

#         Returns
#         -------
#         torch.Tensor
#             Adapted output, same shape as input.
#         """
#         B, C, T = x.shape
#         x = x.clamp(min=0.0)

#         # Initialize fast and slow memory tracks
#         M_fast = x[:, :, 0]
#         M_slow = x[:, :, 0]

#         M_fast_list = [M_fast.unsqueeze(-1)]
#         M_slow_list = [M_slow.unsqueeze(-1)]

#         for t in range(1, T):
#             M_fast = (1 - self.mu_fast) * M_fast + self.mu_fast * x[:, :, t]
#             M_slow = (1 - self.mu_slow) * M_slow + self.mu_slow * x[:, :, t]
#             M_fast_list.append(M_fast.unsqueeze(-1))
#             M_slow_list.append(M_slow.unsqueeze(-1))

#         # Concatenate along time dimension
#         M_fast = torch.cat(M_fast_list, dim=-1)
#         M_slow = torch.cat(M_slow_list, dim=-1)

#         # Mix fast and slow
#         mix = torch.sigmoid(self.mix_weight)
#         M = mix * M_fast + (1 - mix) * M_slow

#         # Normalize
#         y = x / (self.eps + M)

#         return y
# @AdaptationRegistry.register("dual_tc")
# class DualTimeConstantAdaptationFast(nn.Module):
#     """
#     High-speed Dual-Time-Constant (Dual-TC) adaptation module (vectorized version).

#     Compatible with (batch_size, n_filters, time) input shape.
#     """

#     def __init__(self,
#                  fs=16000,
#                  tau_fast_ms=2.0,
#                  tau_slow_ms=60.0,
#                  learnable=True,
#                  eps=1e-6):
#         super().__init__()
#         self.fs = fs
#         self.eps = float(eps)

#         # Convert time constants to smoothing coefficients
#         init_mu_fast = 1 - math.exp(-1 / (fs * tau_fast_ms / 1000))
#         init_mu_slow = 1 - math.exp(-1 / (fs * tau_slow_ms / 1000))

#         if learnable:
#             self.mu_fast = nn.Parameter(torch.tensor(init_mu_fast, dtype=torch.float32))
#             self.mu_slow = nn.Parameter(torch.tensor(init_mu_slow, dtype=torch.float32))
#             self.mix_weight = nn.Parameter(torch.tensor(0.5))
#         else:
#             self.register_buffer('mu_fast', torch.tensor(init_mu_fast, dtype=torch.float32))
#             self.register_buffer('mu_slow', torch.tensor(init_mu_slow, dtype=torch.float32))
#             self.register_buffer('mix_weight', torch.tensor(0.5))

#     def forward(self, x):
#         """
#         Forward pass of the fast Dual-TC adaptation.

#         Parameters
#         ----------
#         x : torch.Tensor
#             Input tensor, shape (batch_size, n_filters, time).

#         Returns
#         -------
#         torch.Tensor
#             Adapted output, same shape as input.
#         """
#         B, C, T = x.shape
#         device = x.device
#         x = x.clamp(min=0.0)

#         # Create decay factors
#         decay_fast = 1.0 - self.mu_fast
#         decay_slow = 1.0 - self.mu_slow

#         # Build decay power vector: [1, decay, decay^2, ..., decay^(T-1)]
#         powers = torch.arange(T, device=device).float()

#         decay_factors_fast = decay_fast ** powers  # (T,)
#         decay_factors_slow = decay_slow ** powers  # (T,)

#         # Weighted inputs
#         x_weighted_fast = x * decay_factors_fast.view(1, 1, -1)  # (B, C, T)
#         x_weighted_slow = x * decay_factors_slow.view(1, 1, -1)  # (B, C, T)

#         # Cumulative sums
#         cumsum_fast = torch.cumsum(x_weighted_fast.flip(dims=[-1]), dim=-1).flip(dims=[-1])
#         cumsum_slow = torch.cumsum(x_weighted_slow.flip(dims=[-1]), dim=-1).flip(dims=[-1])

#         # Restore scaling
#         one_minus_decay_fast = 1.0 - decay_fast
#         one_minus_decay_slow = 1.0 - decay_slow

#         M_fast = one_minus_decay_fast * cumsum_fast
#         M_slow = one_minus_decay_slow * cumsum_slow

#         # Mixing
#         mix = torch.sigmoid(self.mix_weight)
#         M = mix * M_fast + (1.0 - mix) * M_slow

#         # Normalize
#         y = x / (self.eps + M)

#         return y


@AdaptationRegistry.register("dual_tc")
class DualTimeConstantAdaptation(nn.Module):
    def __init__(self, fs=16000, tau_r=0.002, tau_st=0.060, kernel_dur=0.3):
        """
        Softplus + double-exponential adaptation module (fixed kernel, auto channel-detecting).
        """
        super().__init__()
        self.fs = fs
        self.tau_r = tau_r
        self.tau_st = tau_st
        self.kernel_dur = kernel_dur
        self.softplus = nn.Softplus()

        # kernel 相关变量
        self.kernel = None
        self.groups = None
        self.kernel_len = int(kernel_dur * fs)

    def _build_kernel(self, channels, device):
        dt = 1.0 / self.fs
        t = torch.arange(0, self.kernel_dur, dt, device=device)
        kernel = torch.exp(-t / self.tau_r) + torch.exp(-t / self.tau_st)
        kernel = kernel / kernel.sum()
        kernel = kernel.view(1, 1, -1).repeat(channels, 1, 1)
        return kernel

    def forward(self, x):
        """
        x: (B, C, T)
        returns: (B, C, T)
        """
        B, C, T = x.shape
        x = self.softplus(x)

        if self.kernel is None:
            self.kernel = self._build_kernel(C, x.device)
            self.groups = C

        y = F.conv1d(x, self.kernel, padding=self.kernel.shape[2] - 1, groups=self.groups)
        return y


@AdaptationRegistry.register("dual_tc_hp_mix")
class DualTimeConstantHighPassMixAdaptation(nn.Module):
    """
    Dual-Time-Constant (Dual-TC) adaptation with High-Pass residual mixing.

    Inspired by Zilany auditory nerve model + perceptual enhancement.
    Compatible with (batch_size, n_filters, time) input.
    """

    def __init__(self,
                 fs=16000,
                 tau_fast_ms=2.0,
                 tau_slow_ms=60.0,
                 mix_weight_hp=0.3,
                 learnable=True,
                 eps=1e-6):
        """
        Initialize the Dual-TC-HP adaptation module.

        Parameters
        ----------
        fs : float
            Sampling rate (Hz).
        tau_fast_ms : float
            Fast adaptation time constant in milliseconds.
        tau_slow_ms : float
            Slow adaptation time constant in milliseconds.
        mix_weight_hp : float
            Mixing ratio for high-pass residual component.
        learnable : bool
            Whether to make parameters learnable.
        eps : float
            Small value to avoid division by zero.
        """
        super().__init__()
        self.fs = fs
        self.eps = float(eps)

        # Compute smoothing coefficients
        init_mu_fast = 1 - math.exp(-1 / (fs * tau_fast_ms / 1000))
        init_mu_slow = 1 - math.exp(-1 / (fs * tau_slow_ms / 1000))

        if learnable:
            self.mu_fast = nn.Parameter(torch.tensor(init_mu_fast, dtype=torch.float32))
            self.mu_slow = nn.Parameter(torch.tensor(init_mu_slow, dtype=torch.float32))
            self.mix_weight_adapt = nn.Parameter(torch.tensor(0.5))  # Adapted track mixing weight
            self.mix_weight_hp = nn.Parameter(torch.tensor(mix_weight_hp))  # High-pass residual mixing weight
        else:
            self.register_buffer('mu_fast', torch.tensor(init_mu_fast, dtype=torch.float32))
            self.register_buffer('mu_slow', torch.tensor(init_mu_slow, dtype=torch.float32))
            self.register_buffer('mix_weight_adapt', torch.tensor(0.5))
            self.register_buffer('mix_weight_hp', torch.tensor(mix_weight_hp))

    def forward(self, x):
        """
        Forward pass of the Dual-TC-HP adaptation.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor, shape (batch_size, n_filters, time).

        Returns
        -------
        torch.Tensor
            Adapted output, same shape as input.
        """
        B, C, T = x.shape
        x = x.clamp(min=0.0)

        # Initialize memory tracks
        M_fast = x[:, :, 0]
        M_slow = x[:, :, 0]

        M_fast_list = [M_fast.unsqueeze(-1)]
        M_slow_list = [M_slow.unsqueeze(-1)]

        for t in range(1, T):
            M_fast = (1 - self.mu_fast) * M_fast + self.mu_fast * x[:, :, t]
            M_slow = (1 - self.mu_slow) * M_slow + self.mu_slow * x[:, :, t]
            M_fast_list.append(M_fast.unsqueeze(-1))
            M_slow_list.append(M_slow.unsqueeze(-1))

        M_fast = torch.cat(M_fast_list, dim=-1)
        M_slow = torch.cat(M_slow_list, dim=-1)

        # Mix fast and slow background
        mix_adapt = torch.sigmoid(self.mix_weight_adapt)
        M = mix_adapt * M_fast + (1 - mix_adapt) * M_slow

        # Normalize (low-frequency adapted signal)
        x_adapted = x / (self.eps + M)

        # Calculate high-pass residual
        x_hp = x - M

        # Final output: adapted + weighted high-pass residual
        mix_hp = torch.sigmoid(self.mix_weight_hp)  # Ensure between 0 and 1
        y = x_adapted + mix_hp * x_hp

        return y




@IntegrationRegistry.register("frame_avg")
class FrameBasedAveraging(nn.Module):
    """
    Frame-based averaging module (mean or RMS).

    Typically applied between adaptation and modulation stages.

    Parameters
    ----------
    fs : float
        Sampling frequency in Hz
    window_size_ms : float, optional
        Frame window size in milliseconds (default = 8 ms)
    stride_ms : float, optional
        Stride size in milliseconds (default = same as window size)
    mode : str, optional
        Averaging mode: 'mean' or 'rms' (default = 'rms')

    """

    def __init__(self, fs, window_size_ms=8.0, stride_ms=None, mode="rms", **kwargs):
        super().__init__()
        self.fs = fs
        self.mode = mode.lower()

        if stride_ms is None:
            stride_ms = window_size_ms

        self.frame_size = int(round(window_size_ms * fs / 1000))
        self.stride = int(round(stride_ms * fs / 1000))

        if self.frame_size <= 0 or self.stride <= 0:
            raise ValueError(
                "window_size_ms and stride_ms must convert to positive integers."
            )

    def forward(self, x):
        """Parameters

        ----------
        x : torch.Tensor
            Input tensor with shape (batch_size, ..., time)

        Returns
        -------
        torch.Tensor
            Output tensor with shape (batch_size, ..., num_frames)

        """
        # Save original dimension information
        orig_shape = x.shape
        B = orig_shape[0]  # batch size
        # Reshape input to 3D (batch, channels, time)
        if x.ndim == 2:
            x = x.unsqueeze(1)  # (batch, 1, time)
        elif x.ndim > 3:
            # For higher dimensional input, merge middle dimensions into channels
            x = x.reshape(B, -1, x.shape[-1])

        C = x.shape[1]  # channels
        T = x.shape[-1]  # time

        if T < self.frame_size:
            raise ValueError(
                f"Input length ({T}) is shorter than frame size ({self.frame_size})."
            )

        # Process frames
        x = x.reshape(B * C, 1, T)
        frames = F.unfold(
            x.unsqueeze(-1),
            kernel_size=(self.frame_size, 1),
            stride=(self.stride, 1),
        )
        frames = frames.reshape(B, C, self.frame_size, -1)

        if self.mode == "mean":
            y = frames.mean(dim=2)
        elif self.mode == "rms":
            y = torch.sqrt((frames ** 2).mean(dim=2) + 1e-8)
        else:
            raise ValueError(f"Unknown mode: {self.mode}. Use 'mean' or 'rms'.")

        # Restore original dimension structure
        if len(orig_shape) > 3:
            # Decompose channels dimension back to original dimensions
            y = y.reshape(orig_shape[:-1] + (-1,))
        elif len(orig_shape) == 2:
            # Remove added channel dimension
            y = y.squeeze(1)

        return y
    
@IntegrationRegistry.register("frame_avg_hp_mix")
class FrameAverageHighPassMix(nn.Module):
    """Frame-based averaging with high-pass residual mixed back.

    Smoothing (low-pass) + high-frequency residual signal, maintaining original time resolution.
    """

    def __init__(self, fs, window_size_ms=2.0, mode="mean", mix_weight=1.0, eps=1e-8):
        """Initialize the frame-based averaging with high-pass residual mixing.

        Parameters
        ----------
        fs : float
            Sampling rate (Hz)
        window_size_ms : float, optional
            Smoothing window length (milliseconds)
        stride_ms : float, optional
            Stride length (milliseconds)
        mode : str, optional
            'mean' or 'rms', determines whether to use average or root mean square
        mix_weight : float, optional
            High-frequency component mixing ratio (default 1.0, adjustable)
        eps : float, optional
            Small value to prevent division by zero
        """
        super().__init__()
        self.fs = fs
        self.win_len = int(round(window_size_ms * fs / 1000))    # Window size (samples)
        self.stride = 1          # Stride (samples)
        self.mode = mode.lower()
        self.eps = float(eps)
        self.mix_weight = mix_weight

        # Create simple moving average convolution kernel
        self.conv = nn.Conv1d(
            in_channels=1,
            out_channels=1,
            kernel_size=self.win_len,
            stride=self.stride,          # Add stride here for automatic downsampling
            padding=(self.win_len - 1) // 2,
            bias=False,
            groups=1
        )

        # Initialize convolution kernel as uniform averager
        if self.mode in ["mean", "rms"]:
            nn.init.constant_(self.conv.weight, 1.0 / self.win_len)
        else:
            raise ValueError("mode must be 'mean' or 'rms'")

    def forward(self, x):
        """Process input signal through frame-based averaging and high-pass residual mixing.

        Input:
        - x: (batch_size, channels, time)

        Output:
        - (batch_size, channels, time-reduced signal)
        """
        B, C, T = x.shape
        x = x.view(B * C, 1, T)            # Merge batch and channel for convolution
        x_lp = self.conv(x)                # Smoothed low-pass signal
        x_lp = x_lp.view(B, C, -1)          # Restore batch and channel dimensions

        if self.mode == "rms":
            x_lp = (x_lp.pow(2) + self.eps).sqrt()

        # Note: Because stride>1, x_lp's time dimension is reduced!
        # For residual calculation, high-frequency signal needs to be sampled synchronously

        x_hp = x - x_lp                  # High-frequency component (original - smoothed)
        x_out = x_lp + self.mix_weight * x_hp   # Low-pass + high-pass fusion

        return x_out
    

@IntegrationRegistry.register("none")
class NoFrameBasedAveraging(nn.Module):
    """Identity frame-based averaging."""

    def __init__(self, *args, **kwargs):
        super().__init__()

    def forward(self, x):
        """Return input unchanged."""
        return x


def audiogram_to_ohc_ihc_loss(audiogram, freqs=None):
    """Compute OHC and IHC loss from audiogram.

    Parameters
    ----------
    audiogram : torch.Tensor
        Audiogram. Shape ``(batch_size, n_thresholds, 2)``. First column is frequency in
        Hz, second column is hearing loss in dB.
    freqs : torch.Tensor, optional
        Frequencies to interpolate the audiogram at. Shape ``(batch_size, n_freqs)``
        or ``(n_freqs,)``. If ``None``, uses the input audiogram frequencies.

    Returns
    -------
    ohc_loss : torch.Tensor
        OHC loss in [0, 1]. Shape ``(batch_size, n_freqs)``.
    ihc_loss : torch.Tensor
        IHC loss in [0, 1]. Shape ``(batch_size, n_freqs)``.

    """
    max_ohc_loss = torch.tensor(
        [
            [250, 18.5918602171780],
            [375, 23.0653774100513],
            [500, 25.2602607820868],
            [750, 30.7013288310918],
            [1000, 34.0272671055467],
            [1500, 38.6752655699390],
            [2000, 39.5318838824221],
            [3000, 39.4930128714544],
            [4000, 39.3156363872299],
            [6000, 40.5210536471565],
        ],
        device=audiogram.device,
        dtype=audiogram.dtype,
    )
    if freqs is None:
        total_loss = audiogram[..., 1]
    else:
        max_ohc_loss = linear_interpolation(
            torch.log10(freqs), torch.log10(max_ohc_loss[:, 0]), max_ohc_loss[:, 1]
        ).clamp(0, 105)
        total_loss = linear_interpolation(
            torch.log10(freqs), torch.log10(audiogram[..., 0]), audiogram[..., 1]
        ).clamp(0, 105)
    # 2/3 OHC loss and 1/3 IHC loss
    ohc_loss = torch.clamp(2 / 3 * total_loss, max=max_ohc_loss)
    ihc_loss = total_loss - ohc_loss
    ohc_loss = 10 ** (-ohc_loss / 20)
    ihc_loss = 10 ** (-ihc_loss / 20)
    return ohc_loss, ihc_loss


def debug_plot_frame_avg_stacked(x_orig, x_avg, fs, window_size_ms, stride_ms,
                                  time_limit_s=None, save_path="frame_avg_debug.png"):
    """
    Stacked plot comparing original signal and frame-averaged RMS output.

    Parameters
    ----------
    x_orig : torch.Tensor
        Original signal. Shape (B, C, T)
    x_avg : torch.Tensor
        Frame-averaged RMS signal. Shape (B, C, N)
    fs : float
        Sampling rate in Hz
    window_size_ms : float
        Frame window size in milliseconds
    stride_ms : float
        Stride size in milliseconds
    time_limit_s : float or None
        Time window to plot in seconds. If None, automatically selects ~20 frames
    save_path : str
        Where to save the figure

    """
    B, C, T = x_orig.shape
    _, _, N = x_avg.shape

    # Default to ~20 frames display length
    if time_limit_s is None:
        time_limit_s = (stride_ms * 20) / 1000.0

    # Limit sample count
    sample_limit = min(int(time_limit_s * fs), T)
    frame_limit = min(int(time_limit_s / (stride_ms / 1000.0)), N)

    # Time axis
    t_orig = np.arange(sample_limit) / fs
    t_avg = np.arange(frame_limit) * (stride_ms / 1000.0)

    # Take first batch and channel
    sig = x_orig[3, 15, :sample_limit].detach().cpu().numpy()
    rms = x_avg[3, 15, :frame_limit].detach().cpu().numpy()

    # Plot
    fig, axs = plt.subplots(2, 1, figsize=(12, 4), sharex=True)
    fig.suptitle(
        f"Frame-Based Averaging (B0, C0) — First {time_limit_s*1000:.0f}ms",
        fontsize=14
    )

    axs[0].plot(t_orig, sig, label="Original", color='steelblue', linewidth=1)
    axs[0].set_ylabel("Amplitude")
    axs[0].set_title("Original Signal")
    axs[0].grid(True)

    axs[1].plot(t_avg, rms, label="Smoothed RMS", color='darkorange', linewidth=2)
    axs[1].set_ylabel("RMS")
    axs[1].set_xlabel("Time (s)")
    axs[1].set_title(f"Smoothed Output — {window_size_ms}ms / {stride_ms}ms")
    axs[1].grid(True)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(save_path)
    plt.close()

    return save_path



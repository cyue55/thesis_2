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
        integration="frame_avg",
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
        x = self.filterbank(x, ohc_loss=ohc_loss)
        x = self.ihc(x)
        if ihc_loss is not None:
            ihc_loss = ihc_loss.unsqueeze(-1)  # (batch_size, n_filters, 1)
            while ihc_loss.ndim < x.ndim:
                ihc_loss = ihc_loss.unsqueeze(1)  # (batch_size, ..., n_filters, 1)
            x = x * ihc_loss
        x = self.adaptation(x)
        if self.frame_average is not None:
            x = self.frame_average(x)
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
        # 保存原始维度信息
        orig_shape = x.shape
        B = orig_shape[0]  # batch size
        # 将输入重塑为3D (batch, channels, time)
        if x.ndim == 2:
            x = x.unsqueeze(1)  # (batch, 1, time)
        elif x.ndim > 3:
            # 对于更高维度的输入，将中间维度合并为channels
            x = x.reshape(B, -1, x.shape[-1])

        C = x.shape[1]  # channels
        T = x.shape[-1]  # time

        if T < self.frame_size:
            raise ValueError(
                f"Input length ({T}) is shorter than frame size ({self.frame_size})."
            )

        # 处理帧
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

        # 恢复原始维度结构
        if len(orig_shape) > 3:
            # 将channels维度分解回原始维度
            y = y.reshape(orig_shape[:-1] + (-1,))
        elif len(orig_shape) == 2:
            # 移除添加的channel维度
            y = y.squeeze(1)

        return y


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

    # Default to ~20帧展示长度
    if time_limit_s is None:
        time_limit_s = (stride_ms * 20) / 1000.0

    # 限制样本数量
    sample_limit = min(int(time_limit_s * fs), T)
    frame_limit = min(int(time_limit_s / (stride_ms / 1000.0)), N)

    # 时间轴
    t_orig = np.arange(sample_limit) / fs
    t_avg = np.arange(frame_limit) * (stride_ms / 1000.0)

    # 取第一个 batch、通道
    sig = x_orig[3, 15, :sample_limit].detach().cpu().numpy()
    rms = x_avg[3, 15, :frame_limit].detach().cpu().numpy()

    # 画图
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

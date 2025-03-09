import math

import numpy as np
import torch

from .stft import STFT


def hz_to_mel(hz, scale="slaney"):
    """Convert frequency in Hz to mel scale.

    Parameters
    ----------
    hz : float or numpy.ndarray or torch.Tensor
        Frequency in Hz.
    scale : {"htk", "slaney"}, optional
        Mel scale to use. ``"htk"`` matches the Hidden Markov Toolkit, while
        ``"slaney"`` matches the Auditory Toolbox by Slaney. The ``"slaney"`` scale is
        linear below 1 kHz and logarithmic above 1 kHz.

    Returns
    -------
    float or numpy.ndarray or torch.Tensor
        Frequency in mel scale.

    """
    if scale == "htk":
        if isinstance(hz, torch.Tensor):
            mel = 2595 * torch.log10(1 + hz / 700)
        elif isinstance(hz, (np.ndarray, np.generic)):
            mel = 2595 * np.log10(1 + hz / 700)
        elif isinstance(hz, (float, int)):
            mel = 2595 * math.log10(1 + hz / 700)
        else:
            raise ValueError(f"Invalid input type: {type(hz)}")
    elif scale == "slaney":
        hz_min = 0
        hz_sp = 200 / 3
        mel = (hz - hz_min) / hz_sp
        min_log_hz = 1000
        min_log_mel = (min_log_hz - hz_min) / hz_sp
        log_step = math.log(6.4) / 27
        if isinstance(hz, torch.Tensor):
            if hz.ndim == 0:
                if hz >= min_log_hz:
                    mel = min_log_mel + torch.log(hz / min_log_hz) / log_step
            else:
                mask = hz >= min_log_hz
                mel[mask] = min_log_mel + torch.log(hz[mask] / min_log_hz) / log_step
        elif isinstance(hz, (np.ndarray, np.generic)):
            if hz.ndim == 0:
                if hz >= min_log_hz:
                    mel = min_log_mel + np.log(hz / min_log_hz) / log_step
            else:
                mask = hz >= min_log_hz
                mel[mask] = min_log_mel + np.log(hz[mask] / min_log_hz) / log_step
        elif isinstance(hz, (float, int)):
            if hz >= min_log_hz:
                mel = min_log_mel + math.log(hz / min_log_hz) / log_step
        else:
            raise ValueError(f"Invalid input type: {type(hz)}")
    else:
        raise ValueError(f"Invalid mel scale: {scale}")
    return mel


def mel_to_hz(mel, scale="slaney"):
    """Convert frequency in mel scale to Hz.

    Parameters
    ----------
    mel : float or numpy.ndarray or torch.Tensor
        Frequency in mel scale.
    scale : {"htk", "slaney"}, optional
        Mel scale to use. ``"htk"`` matches the Hidden Markov Toolkit, while
        ``"slaney"`` matches the Auditory Toolbox by Slaney. The ``"slaney"`` scale is
        linear below 1 kHz and logarithmic above 1 kHz.

    Returns
    -------
    float or numpy.ndarray or torch.Tensor
        Frequency in Hz.

    """
    if scale == "htk":
        hz = 700 * (10 ** (mel / 2595) - 1)
    elif scale == "slaney":
        hz_min = 0
        hz_sp = 200 / 3
        hz = hz_min + hz_sp * mel
        min_log_hz = 1000
        min_log_mel = (min_log_hz - hz_min) / hz_sp
        log_step = math.log(6.4) / 27
        if isinstance(hz, torch.Tensor):
            if hz.ndim == 0:
                if mel >= min_log_mel:
                    hz = min_log_hz * torch.exp(log_step * (mel - min_log_mel))
            else:
                mask = mel >= min_log_mel
                hz[mask] = min_log_hz * torch.exp(log_step * (mel[mask] - min_log_mel))
        elif isinstance(hz, (np.ndarray, np.generic)):
            if hz.ndim == 0:
                if mel >= min_log_mel:
                    hz = min_log_hz * np.exp(log_step * (mel - min_log_mel))
            else:
                mask = mel >= min_log_mel
                hz[mask] = min_log_hz * np.exp(log_step * (mel[mask] - min_log_mel))
        elif isinstance(hz, (float, int)):
            if mel >= min_log_mel:
                hz = min_log_hz * math.exp(log_step * (mel - min_log_mel))
        else:
            raise ValueError(f"Invalid input type: {type(hz)}")
    else:
        raise ValueError(f"Invalid mel scale: {scale}")
    return hz


def mel_filters(
    n_filters=64,
    n_fft=512,
    f_min=0.0,
    f_max=None,
    fs=16000,
    scale="slaney",
    norm="valid",
    dtype=torch.float32,
):
    """Compute mel filterbank.

    Parameters
    ----------
    n_filters : int, optional
        Number of mel filters.
    n_fft : int, optional
        Number of FFT point.
    f_min : float, optional
        Minimum frequency.
    f_max : float, optional
        Maximum frequency. If ``None``, uses ``fs / 2``.
    fs : float, optional
        Sampling frequency.
    scale : {"htk", "slaney"}, optional
        Mel scale to use. ``"htk"`` matches the Hidden Markov Toolkit, while
        ``"slaney"`` matches the Auditory Toolbox by Slaney. The ``"slaney"`` scale is
        linear below 1 kHz and logarithmic above 1 kHz.
    norm : {"valid", "slaney"}, optional
        How to normalize the filters. If ``"slaney"``, the filters are normalized by
        their width in Hz. However this makes the filter response scale with the
        frequency resolution of the FFT ``fs / n_fft``. If ``"valid"``, the
        normalization factor takes the frequency resolution into account. This is the
        same as L1 normalization. If ``None``, no normalization is applied.
    dtype : torch.dtype, optional
        Data type of the filters.

    Returns
    -------
    filters : torch.Tensor
        Mel filterbank. Shape ``(n_filters, n_fft // 2 + 1)``.
    fc : torch.Tensor
        Center frequencies of the filters. Shape ``(n_filters,)``.
    norm_factor : torch.Tensor
        Normalization factor. Shape ``(n_filters)``.

    """
    f_max = fs / 2 if f_max is None else f_max
    mel_min = hz_to_mel(f_min, scale)
    mel_max = hz_to_mel(f_max, scale)
    mel = torch.linspace(mel_min, mel_max, n_filters + 2, dtype=dtype)
    fc = mel_to_hz(mel, scale)
    f = torch.arange(n_fft // 2 + 1) * fs / n_fft
    dfc = fc.diff().unsqueeze(1)
    slopes = fc.unsqueeze(1) - f.unsqueeze(0)
    down_slopes = -slopes[:-2] / dfc[:-1]
    up_slopes = slopes[2:] / dfc[1:]
    filters = torch.min(down_slopes, up_slopes).clamp(min=0)
    if norm is None:
        norm_factor = torch.ones(n_filters)
    elif norm == "valid":
        norm_factor = 0.5 * n_fft / fs * (fc[2:] - fc[:-2])
    elif norm == "slaney":
        norm_factor = 0.5 * (fc[2:] - fc[:-2])
    else:
        raise ValueError(f"Invalid mel filter normalization: {norm}")
    return filters / norm_factor.unsqueeze(1), fc, norm_factor


class MelFilterbank:
    """Triangular mel filterbank.

    Transforms a linear-frequency spectrogram into a mel-spectrogram.

    Parameters
    ----------
    n_filters : int, optional
        Number of mel filters.
    n_fft : int, optional
        Number of FFT point.
    f_min : float, optional
        Minimum frequency.
    f_max : float, optional
        Maximum frequency. If ``None``, uses ``fs / 2``.
    fs : float, optional
        Sampling frequency.
    scale : {"htk", "slaney"}, optional
        Mel scale to use. ``"htk"`` matches the Hidden Markov Toolkit, while
        ``"slaney"`` matches the Auditory Toolbox by Slaney. The ``"slaney"`` scale is
        linear below 1 kHz and logarithmic above 1 kHz.
    norm : {"valid", "slaney"}, optional
        How to normalize the filters. If ``"slaney"``, the filters are normalized by
        their width in Hz. However this makes the filter response scale with the
        frequency resolution of the FFT ``fs / n_fft``. If ``"valid"``, the
        normalization factor takes the frequency resolution into account. This is the
        same as L1 normalization. If ``None``, no normalization is applied.
    dtype : torch.dtype, optional
        Data type of the filters.

    """

    def __init__(
        self,
        n_filters=64,
        n_fft=512,
        f_min=0.0,
        f_max=None,
        fs=16000,
        scale="slaney",
        norm="valid",
        dtype=torch.float32,
    ):
        filters, fc, norm_factor = mel_filters(
            n_filters=n_filters,
            n_fft=n_fft,
            f_min=f_min,
            f_max=f_max,
            fs=fs,
            scale=scale,
            norm=norm,
            dtype=dtype,
        )
        inverse_filters = torch.t(filters * norm_factor.unsqueeze(1))
        self.filters_cpu = filters.cpu()
        self.inverse_filters_cpu = inverse_filters.cpu()
        if torch.cuda.is_available():
            self.filters_cuda = filters.cuda()
            self.inverse_filters_cuda = inverse_filters.cuda()
        self.fc = fc

    def __call__(self, x):
        """Apply mel filterbank.

        Parameters
        ----------
        x : torch.Tensor
            Linear-frequency spectrogram. Shape ``(..., n_fft // 2 + 1, n_frames)``.

        Returns
        -------
        torch.Tensor
            Mel-frequency spectrogram. Shape ``(..., n_filters, n_frames)``.

        """
        if x.is_cuda:
            filters = self.filters_cuda
        else:
            filters = self.filters_cpu
        return torch.einsum("ij,...jk->...ik", filters, x)

    def inverse(self, x):
        """Inverse mel filterbank.

        Parameters
        ----------
        x : torch.Tensor
            Mel-frequency spectrogram. Shape ``(..., n_filters, n_frames)``.

        Returns
        -------
        torch.Tensor
            Linear-frequency spectrogram. Shape ``(..., n_fft // 2 + 1, n_frames)``.

        """
        if x.is_cuda:
            inverse_filters = self.inverse_filters_cuda
        else:
            inverse_filters = self.inverse_filters_cpu
        return torch.einsum("ij,...jk->...ik", inverse_filters, x)


class MelSpectrogram:
    """Mel spectrogram.

    Parameters
    ----------
    frame_length : int, optional
        Frame length.
    hop_length : int, optional
        Hop length.
    n_fft : int, optional
        Number of FFT point.
    window : str, optional
        Window function.
    center : bool, optional
        If ``True``, pad the input so that the output has the same length as the input.
    pad_mode : str, optional
        Padding mode.
    normalized : bool, optional
        If ``True``, normalize the STFT by the window sum.
    use_torch : bool, optional
        If ``True``, use PyTorch for the STFT. Otherwise, use NumPy.
    n_filters : int, optional
        Number of mel filters.
    f_min : float, optional
        Minimum frequency.
    f_max : float, optional
        Maximum frequency. If ``None``, uses ``fs / 2``.
    fs : float, optional
        Sampling frequency.
    norm : {"valid", "slaney"}, optional
        How to normalize the filters. If ``"slaney"``, the filters are normalized by
        their width in Hz. However this makes the filter response scale with the
        frequency resolution of the FFT ``fs / n_fft``. If ``"valid"``, the
        normalization factor takes the frequency resolution into account. This is the
        same as L1 normalization. If ``None``, no normalization is applied.
    scale : {"htk", "slaney"}, optional
        Mel scale to use. ``"htk"`` matches the Hidden Markov Toolkit, while
        ``"slaney"`` matches the Auditory Toolbox by Slaney. The ``"slaney"`` scale is
        linear below 1 kHz and logarithmic above 1 kHz.
    power : float, optional
        Exponent for the magnitude spectrogram.
    log : bool, optional
        If ``True``, take the logarithm of the mel spectrogram.
    log_eps : float, optional
        Small value to avoid taking the logarithm of zero.
    mean : bool, optional
        If ``True``, compute the mean of the mel spectrogram.
    std : bool, optional
        If ``True``, compute the standard deviation of the mel spectrogram.
    vad_dyn_range : float, optional
        If specified, apply voice activity detection (VAD) to discard frames with
        energy below the maximum energy minus this value in dB.
    dtype : torch.dtype, optional
        Data type of the filters.
    _discard_trailing_samples : bool, optional
        Whether to discard the trailing samples that do not fit a frame. Useful for
        testing against implementations that do so.
    _center_padding_is_half_frame_length : bool, optional
        Whether the padding when ``center=True`` is ``frame_length // 2`` instead of
        ``frame_length - hop_length``. Useful for testing against implementations that
        use ``frame_length // 2``.

    """

    def __init__(
        self,
        frame_length=512,
        hop_length=128,
        n_fft=None,
        window="hann",
        center=True,
        pad_mode="constant",
        normalized=False,
        use_torch=True,
        n_filters=64,
        f_min=0.0,
        f_max=None,
        fs=16000,
        norm="valid",
        scale="slaney",
        power=1,
        log=True,
        log_eps=1e-7,
        mean=False,
        std=False,
        vad_dyn_range=None,
        dtype=torch.float32,
        _discard_trailing_samples=False,
        _center_padding_is_half_frame_length=False,
    ):
        self.stft = STFT(
            frame_length=frame_length,
            hop_length=hop_length,
            n_fft=frame_length if n_fft is None else n_fft,
            window=window,
            center=center,
            pad_mode=pad_mode,
            normalized=normalized,
            use_torch=use_torch,
            _discard_trailing_samples=_discard_trailing_samples,
            _center_padding_is_half_frame_length=_center_padding_is_half_frame_length,
            onesided=True,
            compression_factor=1,
            scale_factor=1,
        )
        self.mel_fb = MelFilterbank(
            n_filters=n_filters,
            n_fft=frame_length if n_fft is None else n_fft,
            f_min=f_min,
            f_max=f_max,
            fs=fs,
            norm=norm,
            scale=scale,
            dtype=dtype,
        )
        self.power = power
        self.log = log
        self.log_eps = log_eps
        self.mean = mean
        self.std = std
        self.vad_dyn_range = vad_dyn_range

    def __call__(self, x):
        """Compute mel spectrogram.

        Parameters
        ----------
        x : torch.Tensor
            Input signal. Shape ``(..., n_samples)``.

        Returns
        -------
        torch.Tensor
            Mel spectrogram. Shape ``(..., n_filters, n_frames)``.

        """
        stft = self.stft(x)
        x = stft.abs().pow(self.power)
        x = self.mel_fb(x)
        if self.log:
            x = x.clamp(min=self.log_eps).log()
        if self.vad_dyn_range is None:
            mask = torch.ones(
                *x.shape[:-2],
                x.shape[-1],
                device=x.device,
                dtype=x.dtype,
            )
        else:
            x_dB = 10 * stft.abs().pow(2).sum(dim=-2).clamp(min=self.log_eps).log10()
            x_dB_max = x_dB.amax(dim=-1, keepdim=True)
            mask = x_dB > x_dB_max - self.vad_dyn_range
        x = x * mask.unsqueeze(-2)
        if self.mean or self.std:
            n = mask.sum(dim=-1, keepdim=True)
            mean = x.sum(dim=-1) / n
            if self.std:
                var = x.pow(2).sum(dim=-1) / n - mean.pow(2)
                std = var.clamp(min=0).sqrt()
            if self.mean and self.std:
                x = torch.cat([mean, std], dim=-1)
            elif self.mean:
                x = mean
            else:
                x = std
        return x

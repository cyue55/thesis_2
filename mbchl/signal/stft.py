# Copyright 2024 Philippe Gonzalez
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import functools
import math

import numpy as np
import scipy.signal
import torch
import torch.nn.functional as F


class STFT:
    """Short-time Fourier transform (STFT).

    A wrapper for :func:`torch.stft` and :func:`torch.istft` that fixes some of their
    limitations:

    - They do not allow `str` or callable input for `window`. See
      `here <https://github.com/pytorch/pytorch/issues/88919>`__.
    - They use a wrong normalization factor. See
      `here <https://github.com/pytorch/pytorch/issues/81428>`__.
    - :func:`torch.stft` discards trailing samples if they do not fit a frame. This can
      happen even if ``center=True``. This means data can be lost! See
      `here <https://github.com/pytorch/pytorch/issues/70073>`__.
    - Using ``center=False`` and a non-rectangular window raises a ``RuntimeError``. See
      `here <https://github.com/pytorch/pytorch/issues/91309>`__.
    - Using windows with many zeroes like the Mauler windows raises a ``RuntimeError``.
    - The frames are padded left and right when ``n_fft > frame_length``. See
      `here <https://github.com/pytorch/pytorch/blob/75c22dd8bf8801a6eaf9bbe30e08cf8c05ded6a1/aten/src/ATen/native/SpectralOps.cpp#L937)>`__.
    - :func:`torch.istft` assumes the output is real and trims the input FFT to call
      :func:`torch.fft.irfft`. See
      `here <https://github.com/pytorch/pytorch/blob/bf5c7bf06debad4f1384d4b3e8ada936b6b22b19/aten/src/ATen/native/SpectralOps.cpp#L1135C1-L1135C49)>`__.
    - They do not support arbitrary input shapes.

    Parameters
    ----------
    frame_length : int, optional
        Frame length.
    hop_length : int, optional
        Hop length.
    n_fft : int, optional
        Length of the FFT. If ``None``, uses ``frame_length``.
    window : str or callable or numpy.ndarray or dict, optional
        Window function. If different windows are used for analysis and synthesis, pass
        a dict with keys "analysis" and "synthesis".
    center : bool, optional
        Whether to pad the input on both sides.
    pad_mode : str, optional
        Padding mode.
    normalized : bool, optional
        Whether to normalize the STFT output.
    onesided : bool, optional
        Whether to return only the positive frequencies.
    compression_factor : float, optional
        Compression factor for the output magnitude.
    scale_factor : float, optional
        Scaling factor for the output.
    use_torch : bool, optional
        Whether to use :func:`torch.stft` and :func:`torch.istft` or to manually segment
        the input and compute the FFT using :func:`torch.fft.fft` and
        :func:`torch.fft.ifft`. Using ``use_torch=False`` fixes errors when using
        non-rectangular windows.
    _pad_frames_right : bool, optional
        Whether to pad the frames to the right instead of left and right before
        computing the FFT. Ignored if ``use_torch=True`` as :func:`torch.stft` pads the
        frames left and right. Useful for testing when comparing ``use_torch=True`` and
        ``use_torch=False``.
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
        onesided=True,
        compression_factor=1.0,
        scale_factor=1.0,
        use_torch=False,
        _pad_frames_right=True,
        _discard_trailing_samples=False,
        _center_padding_is_half_frame_length=False,
    ):
        n_fft = frame_length if n_fft is None else n_fft

        if window == "mauler":
            analysis_window, synthesis_window = mauler_windows(
                analysis_length=frame_length,
                synthesis_length=2 * hop_length,
                n_fft=n_fft,
            )
            window = {"analysis": analysis_window, "synthesis": synthesis_window}
            frame_length = n_fft
        if isinstance(window, dict):
            analysis_window = _check_window(window["analysis"], frame_length)
            synthesis_window = _check_window(window["synthesis"], frame_length)
        else:
            analysis_window = synthesis_window = _check_window(window, frame_length)

        # apply correct normalization
        if normalized:
            norm_factor = 1 / analysis_window.pow(2).sum().sqrt()
        else:
            norm_factor = 1
        analysis_window = analysis_window * norm_factor
        synthesis_window = synthesis_window * norm_factor

        self.frame_length = frame_length
        self.hop_length = hop_length
        self.center = center
        self.pad_mode = pad_mode
        self.normalized = normalized
        self.onesided = onesided
        self.compression_factor = compression_factor
        self.scale_factor = scale_factor
        self.n_fft = n_fft
        self.use_torch = use_torch
        self._pad_frames_right = _pad_frames_right
        self._discard_trailing_samples = _discard_trailing_samples
        self._center_padding_is_half_frame_length = _center_padding_is_half_frame_length
        self.analysis_window_cpu = analysis_window.cpu()
        self.synthesis_window_cpu = synthesis_window.cpu()
        if torch.cuda.is_available():
            self.analysis_window_cuda = analysis_window.cuda()
            self.synthesis_window_cuda = synthesis_window.cuda()

    def __call__(self, x, axis=-1, return_type="complex"):
        """Compute the STFT of a signal.

        Parameters
        ----------
        x : torch.Tensor
            Input signal.
        axis : int, optional
            Axis along which to compute the STFT.
        return_type : {"complex", "real_imag", "mag_phase"}, optional
            Type of output.

        Returns
        -------
        torch.Tensor
            STFT of the input signal. Has one more dimension than the input signal. The
            frequency bin axis is ``axis % x.ndim`` and the frame index axis is ``axis %
            x.ndim + 1``. For example if ``x.shape = (batch_size, channels, n_samples)``
            and ``axis = -1``, then ``y.shape = (batch_size, channels, n_bins,
            n_frames)``.

        """
        if x.is_cuda:
            analysis_window = self.analysis_window_cuda
        else:
            analysis_window = self.analysis_window_cpu
        analysis_window = analysis_window.type(x.real.dtype)

        axis = axis % x.ndim
        x = x.moveaxis(axis, -1)  # move time axis

        if not self._discard_trailing_samples:
            x = self._pad(x)

        input_shape = x.shape
        if x.ndim > 2:
            x = x.view(-1, input_shape[-1])

        if self.use_torch:
            x = torch.stft(
                input=x,
                win_length=self.frame_length,
                hop_length=self.hop_length,
                n_fft=self.n_fft,
                window=analysis_window,
                center=self.center,
                pad_mode=self.pad_mode,
                normalized=False,  # avoid default normalization
                onesided=self.onesided,
                return_complex=True,
            )
        else:
            if self.center:
                if self._center_padding_is_half_frame_length:
                    pad = self.frame_length // 2
                else:
                    pad = self.frame_length - self.hop_length
                x = F.pad(x, (pad, pad), mode=self.pad_mode)
            x = x.unfold(-1, self.frame_length, self.hop_length)
            x = x * analysis_window
            if self._pad_frames_right:
                padding = (0, self.n_fft - self.frame_length)
            else:
                padding = (self.n_fft - self.frame_length) // 2
                padding = (padding, self.n_fft - self.frame_length - padding)
            x = F.pad(x, padding)
            if self.onesided:
                x = torch.fft.rfft(x, dim=-1)
            else:
                x = torch.fft.fft(x, dim=-1)
            x = x.moveaxis(-1, -2)

        if self.compression_factor != 1:
            x = torch.polar(x.abs().pow(self.compression_factor), x.angle())
        x = x * self.scale_factor

        x = x.view(*input_shape[:-1], *x.shape[-2:])

        x = x.moveaxis(-1, axis)  # move frame index axis
        x = x.moveaxis(-1, axis)  # move frequency bin axis

        if return_type == "complex":
            return x
        elif return_type == "real_imag":
            return x.real, x.imag
        elif return_type == "mag_phase":
            return x.abs(), x.angle()
        else:
            raise ValueError(
                "return_type must be complex, real_imag or mag_phase, got "
                f"{return_type}"
            )

    def inverse(self, x, axis=-1, input_type="complex", length=None, real_output=True):
        """Compute the inverse STFT of a signal.

        Parameters
        ----------
        x : torch.Tensor
            Input signal.
        axis : int, optional
            Frame index axis. The frequency bin axis must be ``axis - 1``. Cannot be
            ``0``.
        input_type : {"complex", "real_imag", "mag_phase"}, optional
            Type of input.
        length : int, optional
            Length of the original signal to trim the output to. If ``None``, then no
            trimming is performed.
        real_output : bool, optional
            Whether the output can be assumed to be real-valued. If ``True``, the
            imaginary part of inverse FFTs is discarded before overlap-add and the
            output is real. If ``False``, the imaginary part is kept and the output is
            complex.

        Returns
        -------
        torch.Tensor
            Inverse STFT of the input signal. Has one less dimension than the input
            signal. The time axis is ``axis % x.ndim``. For example if ``x.shape =
            (batch_size, channels, n_bins, n_frames)`` and ``axis = -1``, then ``y.shape
            = (batch_size, channels, n_samples)``.

        """
        if input_type == "real_imag":
            real, imag = x
            x = torch.complex(real, imag)
        elif input_type == "mag_phase":
            mag, phase = x
            x = torch.polar(mag, phase)
        elif input_type != "complex":
            raise ValueError(
                "input_type must be complex, real_imag or mag_phase, got "
                f"{input_type}"
            )

        if x.is_cuda:
            analysis_window = self.analysis_window_cuda
            synthesis_window = self.synthesis_window_cuda
        else:
            analysis_window = self.analysis_window_cpu
            synthesis_window = self.synthesis_window_cpu
        analysis_window = analysis_window.type(x.real.dtype)
        synthesis_window = synthesis_window.type(x.real.dtype)

        x = x / self.scale_factor
        if self.compression_factor != 1:
            x = torch.polar(x.abs().pow(1 / self.compression_factor), x.angle())

        axis = axis % x.ndim
        if axis == 0:
            raise ValueError(
                "axis cannot be the first dimension since the frequency "
                "bin axis must be axis - 1"
            )
        x = x.moveaxis(axis, -1)  # move frame index axis
        x = x.moveaxis(axis - 1, -2)  # move frequency bin axis

        input_shape = x.shape
        if x.ndim > 3:
            x = x.reshape(-1, *input_shape[-2:])

        if self.use_torch:
            x = torch.istft(
                input=x,
                win_length=self.frame_length,
                hop_length=self.hop_length,
                n_fft=self.n_fft,
                window=synthesis_window,
                center=self.center,
                normalized=False,  # avoid default normalization
                onesided=self.onesided,
                return_complex=False,
            )
        else:
            if len(input_shape) == 2:
                x = x.unsqueeze(0)
            if self.onesided:
                x = torch.fft.irfft(x, dim=-2)
            else:
                x = torch.fft.ifft(x, dim=-2)
                if real_output:
                    x = x.real
            if self._pad_frames_right:
                x = x[:, : self.frame_length, :]
            else:
                padding = (self.n_fft - self.frame_length) // 2
                x = x[:, padding : padding + self.frame_length, :]
            x = x * synthesis_window.reshape(-1, 1)
            # replace NaNs with zeros where window is zero to pass latency test
            x[:, synthesis_window == 0] = torch.nan_to_num(x[:, synthesis_window == 0])
            output_size = (x.shape[-1] - 1) * self.hop_length + self.frame_length
            window_envelope = (
                F.fold(
                    (analysis_window * synthesis_window)
                    .expand(1, x.shape[-1], -1)
                    .transpose(1, 2),
                    output_size=(1, output_size),
                    kernel_size=(1, self.frame_length),
                    stride=(1, self.hop_length),
                )
                .squeeze(-2)
                .squeeze(-2)
            )
            x = (
                F.fold(
                    x,
                    output_size=(1, output_size),
                    kernel_size=(1, self.frame_length),
                    stride=(1, self.hop_length),
                )
                .squeeze(-2)
                .squeeze(-2)
            )
            if self.center and self.frame_length > self.hop_length:
                pad = self.frame_length - self.hop_length
                window_envelope = window_envelope[:, pad:-pad]
                x = x[:, pad:-pad]
            assert window_envelope.abs().min() > 1e-11
            x = x / window_envelope
            if len(input_shape) == 2:
                x = x.squeeze(0)

        if length is None:
            length = x.shape[-1]
        x = x[..., :length]

        x = x.view(*input_shape[:-2], -1)

        return x.moveaxis(-1, axis - 1)  # move time axis

    def _pad(self, x):
        # padding to fit the trailing samples in a frame
        if not self.center:
            samples = x.shape[-1]
        elif self.use_torch:
            # torch.stft pads n_fft // 2 on both sides
            samples = x.shape[-1] + 2 * (self.n_fft // 2)
        else:
            samples = x.shape[-1] + 2 * (self.frame_length - self.hop_length)
        frames = self._frame_count(samples)
        padding = (frames - 1) * self.hop_length + self.frame_length - samples
        return F.pad(x, (0, padding), mode=self.pad_mode)

    def _frame_count(self, samples):
        # number of frames required to fit samples
        return math.ceil(max(samples - self.frame_length, 0) / self.hop_length) + 1


def _check_window(window, frame_length):
    if window is None:
        window = "boxcar"
    if isinstance(window, str):
        window = functools.partial(scipy.signal.get_window, window)
    if callable(window):
        window = window(frame_length)
    if isinstance(window, np.ndarray):
        window = torch.from_numpy(window)
        assert window.ndim == 1, window.shape
        assert len(window) == frame_length, (len(window), frame_length)
    else:
        raise ValueError(
            "window must be a str, callable or np.ndarray, ",
            f"got {window.__class__.__name__}",
        )
    return window


def mauler_windows(analysis_length, synthesis_length, n_fft):
    """Dual STFT windows.

    Calculates a pair of analysis and synthesis windows as in [1] and [2].

    .. [1] D. Mauler and R. Martin, "A low delay, variable resolution, perfect
       reconstruction spectral analysis-synthesis system for speech enhancement", in
       Proc. EUSIPCO, 2007.
    .. [2] D. Mauler and R. Martin, "Improved reproduction of stops in noise reduction
       systems with adaptive windows and nonstationarity detection", in EURASIP J. Adv.
       Signal Process., 2009.

    Parameters
    ----------
    analysis_length : int
        Analysis frame length. Equal to ``K - d`` in [1] and [2].
    synthesis_length : int
        Synthesis frame length. Must be even. Equal to ``2 * M`` in [1] and [2].
    n_fft : int
        FFT length. Equal to ``K`` in [1] and [2].

    Returns
    -------
    analysis_window : numpy.ndarray
        Analysis window. Shape ``(n_fft,)``.
    synthesis_window : numpy.ndarray
        Synthesis window.  Shape ``(n_fft,)``.

    """
    assert analysis_length <= n_fft
    assert synthesis_length <= analysis_length
    assert synthesis_length % 2 == 0
    K = n_fft
    d = n_fft - analysis_length
    M = synthesis_length // 2

    def hann(L, n):
        return 0.5 * (1 - np.cos(2 * np.pi * n / L))

    analysis_window = np.zeros(K)
    analysis_window[d : K - 2 * M] = (
        hann(2 * (K - M - d), np.arange(K - 2 * M - d)) ** 0.5
    )
    analysis_window[K - 2 * M : K - M] = (
        hann(2 * (K - M - d), np.arange(M) + K - 2 * M - d) ** 0.5
    )
    analysis_window[K - M : K] = hann(2 * M, np.arange(M) + M) ** 0.5

    synthesis_window = np.zeros(K)
    # hack to prevent normalization by 0 when analysis_length == synthesis_length
    i = analysis_length == synthesis_length
    synthesis_window[K - 2 * M + i : K - M] = (
        hann(2 * M, np.arange(i, M))
        / hann(2 * (K - M - d), np.arange(i, M) + K - 2 * M - d) ** 0.5
    )
    synthesis_window[K - M : K] = hann(2 * M, np.arange(M) + M) ** 0.5

    return analysis_window, synthesis_window

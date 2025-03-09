from functools import partial
from multiprocessing import Pool, cpu_count
from typing import override

import numpy as np
import soxr
from pesq._pesq import (
    USAGE,
    USAGE_BATCH,
    _check_fs_mode,
    _pesq_inner,
    _processor_mapping,
)
from pesq.cypesq import PesqError

from ..utils import soxr_output_lenght
from .base import BaseMetric
from .registry import MetricRegistry


@MetricRegistry.register("pesq")
class PESQMetric(BaseMetric):
    """Perceptual Evaluation of Speech Quality (PESQ).

    Proposed in [1].

    .. [1] A. W. Rix, J. G. Beerends, M. P. Hollier and A. P. Hekstra, "Perceptual
       evaluation of speech quality (PESQ)—A new method for speech quality assessment
       of telephone networks and codecs", in Proc. ICASSP, 2001.

    Parameters
    ----------
    fs : int, optional
        Input sampling frequency.
    mode : {"nb", "wb"}, optional
        Wide-band (`"wb"`) or narrow-band (`"nb"`).
    normalized : bool, optional
        Whether to return a normalized score between 0 and 1.
    pesq_fs : {8000, 16000}, optional
        The internal sampling frequency used by the PESQ library. Input signals are
        resampled from ``fs`` to ``pesq_fs`` before calculating PESQ. Must be either
        8000 or 16000 Hz.
    multiprocessing : bool, optional
        Whether to use multiprocessing for batched inputs.

    """

    def __init__(
        self,
        fs=16000,
        mode="wb",
        normalized=False,
        pesq_fs=16000,
        multiprocessing=True,
    ):
        self.fs = fs
        self.mode = mode
        self.normalized = normalized
        self.pesq_fs = pesq_fs
        self.multiprocessing = multiprocessing

    @override
    def compute(self, x, y, lengths):
        if self.fs != self.pesq_fs:
            x = soxr.resample(x.T, self.fs, self.pesq_fs).T
            y = soxr.resample(y.T, self.fs, self.pesq_fs).T
            lengths = [soxr_output_lenght(l_, self.fs, self.pesq_fs) for l_ in lengths]
        if x.shape[0] == 1:
            output = [
                _pesq(
                    self.pesq_fs,
                    y[0, : lengths[0]],
                    x[0, : lengths[0]],
                    mode=self.mode,
                )
            ]
        elif self.multiprocessing:
            output = _batched_pesq(self.pesq_fs, y, x, mode=self.mode, lengths=lengths)
        else:
            output = [
                _pesq(
                    self.pesq_fs,
                    y[i, : lengths[i]],
                    x[i, : lengths[i]],
                    mode=self.mode,
                )
                for i in range(x.shape[0])
            ]
        output = [float("nan") if isinstance(o, Exception) else o for o in output]
        output = np.array(output)
        if x.ndim == 1:
            output = output.item()
        if self.normalized:
            # see https://github.com/ludlows/PESQ/issues/13
            if self.mode == "nb":
                # min = 1.016843313292765
                min = 1.0
                max = 4.548638319075995
            elif self.mode == "wb":
                # min = 1.042694226789194
                min = 1.0
                max = 4.643888749336258
            else:
                raise ValueError(f"mode must be 'nb' or 'wb', got '{self.mode}'")
            output = (output - min) / (max - min)
            if any(output < 0) or any(output > 1):
                raise RuntimeError(f"normalized PESQ score is out of bounds: {output}")
        return output


def _pesq(fs, ref, deg, mode="wb", on_error=PesqError.RAISE_EXCEPTION):
    _check_fs_mode(mode, fs, USAGE)
    return _pesq_inner(ref, deg, fs, mode, on_error)


def _batched_pesq(
    fs,
    ref,
    deg,
    mode,
    n_processor=cpu_count(),
    on_error=PesqError.RAISE_EXCEPTION,
    lengths=None,
):
    """Batched PESQ with lengths argument support.

    This is a copy/paste of https://github.com/ludlows/PESQ/pull/46 and should be
    removed if the PR ever gets approved.
    """
    _check_fs_mode(mode, fs, USAGE_BATCH)
    # check dimension
    if len(ref.shape) == 1:
        if lengths is not None:
            raise ValueError("cannot provide lengths if ref is 1D")
        if len(deg.shape) == 1 and ref.shape == deg.shape:
            return [_pesq_inner(ref, deg, fs, mode, PesqError.RETURN_VALUES)]
        elif len(deg.shape) == 2 and ref.shape[-1] == deg.shape[-1]:
            if n_processor <= 0:
                pesq_score = [np.nan for i in range(deg.shape[0])]
                for i in range(deg.shape[0]):
                    pesq_score[i] = _pesq_inner(ref, deg[i, :], fs, mode, on_error)
                return pesq_score
            else:
                with Pool(n_processor) as p:
                    return p.map(
                        partial(_pesq_inner, ref, fs=fs, mode=mode, on_error=on_error),
                        [deg[i, :] for i in range(deg.shape[0])],
                    )
        else:
            raise ValueError("The shapes of `deg` is invalid!")
    elif len(ref.shape) == 2:
        if deg.shape == ref.shape:
            if lengths is None:
                lengths = [ref.shape[-1] for _ in range(ref.shape[0])]
            elif len(lengths) != ref.shape[0]:
                raise ValueError("len(lengths) does not match the batch size")
            if n_processor <= 0:
                pesq_score = [np.nan for i in range(deg.shape[0])]
                for i in range(deg.shape[0]):
                    pesq_score[i] = _pesq_inner(
                        ref[i, : lengths[i]],
                        deg[i, : lengths[i]],
                        fs,
                        mode,
                        on_error,
                    )
                return pesq_score
            else:
                return _processor_mapping(
                    _pesq_inner,
                    [
                        (
                            ref[i, : lengths[i]],
                            deg[i, : lengths[i]],
                            fs,
                            mode,
                            on_error,
                        )
                        for i in range(deg.shape[0])
                    ],
                    n_processor,
                )
        else:
            raise ValueError("The shape of `deg` is invalid!")
    else:
        raise ValueError("The shape of `ref` should be either 1D or 2D!")

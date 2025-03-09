from typing import override

import numpy as np
import torch

from ..utils import snr
from .base import BaseMetric
from .registry import MetricRegistry


class _BaseSNRMetric(BaseMetric):
    to_numpy = False

    def __init__(self, scale_invariant=False, zero_mean=True, eps=1e-7):
        self.scale_invariant = scale_invariant
        self.zero_mean = zero_mean
        self.eps = eps

    @override
    def compute(self, x, y, lengths):
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x)
        if isinstance(y, np.ndarray):
            y = torch.from_numpy(y)
        if isinstance(lengths, np.ndarray):
            lengths = torch.from_numpy(lengths)
        return snr(
            x,
            y,
            scale_invariant=self.scale_invariant,
            zero_mean=self.zero_mean,
            eps=self.eps,
            lengths=lengths,
        )


@MetricRegistry.register("snr")
class SNRMetric(_BaseSNRMetric):
    """Signal-to-noise ratio (SNR).

    Parameters
    ----------
    zero_mean : bool
        Whether to subtract the mean to the signals before calculating.
    eps : float
        Small value to avoid division by zero.

    """

    def __init__(self, zero_mean=True, eps=1e-7):
        super().__init__(scale_invariant=False, zero_mean=zero_mean, eps=eps)


@MetricRegistry.register("sisnr")
class SISNRMetric(_BaseSNRMetric):
    """Scale-invariant signal-to-noise ratio (SNR).

    Parameters
    ----------
    zero_mean : bool
        Whether to subtract the mean to the signals before calculating.
    eps : float
        Small value to avoid division by zero.

    """

    def __init__(self, zero_mean=True, eps=1e-7):
        super().__init__(scale_invariant=True, zero_mean=zero_mean, eps=eps)

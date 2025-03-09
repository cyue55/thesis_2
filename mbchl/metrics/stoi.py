from typing import override

import batch_pystoi

from .base import BaseMetric
from .registry import MetricRegistry


class _BaseSTOIMetric(BaseMetric):
    def __init__(self, fs=16000, extended=False):
        self.fs = fs
        self.extended = extended

    @override
    def compute(self, x, y, lengths):
        return batch_pystoi.stoi(x, y, self.fs, extended=self.extended, lengths=lengths)


@MetricRegistry.register("stoi")
class STOIMetric(_BaseSTOIMetric):
    """Short-Time Objective Intelligibility (STOI).

    Proposed in [1].

    .. [1] C. H. Taal, R. C. Hendriks, R. Heusdens and J. Jensen, "An algorithm for
       intelligibility prediction of time-frequency weighted noisy speech", in IEEE
       Trans. Audio, Speech, Lang. Process., 2011.

    Parameters
    ----------
    fs : int, optional
        Input sampling frequency.

    """

    def __init__(self, fs=16000):
        super().__init__(fs, extended=False)


@MetricRegistry.register("estoi")
class ESTOIMetric(_BaseSTOIMetric):
    """Extended Short-Time Objective Intelligibility (ESTOI).

    Proposed in [1].

    .. [1] J. Jensen and C. H. Taal, "An algorithm for predicting the intelligibility of
       speech masked by modulated noise maskers", in IEEE/ACM Trans. Audio, Speech,
       Lang. Process., 2016.

    Parameters
    ----------
    fs : int, optional
        Input sampling frequency.

    """

    def __init__(self, fs=16000):
        super().__init__(fs, extended=True)

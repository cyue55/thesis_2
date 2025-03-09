import numpy as np
import pytest
import torch

from mbchl.signal.mel import MelSpectrogram


def test_melspectrogram(np_rng, torch_rng):
    librosa = pytest.importorskip("librosa")
    x = torch.randn(16000, generator=torch_rng, dtype=torch.float64)
    # use same parameters as in torchmetrics DNSMOS
    melspec = MelSpectrogram(
        frame_length=320 + 1,
        hop_length=160,
        n_fft=320 + 1,
        window="hann",
        center=True,
        pad_mode="constant",
        normalized=False,
        use_torch=False,
        n_filters=120,
        f_min=0.0,
        f_max=None,
        fs=16000,
        norm="slaney",
        scale="slaney",
        power=2,
        log=False,
        dtype=torch.float64,
        _discard_trailing_samples=True,
        _center_padding_is_half_frame_length=True,
    )
    y_ours = melspec(x).numpy()
    y_librosa = librosa.feature.melspectrogram(
        y=x.numpy(),
        sr=16000,
        n_fft=320 + 1,
        hop_length=160,
        n_mels=120,
    )
    assert np.allclose(y_ours, y_librosa, atol=5e-5, rtol=0)
    assert np.allclose(y_ours, y_librosa, atol=0, rtol=2e-4)

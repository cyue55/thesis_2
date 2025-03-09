import numpy as np
import pytest

from mbchl.utils import pad, set_dbfs, set_snr


@pytest.mark.parametrize("shape", [(8, 10, 12)])
@pytest.mark.parametrize("padding", [3])
@pytest.mark.parametrize("axis", [0, 1, 2])
@pytest.mark.parametrize("where", ["left", "right", "both"])
def test_pad(np_rng, shape, padding, axis, where):
    x = np_rng.standard_normal(shape)
    y = pad(x, n=padding, axis=axis, where=where)
    if where == "left":
        assert y.shape[axis] == x.shape[axis] + padding
        assert np.all(y.take(np.arange(padding), axis=axis) == 0)
    elif where == "right":
        assert y.shape[axis] == x.shape[axis] + padding
        assert np.all(y.take(np.arange(-padding, 0), axis=axis) == 0)
    elif where == "both":
        assert y.shape[axis] == x.shape[axis] + 2 * padding
        assert np.all(y.take(np.arange(padding), axis=axis) == 0)
        assert np.all(y.take(np.arange(-padding, 0), axis=axis) == 0)
    else:
        raise ValueError(f"where must be left, right or both, got {where}")


@pytest.mark.parametrize("n", [100])
@pytest.mark.parametrize("snr", [-10, 0, 5])
@pytest.mark.parametrize("zero_mean", [False, True])
@pytest.mark.parametrize("speech_mean", [0, 1])
@pytest.mark.parametrize("noise_mean", [0, 1])
def test_set_snr(np_rng, n, snr, zero_mean, speech_mean, noise_mean):
    speech = np_rng.standard_normal(n) + speech_mean
    noise = np_rng.standard_normal(n) + noise_mean
    new_noise = set_snr(speech, noise, snr, zero_mean=zero_mean)
    _snr = 10 * np.log10(
        np.sum((speech - zero_mean * speech.mean()) ** 2)
        / np.sum((new_noise - zero_mean * new_noise.mean()) ** 2)
    )
    assert np.isclose(snr, _snr)
    if zero_mean:
        assert np.isclose(new_noise.mean(), noise.mean())


@pytest.mark.parametrize("n", [100])
@pytest.mark.parametrize("dbfs", [-10, 0, 5])
@pytest.mark.parametrize("mode", ["peak", "rms"])
def test_set_dbfs(np_rng, n, dbfs, mode):
    x = np_rng.standard_normal(n)
    new_x = set_dbfs(x, dbfs, mode=mode)
    if mode == "peak":
        assert np.isclose(np.max(np.abs(new_x)), 10 ** (dbfs / 20))
    elif mode == "rms":
        assert np.isclose(np.sqrt(np.mean(new_x**2)), 10 ** (dbfs / 20))
    else:
        raise ValueError(f"`mode` must be 'peak' or 'rms', got {mode}")


def test_set_dbfs_sine():
    n = 100
    # sine wave
    x = np.sin(2 * np.pi * np.arange(n) / n)
    x = set_dbfs(x, 0, mode="peak")
    assert x.max() == 1
    x = set_dbfs(x, 0, mode="rms")
    assert np.isclose(x.max(), np.sqrt(2))
    x = set_dbfs(x, 0, mode="aes17")
    assert x.max() == 1
    x = set_dbfs(x, -6, mode="peak")
    assert np.isclose(x.max(), 0.5, atol=0.01)
    # square wave
    x = np.sign(np.sin(2 * np.pi * np.arange(n) / n))
    x = set_dbfs(x, 3, mode="aes17")
    assert np.isclose(x.max(), 1, atol=0.01)

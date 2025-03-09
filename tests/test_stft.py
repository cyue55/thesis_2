import pytest
import torch

from mbchl.signal.stft import STFT


@pytest.mark.parametrize("n_samples", [2500])
@pytest.mark.parametrize("frame_length", [512])
@pytest.mark.parametrize("hop_length", [128, 256])
@pytest.mark.parametrize("n_fft", [None, 1024])
@pytest.mark.parametrize("window", ["hann", "mauler"])
@pytest.mark.parametrize("normalized", [False, True])
@pytest.mark.parametrize("onesided", [False, True])
@pytest.mark.parametrize("compression_factor", [1.0, 0.5])
@pytest.mark.parametrize("scale_factor", [1.0, 0.15])
@pytest.mark.parametrize("use_torch", [False, True])
def test_stft(
    torch_rng,
    n_samples,
    frame_length,
    hop_length,
    n_fft,
    window,
    normalized,
    onesided,
    compression_factor,
    scale_factor,
    use_torch,
):
    if window == "mauler" and frame_length < 2 * hop_length:
        with pytest.raises(ValueError):
            stft = STFT(
                frame_length=frame_length,
                hop_length=hop_length,
                n_fft=n_fft,
                window=window,
                compression_factor=compression_factor,
                scale_factor=scale_factor,
                normalized=normalized,
                onesided=onesided,
                use_torch=use_torch,
            )
        return
    else:
        stft = STFT(
            frame_length=frame_length,
            hop_length=hop_length,
            n_fft=n_fft,
            window=window,
            compression_factor=compression_factor,
            scale_factor=scale_factor,
            normalized=normalized,
            onesided=onesided,
            use_torch=use_torch,
        )
    x = torch.randn(n_samples, generator=torch_rng)
    y = stft(x)
    if (
        window == "mauler"
        and use_torch
        and (n_fft is not None or frame_length != 2 * hop_length)
    ):
        with pytest.raises(RuntimeError):
            z = stft.inverse(y, length=n_samples)
        return
    else:
        z = stft.inverse(y, length=n_samples)
    assert torch.allclose(x, z, rtol=0, atol=1e-6)
    assert torch.allclose(x, z, rtol=6e-4, atol=0)


@pytest.mark.parametrize(
    "shape, axis",
    [
        [(2500, 2, 3, 4), 0],
        [(2, 2500, 3, 4), 1],
        [(2, 3, 2500, 4), 2],
        [(2, 3, 4, 2500), 3],
        [(2, 3, 4, 2500), -1],
        [(2, 3, 2500, 4), -2],
        [(2, 2500, 3, 4), -3],
        [(2500, 2, 3, 4), -4],
    ],
)
@pytest.mark.parametrize("use_torch", [False, True])
def test_axis(torch_rng, shape, axis, use_torch):
    x = torch.randn(shape, generator=torch_rng)
    stft = STFT(use_torch=use_torch)
    y = stft.inverse(
        stft(x, axis=axis),
        axis=axis if axis < 0 else axis + 1,
        length=shape[axis],
    )
    assert torch.allclose(x, y, rtol=0, atol=1e-6)
    assert torch.allclose(x, y, rtol=5e-3, atol=0)


@pytest.mark.parametrize("n_samples", [2500])
@pytest.mark.parametrize("frame_length", [512])
@pytest.mark.parametrize("hop_length", [256])
@pytest.mark.parametrize("n_fft", [None, 1024])
@pytest.mark.parametrize("window", ["hann"])
@pytest.mark.parametrize("normalized", [False, True])
@pytest.mark.parametrize("onesided", [False, True])
@pytest.mark.parametrize("compression_factor", [1.0, 0.5])
@pytest.mark.parametrize("scale_factor", [1.0, 0.15])
def test_stft_use_torch(
    torch_rng,
    n_samples,
    frame_length,
    hop_length,
    n_fft,
    window,
    normalized,
    onesided,
    compression_factor,
    scale_factor,
):
    # if hop_length is not half of frame_length then skip because output shapes will
    # be different
    if hop_length != frame_length // 2:
        return
    stft_torch = STFT(
        frame_length=frame_length,
        hop_length=hop_length,
        n_fft=n_fft,
        window=window,
        compression_factor=compression_factor,
        scale_factor=scale_factor,
        normalized=normalized,
        onesided=onesided,
        use_torch=True,
    )
    stft_no_torch = STFT(
        frame_length=frame_length,
        hop_length=hop_length,
        n_fft=n_fft,
        window=window,
        compression_factor=compression_factor,
        scale_factor=scale_factor,
        normalized=normalized,
        onesided=onesided,
        use_torch=False,
        _pad_frames_right=False,
    )
    x = torch.randn(n_samples, generator=torch_rng)
    y_torch = stft_torch(x)
    y_no_torch = stft_no_torch(x)
    assert torch.allclose(y_torch, y_no_torch)


@pytest.mark.parametrize("n_frames", [20])
@pytest.mark.parametrize("frame_length", [512])
@pytest.mark.parametrize("hop_length", [256])
@pytest.mark.parametrize("n_fft", [None, 1024])
@pytest.mark.parametrize("window", ["hann"])
@pytest.mark.parametrize("normalized", [False, True])
@pytest.mark.parametrize("onesided", [False, True])
@pytest.mark.parametrize("compression_factor", [1.0, 0.5])
@pytest.mark.parametrize("scale_factor", [1.0, 0.15])
def test_istft_use_torch(
    torch_rng,
    n_frames,
    frame_length,
    hop_length,
    n_fft,
    window,
    normalized,
    onesided,
    compression_factor,
    scale_factor,
):
    # if hop_length is not half of frame_length then skip because output shapes will
    # be different
    if hop_length != frame_length // 2:
        return
    # if not onesided then skip because torch.stft assumes the output is real
    if not onesided:
        return
    stft_torch = STFT(
        frame_length=frame_length,
        hop_length=hop_length,
        n_fft=n_fft,
        window=window,
        compression_factor=compression_factor,
        scale_factor=scale_factor,
        normalized=normalized,
        onesided=onesided,
        use_torch=True,
    )
    stft_no_torch = STFT(
        frame_length=frame_length,
        hop_length=hop_length,
        n_fft=n_fft,
        window=window,
        compression_factor=compression_factor,
        scale_factor=scale_factor,
        normalized=normalized,
        onesided=onesided,
        use_torch=False,
        _pad_frames_right=False,
    )
    n_fft = n_fft or frame_length
    n_bins = n_fft // 2 + 1 if onesided else n_fft
    x = torch.randn(n_bins, n_frames, generator=torch_rng, dtype=torch.complex64)
    y_torch = stft_torch.inverse(x)
    y_no_torch = stft_no_torch.inverse(x)
    assert torch.allclose(y_torch, y_no_torch)


def test_mauler_latency(np_rng):
    stft = STFT(window="mauler", frame_length=512, hop_length=40, use_torch=False)
    latency = 80
    for i in range(100):
        length = np_rng.integers(1600, 3200)
        input_ = torch.randn(length)
        if i == 0:
            # try special case where the first nan is exactly at j=latency
            nan_start = latency
        else:
            nan_start = np_rng.integers(latency, input_.shape[-1] - 1)
        input_[..., nan_start:] = float("nan")
        output = stft.inverse(stft(input_))
        j = next(k for k in range(output.shape[-1]) if output[..., k].isnan().any())
        assert j >= nan_start - latency + 1

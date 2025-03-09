import numpy as np
import pytest
import torch

from mbchl.metrics import MetricRegistry


# unfortunately the minimum length has to be quite high to avoid errors and warnings
# raised by PESQ and STOI for short inputs
@pytest.mark.parametrize("min_length", [7000])
@pytest.mark.parametrize("metric", MetricRegistry.keys())
@pytest.mark.parametrize(
    "shape, axis",
    [
        [(10000, 2, 3), 0],
        [(2, 10000, 3), 1],
        [(2, 3, 10000), 2],
        [(2, 3, 10000), -1],
        [(2, 10000, 3), -2],
        [(10000, 2, 3), -3],
    ],
)
def test_nd(np_rng, metric, min_length, shape, axis, metric_kw={}):
    if metric in ["dnsmos", "nisqa"]:
        return
    # randomize lengths
    lengths = np_rng.integers(
        min_length,
        shape[axis],
        [n for i, n in enumerate(shape) if i != axis % len(shape)],
    )

    # create unbatched targets with length lengths
    y_unbatched = [np_rng.standard_normal(length) for length in lengths.flatten()]

    # pad targets and make batch
    y_batched = np.stack(
        [np.pad(t, (0, shape[axis] - t.shape[-1])) for t in y_unbatched]
    ).reshape(*lengths.shape, shape[axis])
    y_batched = np.moveaxis(y_batched, -1, axis)

    # create batched and unbatched inputs
    x_batched = y_batched + 0.5 * np_rng.standard_normal(shape)
    x_unbatched = [
        x[:length]
        for x, length in zip(
            np.moveaxis(x_batched, axis, -1).reshape(-1, shape[axis]), lengths.flatten()
        )
    ]

    # init metric
    metric_cls = MetricRegistry.get(metric)
    metric_obj = metric_cls(**metric_kw)

    # 2 ways of calculating: either batch processing...
    batched_metrics = metric_obj(x_batched, y_batched, axis=axis, lengths=lengths)

    # ...or one-by-one
    metrics = [metric_obj(x, y) for x, y in zip(x_unbatched, y_unbatched)]

    # both should give the same result
    assert batched_metrics.shape == lengths.shape
    assert np.allclose(batched_metrics.flatten(), metrics, atol=2e-5)


@pytest.mark.parametrize("metric", ["pesq", "stoi", "estoi"])
@pytest.mark.parametrize("fs", [48000, 44100, 32000, 22050])
@pytest.mark.parametrize(
    "shape, axis",
    [
        [[10000, 3], 0],
    ],
)
def test_pesq_stoi_nd(np_rng, metric, shape, axis, fs):
    min_length = round(7000 * fs / 16000)
    shape = (round(10000 * fs / 16000), 3)
    axis = 0
    test_nd(np_rng, metric, min_length, shape, axis, metric_kw={"fs": fs})


@pytest.mark.parametrize("length", [7000])
@pytest.mark.parametrize("metric", MetricRegistry.keys())
@pytest.mark.parametrize("shape", [(10000,)])
def test_1d(np_rng, metric, length, shape):
    if metric in ["dnsmos", "nisqa"]:
        return
    x = np_rng.standard_normal(shape)
    y = np_rng.standard_normal(shape)
    metric_cls = MetricRegistry.get(metric)
    metric_obj = metric_cls()
    z_1 = metric_obj(x[:length], y[:length], axis=0)
    z_2 = metric_obj(x, y, lengths=length, axis=0)
    assert isinstance(z_1, float)
    assert isinstance(z_2, float)
    assert np.isclose(z_1, z_2)
    z_3 = metric_obj(x[:length], y[:length], axis=-1)
    z_4 = metric_obj(x, y, lengths=length, axis=-1)
    assert isinstance(z_3, float)
    assert isinstance(z_4, float)
    assert z_1 == z_3
    assert z_2 == z_4
    assert np.isclose(z_3, z_4)


@pytest.mark.parametrize("metric", MetricRegistry.keys())
def test_errors(np_rng, metric):
    if metric == "dnsmos":
        pytest.importorskip("librosa")
        pytest.importorskip("onnxruntime")
    if metric == "nisqa":
        pytest.importorskip("librosa")
    metric_cls = MetricRegistry.get(metric)
    metric_obj = metric_cls()
    with pytest.raises(ValueError):
        metric_obj(
            np_rng.standard_normal(10000),
            np_rng.standard_normal(16000),
        )
    with pytest.raises(TypeError):
        metric_obj(
            np_rng.standard_normal(10000),
            np_rng.standard_normal(10000),
            axis="foo",
        )
    with pytest.raises(TypeError):
        metric_obj(
            np_rng.standard_normal(10000),
            np_rng.standard_normal(10000),
            lengths="foo",
        )
    with pytest.raises(TypeError):
        metric_obj(
            np_rng.standard_normal((1, 10000)),
            np_rng.standard_normal((1, 10000)),
            lengths="foo",
        )
    with pytest.raises(TypeError):
        metric_obj(
            np_rng.standard_normal(10000),
            np_rng.standard_normal(10000),
            lengths=[1000],
        )
    with pytest.raises(TypeError):
        metric_obj(
            np_rng.standard_normal((1, 10000)),
            np_rng.standard_normal((1, 10000)),
            lengths=1000,
        )
    with pytest.raises(ValueError):
        metric_obj(
            np_rng.standard_normal((1, 10000)),
            np_rng.standard_normal((1, 10000)),
            lengths=[1000, 1000],
        )


@pytest.mark.parametrize("metric", MetricRegistry.keys())
def test_tensor_input(torch_rng, metric):
    if metric == "dnsmos":
        pytest.importorskip("librosa")
        pytest.importorskip("onnxruntime")
    if metric == "nisqa":
        pytest.importorskip("librosa")
    metric_cls = MetricRegistry.get(metric)
    metric_obj = metric_cls()
    x = torch.rand(2, 10000, generator=torch_rng)
    y = torch.rand(2, 10000, generator=torch_rng)
    lengths = torch.tensor([7000, 8000])
    out = metric_obj(x, y, lengths=lengths)
    if metric == "nisqa":
        assert np.allclose(out, np.array([0.8972699, 0.83377832]))


@pytest.mark.parametrize("metric", MetricRegistry.keys())
def test_cuda_input(torch_rng, metric):
    if metric == "dnsmos":
        pytest.importorskip("librosa")
        pytest.importorskip("onnxruntime")
    if metric == "nisqa":
        pytest.importorskip("librosa")
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    metric_cls = MetricRegistry.get(metric)
    metric_obj = metric_cls()
    x = torch.rand(2, 10000, generator=torch_rng).cuda()
    y = torch.rand(2, 10000, generator=torch_rng).cuda()
    lengths = torch.tensor([7000, 8000])
    metric_obj(x, y, lengths=lengths)

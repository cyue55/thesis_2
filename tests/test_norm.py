import pytest
import torch

from mbchl.layers import CausalInstanceNorm, CausalLayerNorm

BATCH_SIZE = 16
CHANNELS = 3
FREQS = 64
FRAMES = 100


@pytest.mark.parametrize("norm", [CausalLayerNorm, CausalInstanceNorm])
def test_causal_norm(torch_rng, norm):
    norm = CausalLayerNorm(CHANNELS)
    norm.eval()
    _test_normalization(torch_rng, norm, aggregated_dims=(1, 2, 3))
    _test_causality(torch_rng, norm)


def _test_normalization(torch_rng, norm, aggregated_dims):
    x = torch.randn(BATCH_SIZE, CHANNELS, FREQS, FRAMES, generator=torch_rng)
    y = norm(x)
    for i in range(1, FRAMES):
        mean = x[..., : i + 1].mean(aggregated_dims, keepdims=True)
        std = x[..., : i + 1].std(aggregated_dims, keepdims=True, unbiased=False)
        assert torch.allclose(y[..., [i]], (x[..., [i]] - mean) / std, atol=1e-7)


def _test_causality(torch_rng, norm):
    for i in range(1, FRAMES):
        x = torch.randn(BATCH_SIZE, CHANNELS, FREQS, FRAMES, generator=torch_rng)
        x[..., i] = float("nan")
        y = norm(x)
        assert not y[..., :i].isnan().any()

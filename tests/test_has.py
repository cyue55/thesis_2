import copy
from contextlib import nullcontext

import pytest
import torch

from mbchl.has import HARegistry
from mbchl.utils import random_audiogram

LATENCY_MIN_LENGTH = 1600
LATENCY_MAX_LENGTH = 3200


class _TestModel:
    default_model_kwargs = {}
    forward_model_kwargs = None
    enhance_model_kwargs = None
    latency_model_kwargs = None
    params_model_kwargs = None
    latency = None
    n_params = None
    enhance_input_output_channels = 2
    enhance_samples = 10069
    latency_repeats = 100
    audiogram_support = True

    def init_model(self, model_kwargs=None):
        if model_kwargs is None:
            model_kwargs = self.default_model_kwargs
        net = HARegistry.get(self.model_name)(**model_kwargs)
        net.eval()
        return net

    def test_forward(self):
        net = self.init_model(self.forward_model_kwargs)
        net(*self.forward_args)

    def test_forward_with_audiogram(self):
        audiogram = random_audiogram(dtype="float32")
        audiogram = torch.from_numpy(audiogram).unsqueeze(0)
        kwargs = (
            self.default_model_kwargs
            if self.forward_model_kwargs is None
            else self.forward_model_kwargs
        )
        kwargs = copy.deepcopy(kwargs)
        kwargs = self._update_kwargs_for_audiogram(kwargs, audiogram.numel())
        net = self.init_model(kwargs)
        with nullcontext() if self.audiogram_support else pytest.raises(
            NotImplementedError
        ):
            net(*self.forward_args, audiogram=audiogram)

    def test_enhance(self, torch_rng):
        net = self.init_model(self.enhance_model_kwargs)
        input_ = torch.randn(
            self.enhance_input_output_channels,
            self.enhance_samples,
            generator=torch_rng,
        )
        output = net.enhance(input_)
        assert input_.shape == output.shape

    def test_latency(self, np_rng):
        if self.latency is None:
            return
        net = self.init_model(self.latency_model_kwargs)
        latency = self.latency(net)
        for i in range(self.latency_repeats):
            length = np_rng.integers(LATENCY_MIN_LENGTH, LATENCY_MAX_LENGTH)
            input_ = torch.randn(self.enhance_input_output_channels, length)
            if latency == float("inf"):
                nan_start = np_rng.integers(0, input_.shape[-1])
            else:
                if i == 0:
                    # try special case where the first nan is exactly at j=latency
                    nan_start = latency
                else:
                    nan_start = np_rng.integers(latency, input_.shape[-1])
            input_[..., nan_start:] = float("nan")
            output = net.enhance(input_)
            j = next(k for k in range(output.shape[-1]) if output[..., k].isnan().any())
            assert j >= nan_start - latency + 1

    def test_n_params(self):
        if self.n_params is None:
            return
        net = self.init_model(self.params_model_kwargs)
        assert net.count_params() == self.n_params

    def _update_kwargs_for_audiogram(self, kwargs, emb_dim):
        kwargs["audiogram"] = True
        kwargs["emb_dim"] = emb_dim
        return kwargs


class TestBSRNN(_TestModel):
    model_name = "bsrnn"
    forward_args = [torch.randn(1, 2, 257, 160, dtype=torch.cfloat)]
    latency = lambda self, x: 240  # noqa: E731
    default_model_kwargs = {
        "input_channels": 2,
        "reference_channels": [0, 1],
        "layers": 1,
        "base_channels": 1,
        "stft_kw": {
            "frame_length": 480,
            "hop_length": 240,
            "n_fft": 512,
        },
        "stft_future_frames": 1,
        "causal": True,
    }
    params_model_kwargs = {}
    n_params = 3_407_180
    audiogram_support = True


class TestConvTasNet(_TestModel):
    model_name = "convtasnet"
    forward_args = [torch.randn(1, 2, 8000)]
    latency = None  # TODO
    default_model_kwargs = {
        "input_channels": 2,
        "reference_channels": [0, 1],
        "filters": 1,
        "bottleneck_channels": 1,
        "hidden_channels": 1,
        "skip_channels": 1,
        "layers": 1,
        "repeats": 1,
    }
    params_model_kwargs = {}
    n_params = 4_935_217

    def _update_kwargs_for_audiogram(self, kwargs, emb_dim):
        kwargs["audiogram"] = True
        kwargs["emb_dim"] = emb_dim
        kwargs["fusion_layer"] = "film"
        return kwargs


class TestFFNN(_TestModel):
    model_name = "ffnn"
    forward_args = [torch.randn(1, 768, 160)]
    latency = None  # TODO
    default_model_kwargs = {
        "input_channels": 2,
        "reference_channels": [0, 1],
        "hidden_sizes": [1, 1],
    }
    params_model_kwargs = {}
    n_params = 1_509_440

    def _update_kwargs_for_audiogram(self, kwargs, emb_dim):
        kwargs["audiogram"] = True
        kwargs["emb_dim"] = emb_dim
        kwargs["fusion_layer"] = "film"
        return kwargs


class TestiNeuBe(_TestModel):
    model_name = "ineube"
    forward_args = [torch.randn(1, 2, 257, 160, dtype=torch.cfloat)]
    latency = None  # TODO
    default_model_kwargs = {
        "net1_cls": "bsrnn",
        "net1_kw": {
            "input_channels": 2,
            "reference_channels": [0, 1],
            "layers": 1,
            "base_channels": 1,
        },
        "net2_cls": "bsrnn",
        "net2_kw": {
            "input_channels": 6,
            "reference_channels": [0, 1],
            "layers": 1,
            "base_channels": 1,
        },
    }
    params_model_kwargs = {
        "net1_cls": "bsrnn",
        "net1_kw": {
            "input_channels": 1,
            "reference_channels": [0],
        },
        "net2_cls": "bsrnn",
        "net2_kw": {
            "input_channels": 3,
            "reference_channels": [0],
        },
    }
    n_params = 6_882_208

    def _update_kwargs_for_audiogram(self, kwargs, emb_dim):
        kwargs["audiogram"] = True
        kwargs["net1_kw"]["emb_dim"] = emb_dim
        kwargs["net2_kw"]["emb_dim"] = emb_dim
        return kwargs


class TestSGMSEp(_TestModel):
    model_name = "sgmsep"
    forward_args = [
        torch.randn(1, 2, 256, 32, dtype=torch.cfloat),
        torch.randn(1, 2, 256, 32, dtype=torch.cfloat),
        torch.randn(1, 2, 256, 32, dtype=torch.cfloat),
        torch.randn(1, 1, 1, 1),
        torch.randn(1, 1, 1, 1),
    ]
    latency = lambda self, x: 256  # noqa: E731
    default_model_kwargs = {
        "input_channels": 2,
        "reference_channels": [0, 1],
        "net_kw": {
            "in_channels": 8,
            "out_channels": 4,
            "aux_out_channels": 8,
            "base_channels": 4,
            "channel_mult": [1, 1, 1, 1],
            "num_blocks_per_res": 1,
            "noise_channel_mult": 1,
            "emb_channel_mult": 1,
            "fir_kernel": [1, 1],
            "causal": True,
        },
        "solver_kw": {"num_steps": 1},
        "wav_norm": None,
        "stft_kw": {
            "frame_length": 512,
            "hop_length": 128,
            "window": "mauler",
        },
        "stft_future_frames": 0,
    }
    params_model_kwargs = {}
    n_params = 65_590_694

    def _update_kwargs_for_audiogram(self, kwargs, emb_dim):
        kwargs["audiogram"] = True
        kwargs["net_kw"]["emb_dim"] = emb_dim
        return kwargs


class TestTCNDenseUNet(_TestModel):
    model_name = "tcndenseunet"
    forward_args = [torch.randn(1, 2, 17, 160, dtype=torch.cfloat)]
    latency = None  # TODO
    default_model_kwargs = {
        "input_channels": 2,
        "output_channels": 2,
        "hidden_channels": 1,
        "hidden_channels_dense": 1,
        "tcn_repeats": 1,
        "tcn_blocks": 1,
        "tcn_channels": 1,
        "stft_kw": {"frame_length": 32, "hop_length": 16},
    }
    params_model_kwargs = {}
    n_params = 7_706_542
    audiogram_support = False


class TestTFGridNet(_TestModel):
    model_name = "tfgridnet"
    forward_args = [torch.randn(1, 2, 9, 160, dtype=torch.cfloat)]
    latency = None  # TODO
    default_model_kwargs = {
        "input_channels": 2,
        "output_channels": 2,
        "layers": 1,
        "lstm_hidden_units": 1,
        "attn_heads": 1,
        "attn_approx_qk_dim": 1,
        "_emb_dim": 1,
        "_emb_ks": 1,
        "_emb_hs": 1,
        "stft_kw": {"frame_length": 16, "hop_length": 8},
    }
    params_model_kwargs = {}
    n_params = 3_829_712
    audiogram_support = False

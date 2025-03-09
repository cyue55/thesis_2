import importlib
import random

import numpy as np
import pytest
import torch

from mbchl.data.datasets import AudioDataset
from mbchl.has.base import BaseHA


def pytest_addoption(parser):
    parser.addoption("--amt_path", help="path to AMT")


@pytest.fixture(scope="session")
def matlab_engine(request):
    amt_path = request.config.getoption("--amt_path")
    if amt_path is None:
        pytest.skip("AMT path is not provided")
    matlab_engine = importlib.import_module("matlab.engine")
    eng = matlab_engine.start_matlab()
    eng.beep("off")
    eng.addpath(eng.genpath(amt_path))
    eng.amt_start(nargout=0)
    return eng


@pytest.fixture(scope="function")
def np_rng():
    return np.random.default_rng(seed=0)


@pytest.fixture(scope="function")
def torch_rng():
    g = torch.Generator()
    g.manual_seed(0)
    return g


@pytest.fixture(scope="session")
def dummy_dataset():
    class DummyDataset(AudioDataset):
        def __init__(
            self, examples, channels, min_length, max_length, fs, transform=None
        ):
            random_generator = random.Random(42)
            self.segments = [
                (i, (0, random_generator.randint(min_length, max_length)))
                for i in range(examples)
            ]
            torch_generator = torch.Generator().manual_seed(42)
            self.items = [
                [
                    torch.randn((ch, self.segments[i][1][1]), generator=torch_generator)
                    for i in range(examples)
                ]
                for ch in channels
            ]
            self.n_examples = examples
            self.transform = transform
            self.preloaded_data = None
            self._duration = sum(x[1][1] for x in self.segments) / fs
            self._effective_duration = self._duration
            self.segment_strategy = "pass"
            self.squeeze_signals = not len(channels) > 1
            self.dirs = list(range(len(channels)))
            self.segment_length = 0.0
            self._seed = 0
            self._epoch = 0

        def _load_segment(self, index, d, _return_metadata=False):
            signal = self.items[d][index]
            return signal

        def __len__(self):
            return self.n_examples

    return DummyDataset


@pytest.fixture(scope="session")
def dummy_model():
    class DummyModel(BaseHA):
        def __init__(self, input_channels=2, output_channels=1, use_transform=False):
            super().__init__()
            self.net = torch.nn.Conv1d(input_channels, output_channels, 1)
            self.input_channels = input_channels
            self.output_channels = output_channels
            self.use_transform = use_transform
            self.post_init(optimizer="Adam", optimizer_kw={"lr": 1e-3}, loss="mse")

        def forward(self, x, spk_adapt=None, audiogram=None):
            return self.net(x)

        def transform(self, x):
            # dummy pre-processing; trim inputs by factor 100
            if self.use_transform:
                is_tuple = isinstance(x, tuple)
                if not is_tuple:
                    x = (x,)
                x = tuple(x_[..., : self.transform_length(x_.shape[-1])] for x_ in x)
                if not is_tuple:
                    x = x[0]
            return x

        def transform_length(self, segment_length):
            return segment_length // 100 if self.use_transform else segment_length

    return DummyModel


@pytest.fixture(scope="session")
def sample_tensor():
    def _sample_tensor(x, n=10, seed=0):
        generator = torch.Generator()
        generator.manual_seed(seed)
        idx = torch.randint(x.numel(), (n,), generator=generator)
        return x.flatten()[idx]

    return _sample_tensor

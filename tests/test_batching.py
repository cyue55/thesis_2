import numpy as np
import pytest
import torch

from mbchl.data.batching import AudioBatchSampler, BatchSamplerRegistry
from mbchl.data.dataloader import AudioDataLoader

FS = 16000
CHANNELS = [1, 2, 3]
BATCH_SIZE = 4
EXAMPLES = 100
MIN_LENGTH = 1600
MAX_LENGTH = 80000


class BatchTester:
    def __init__(
        self, batch_sampler_name, batch_sampler, batch_size, dynamic, transform_length
    ):
        self.batch_sampler_name = batch_sampler_name
        self.batch_sampler = batch_sampler
        self.batch_size = batch_size
        self.dynamic = dynamic
        self.batch_sizes, self.pad_amounts = batch_sampler.calc_batch_stats(
            transform_length=transform_length
        )
        self.batch_size_error_count = 0
        if batch_sampler_name == "bucket":
            self.max_batch_size_error = batch_sampler.num_buckets
        else:
            self.max_batch_size_error = 1

    def reset_error_count(self):
        # need to reset error count between model inputs
        self.batch_size_error_count = 0

    def test(self, inputs, all_lengths):
        # convert to single item tuple if inputs is a tensor
        assert isinstance(all_lengths, torch.Tensor)
        if isinstance(inputs, torch.Tensor):
            inputs = (inputs,)
            assert all_lengths.ndim == 1
            all_lengths = all_lengths.reshape(-1, 1)
        else:
            assert isinstance(inputs, tuple)
            assert all_lengths.ndim == 2
        for x, lengths in zip(inputs, all_lengths.T):
            self.reset_error_count()
            if self.dynamic:
                assert x.shape[0] * x.shape[-1] <= self.batch_sampler.batch_size
            else:
                # batch size can be different only when it's the last one
                # except for bucketing where it can happen for each bucket
                try:
                    assert x.shape[0] == self.batch_size
                except AssertionError:
                    if self.batch_size_error_count < self.max_batch_size_error:
                        self.batch_size_error_count += 1
                    else:
                        raise
            assert x.shape[-1] == max(lengths)
            assert all((y[..., k:] == 0).all() for y, k in zip(x, lengths))
            assert all((y[..., k - 1] != 0).all() for y, k in zip(x, lengths))
            if self.batch_sampler_name == "sorted":
                assert _is_sorted(lengths)
            elif self.batch_sampler_name == "bucket":
                i = np.searchsorted(self.batch_sampler.right_bucket_limits, x.shape[-1])
                left = 0 if i == 0 else self.batch_sampler.right_bucket_limits[i - 1]
                right = self.batch_sampler.right_bucket_limits[i]
                assert all(left <= k <= right for k in lengths)
            elif self.batch_sampler_name != "random":
                raise ValueError(
                    f"unexpected batch_sampler_name: {self.batch_sampler_name}"
                )
        x = inputs[0]
        batch_size = x.shape[0] * x.shape[-1]
        pad_amount = sum(x.shape[-1] - k for k in all_lengths[:, 0]).item()
        self.batch_sizes.remove(batch_size)
        self.pad_amounts.remove(pad_amount)


def _init_dataloader(
    batch_sampler_name,
    dataset,
    batch_size,
    dynamic,
    shuffle,
):
    batch_sampler_cls = BatchSamplerRegistry.get(batch_sampler_name)
    batch_sampler = batch_sampler_cls(
        dataset=dataset,
        # batch_size=(BATCH_SIZE*MAX_LENGTH)/FS if dynamic else BATCH_SIZE,
        batch_size=batch_size,
        dynamic=dynamic,
        shuffle=shuffle,
    )
    dataloader = AudioDataLoader(
        dataset=dataset,
        batch_sampler=batch_sampler,
    )
    return batch_sampler, dataloader


def _is_sorted(seq):
    return all(seq[i] <= seq[i + 1] for i in range(len(seq) - 1))


def _test_batch_sampler_on_dset(
    batch_sampler_name, dataset, batch_size, dynamic, transform_length
):
    batch_sampler, dataloader = _init_dataloader(
        batch_sampler_name, dataset, batch_size, dynamic, shuffle=True
    )
    n = len(batch_sampler)
    # test every batch
    batch_tester = BatchTester(
        batch_sampler_name, batch_sampler, batch_size, dynamic, transform_length
    )
    i = 0
    for inputs, lengths in dataloader:
        batch_tester.test(inputs, lengths)
        i += 1
    assert n == i
    # test error when attempting to iterate again
    with pytest.raises(ValueError):
        for inputs, lengths in dataloader:
            break
    # test error fix by setting epoch
    dataloader.set_epoch(1)
    for inputs, lengths in dataloader:
        break
    # check shuffle
    dataloader.set_epoch(2)
    for inputs, lengths in dataloader:
        break
    dataloader.set_epoch(3)
    for (inputs_), lengths_ in dataloader:
        for x, x_ in zip(inputs, inputs_):
            if x.shape == x_.shape:
                assert (x != x_).any()
        if lengths.shape == lengths_.shape:
            assert (lengths != lengths_).any()
        break
    # check no shuffle
    batch_sampler, dataloader = _init_dataloader(
        batch_sampler_name, dataset, batch_size, dynamic, shuffle=False
    )
    for inputs, lengths in dataloader:
        break
    dataloader.set_epoch(1)
    for (inputs_), lengths_ in dataloader:
        for x, x_ in zip(inputs, inputs_):
            assert (x == x_).all()
        assert (lengths == lengths_).all()
        break


@pytest.mark.parametrize("batch_sampler_name", BatchSamplerRegistry.keys())
@pytest.mark.parametrize("dynamic", [False, True])
@pytest.mark.parametrize(
    "use_model, use_transform",
    [
        [False, False],
        [True, False],
        [True, True],
    ],
)
def test_batch_sampler(
    dummy_dataset,
    dummy_model,
    batch_sampler_name,
    dynamic,
    use_model,
    use_transform,
):
    if batch_sampler_name == "default":
        return
    if use_model:
        model = dummy_model(use_transform=use_transform)
        transform = model.transform
        transform_length = model.transform_length
    else:
        transform = None
        transform_length = None
    dataset = dummy_dataset(
        examples=EXAMPLES,
        channels=CHANNELS,
        min_length=MIN_LENGTH,
        max_length=MAX_LENGTH,
        fs=FS,
        transform=transform,
    )
    batch_size = (BATCH_SIZE * MAX_LENGTH) / FS if dynamic else BATCH_SIZE
    _test_batch_sampler_on_dset(
        batch_sampler_name,
        dataset,
        batch_size,
        dynamic,
        transform_length,
    )
    # test on a torch.utils.data.Subset
    train_split, _ = torch.utils.data.random_split(
        dataset,
        [EXAMPLES // 2, EXAMPLES // 2],
        torch.Generator().manual_seed(0),
    )
    _test_batch_sampler_on_dset(
        batch_sampler_name,
        train_split,
        batch_size,
        dynamic,
        transform_length,
    )


def test_errors(dummy_dataset):
    dataset = dummy_dataset(
        examples=EXAMPLES,
        channels=CHANNELS,
        min_length=MIN_LENGTH,
        max_length=MAX_LENGTH,
        fs=FS,
    )
    batch_sampler = AudioBatchSampler(dataset, BATCH_SIZE)
    with pytest.raises(NotImplementedError):
        next(iter(batch_sampler))
    with pytest.raises(NotImplementedError):
        len(batch_sampler)

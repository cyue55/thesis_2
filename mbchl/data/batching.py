import random
import warnings

import numpy as np
import torch

from ..utils import Registry

BatchSamplerRegistry = Registry("batch_sampler")


class AudioBatchSampler(torch.utils.data.Sampler):
    """Base class for all samplers.

    Integrates with AudioDataset to make batches of segments of variable lengths.

    Subclasses should implement the :meth:`_generate_batches` method which takes as
    input the dataset indices and returns a list of batches where each batch is a list
    of ``(segment_idx, segment_length)`` tuples. The :meth:`__init__` method should also
    be overwritten in case the sampler requires extra arguments.

    Also implements a :meth:`set_epoch` method to shuffle batches with the correct seed
    when resuming training. The :meth:`set_epoch` method must be called before iterating
    over the batch sampler, unless ``shuffle`` is ``False``.

    Parameters
    ----------
    dataset : AudioDataset
        AudioDataset instance.
    batch_size : int or float
        Batch size. If ``dynamic`` is ``False``, it is defined as a number of segments
        in each batch (fixed batch size). If ``dynamic`` is ``True``, it is a total
        length of segments in seconds (dynamic batch size).
    drop_last : bool, optional
        Whether to drop the last segments in the dataset if they don't form a full
        batch.
    shuffle : bool, optional
        Whether to shuffle the batches before each epoch.
    seed : int, optional
        Random seed for shuffling.
    dynamic : bool, optional
        Whether ``batch_size`` is defined as a number of segments in each batch (fixed
        batch size) instead of a total length of segments (dynamic batch size) in
        seconds.
    sort : bool, optional
        Whether to sort the segments by length before generating the batches. If
        ``shuffle`` is ``True``, segments are sorted but segments of equal length are
        shuffled.
    fs : int, optional
        Sampling rate. Used when ``dynamic`` is ``True`` to convert the batch size in
        seconds to a number of samples. Ignored if ``dynamic`` is ``False``.
    reverse : bool, optional
        Whether to reverse the order of the batches. Ignored if ``sort`` is ``False``.

    """

    def __init__(
        self,
        dataset,
        batch_size,
        drop_last=False,
        shuffle=True,
        seed=0,
        dynamic=False,
        sort=False,
        fs=16000,
        reverse=False,
    ):
        self.dataset = dataset
        if dynamic:
            self.batch_size = round(fs * batch_size)
        else:
            if isinstance(batch_size, float):
                warnings.warn(
                    "Got float batch_size even though dynamic is "
                    "False. Casting batch_size to int."
                )
            self.batch_size = int(batch_size)
        self.drop_last = drop_last
        self.shuffle = shuffle
        self.dynamic = dynamic
        self.sort = sort
        self.reverse = reverse
        self._seed = random.Random(seed).randrange(2**32)
        self._epoch = 0
        self._previous_epoch = -1
        self._segment_lengths = None
        self._batches = None

    def __iter__(self):
        """Iterate over the batches."""
        if self.shuffle:
            if self._epoch == self._previous_epoch:
                raise ValueError(
                    "the set_epoch method must be called before iterating "
                    "over the dataloader in order to regenerate the batches "
                    "with the correct seed"
                )
            self.generate_batches()
            self._shuffle_batches()
            self._previous_epoch = self._epoch
        elif self._batches is None:
            self.generate_batches()
        for batch in self._batches:
            yield [idx for idx, length in batch]

    def generate_batches(self):
        """Generate the batches. Called at the beginning of each epoch."""
        indices = self._generate_indices()
        self._batches = self._generate_batches(indices)

    def _generate_indices(self):
        self._get_segment_lengths()
        if self.sort:
            if self.shuffle:
                # sort by length but randomize segments of same length
                randomizer = random.Random(self._seed + self._epoch)
                lengths = sorted(
                    self._segment_lengths,
                    key=lambda x: (x[1], randomizer.random()),
                    reverse=self.reverse,
                )
            else:
                lengths = sorted(
                    self._segment_lengths, key=lambda x: x[1], reverse=self.reverse
                )
            indices = [idx for idx, length in lengths]
        else:
            indices = list(range(len(self._segment_lengths)))
            if self.shuffle:
                randomizer = random.Random(self._seed + self._epoch)
                randomizer.shuffle(indices)
        return indices

    def _get_segment_lengths(self):
        if isinstance(self.dataset, torch.utils.data.Subset):
            dataset = self.dataset.dataset
            indices = self.dataset.indices
        else:
            dataset = self.dataset
            indices = range(len(dataset))
        if self._segment_lengths is None:
            self._segment_lengths = [
                (i, dataset.get_segment_length(j)) for i, j in enumerate(indices)
            ]
        if self.dynamic and any(x[1] > self.batch_size for x in self._segment_lengths):
            raise ValueError(
                "Found a segment that is longer than the dynamic batch size. "
                "Consider increasing the dynamic batch size or reducing the maximum "
                "segment length in the dataset."
            )

    def _generate_batches(self, indices):
        raise NotImplementedError

    def set_epoch(self, epoch):
        """Set the internal epoch counter.

        Useful for reproducibility when training is resumed.
        """
        self._epoch = epoch

    def _shuffle_batches(self):
        randomizer = random.Random(self._seed + self._epoch)
        randomizer.shuffle(self._batches)

    def __len__(self):
        """Return the number of batches."""
        if self._batches is None:
            self.generate_batches()
        return len(self._batches)

    def calc_batch_stats(self, transform_length=None):
        """Get batch sizes and padding amounts without iterating over the batches."""
        if transform_length is None:
            transform_length = lambda x: x  # noqa: E731
        batch_sizes = []
        pad_amounts = []
        for batch in self._batches:
            batch_lengths = [transform_length(length) for idx, length in batch]
            max_length = max(batch_lengths)
            batch_sizes.append(len(batch) * max_length)
            pad_amounts.append(sum(max_length - length for length in batch_lengths))
        return batch_sizes, pad_amounts


class _BaseRandSortBatchSampler(AudioBatchSampler):
    """Base class for the random and sorted batch samplers."""

    def _generate_batches(self, indices):
        batches = []
        batch = []
        for i in indices:
            segment_idx, segment_length = self._segment_lengths[i]
            if self._new_batch(batch, segment_length):
                batches.append(batch)
                batch = []
                batch.append((segment_idx, segment_length))
            else:
                batch.append((segment_idx, segment_length))
        if len(batch) > 0 and not self.drop_last:
            batches.append(batch)
        return batches

    def _new_batch(self, batch, segment_length):
        output = False
        if self.dynamic:
            if segment_length > self.batch_size:
                raise ValueError(
                    "got a segment that is longer than the dynamic batch size"
                )
            batch_length = max(x[1] for x in batch) if batch else 0
            if (len(batch) + 1) * max(segment_length, batch_length) > self.batch_size:
                output = True
        elif len(batch) + 1 > self.batch_size:
            output = True
        return output


@BatchSamplerRegistry.register("random")
class RandomBatchSampler(_BaseRandSortBatchSampler):
    """Random batching."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, sort=False, **kwargs)


@BatchSamplerRegistry.register("sorted")
class SortedBatchSampler(_BaseRandSortBatchSampler):
    """Sorted batching."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, sort=True, **kwargs)


@BatchSamplerRegistry.register("bucket")
class BucketBatchSampler(AudioBatchSampler):
    """Bucket batching.

    Segments are grouped into different buckets according to their length. Batches are
    formed with segments from the same bucket. This reduces the amount of zero-padding
    while keeping some randomness.

    Parameters
    ----------
    *args : list
        Passed to parent class constructor.
    num_buckets : int, optional
        The number of buckets. This defines a compromise between padding and
        randomization; the more buckets, the less the padding, but also the less
        randomization. Bucket limits are uniformly spaced between the minimum and
        maximum segment length in the dataset.
    **kwargs : dict
        Passed to parent class constructor.

    """

    def __init__(self, *args, num_buckets=10, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_buckets = num_buckets

    def _generate_batches(self, indices):
        max_length = max(x[1] for x in self._segment_lengths)

        right_bucket_limits = np.linspace(
            max_length / self.num_buckets,
            max_length,
            self.num_buckets,
        )
        self.right_bucket_limits = right_bucket_limits  # for unit testing

        if self.dynamic:
            bucket_sizes = self.batch_size // right_bucket_limits
        else:
            bucket_sizes = [self.batch_size] * self.num_buckets

        batches = []
        buckets = [[] for _ in range(self.num_buckets)]
        for i in indices:
            segment_idx, segment_length = self._segment_lengths[i]
            bucket_idx = np.searchsorted(right_bucket_limits, segment_length)
            if not 0 <= bucket_idx < self.num_buckets:
                raise ValueError(
                    "attempted to assign a segment to a non-existent bucket"
                )
            buckets[bucket_idx].append((segment_idx, segment_length))
            if len(buckets[bucket_idx]) == bucket_sizes[bucket_idx]:
                batches.append(buckets[bucket_idx])
                buckets[bucket_idx] = []
            elif len(buckets[bucket_idx]) > bucket_sizes[bucket_idx]:
                raise ValueError(
                    "maximum number of segments allowed in bucket exceeded"
                )

        if not self.drop_last:
            for bucket_idx, batch in enumerate(buckets):
                if len(batch) > 0:
                    batches.append(batch)

        return batches


@BatchSamplerRegistry.register("default")
class TorchDefaultBatchSampler(torch.utils.data.BatchSampler):
    """Default PyTorch batching.

    If ``shuffle`` is ``True``, uses :class:`torch.utils.data.RandomSampler`. Else, uses
    :class:`torch.utils.data.SequentialSampler`.
    """

    def __init__(self, dataset, batch_size, shuffle=True, drop_last=False):
        if shuffle:
            sampler = torch.utils.data.RandomSampler(dataset)
        else:
            sampler = torch.utils.data.SequentialSampler(dataset)
        super().__init__(sampler, batch_size, drop_last)


class DistributedBatchSamplerWrapper(torch.utils.data.DistributedSampler):
    """Distributed batch sampler wrapper."""

    def __init__(self, sampler, *args, **kwargs):
        super().__init__(dataset=sampler, *args, **kwargs)
        self.sampler = sampler

    def __iter__(self):
        """Iterate over the batches."""
        for dist_index in super().__iter__():
            yield [i for i, length in self.sampler._batches[dist_index]]

    def set_epoch(self, epoch):
        """Set the internal epoch counter.

        Useful for reproducibility when training is resumed.
        """
        super().set_epoch(epoch)
        self.sampler.set_epoch(epoch)

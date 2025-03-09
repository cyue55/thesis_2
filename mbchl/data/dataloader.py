import math
import random
import warnings

import torch
import torch.nn.functional as F

from .batching import AudioBatchSampler, TorchDefaultBatchSampler


class AudioDataLoader(torch.utils.data.DataLoader):
    """Audio file dataloader class.

    - Implements the collating function to form batches of variable size tensors.
    - Supports buffering of batches for shuffling outputs from dataset instances of
      :class:`torch.utils.data.IterableDataset`.
    - Supports worker state saving and loading for dataset instances of
      :class:`torch.utils.data.IterableDataset`.

    When buffering, the memory sharing strategy is set to ``"file_system"`` to avoid
    a ``RuntimeError: Too many open files``. See `here <https://github.com/pytorch/pytorch/issues/11201#issuecomment-421146936>`__.
    Note that this is prone to memory leaks. More information about the different
    sharing strategies is available `here <https://pytorch.org/docs/main/multiprocessing.html#sharing-strategies>`__.

    To save worker states, the dataset outputs must include single-element dictionaries
    of the form ``{worker_id: state}`` where ``state`` is a dictionary containing
    arbitrary information about the worker state. The state of each worker is tracked
    by the dataloader such that it can be saved and loaded when training is resumed. The
    dataset must implement :meth:`load_state_dict` to load the worker state. The worker
    state is not yielded by the dataloader.

    Currently, if training is resumed, the training examples that were in the buffer
    before interruption are not re-loaded.

    Parameters
    ----------
    *args
        Passed to :class:`torch.utils.data.DataLoader`.
    buffer_size : int, optional
        Number of batches to buffer for shuffling. If ``None``, no buffering is
        performed.
    batch_mix : bool, optional
        If ``True``, batches in the buffer are concatenated, shuffled and split again
        before yielding. Output batches can thus be made of items that were initially in
        different batches before pushing to the buffer. This ensures that batches can
        contain items from different workers, which can be useful for dataset instances
        of :class:`torch.utils.data.IterableDataset` that read from different streams
        (e.g. see `here <https://stackoverflow.com/questions/57729279/how-to-ensure-that-a-batch-contains-samples-from-all-workers-with-pytorchs-data>`__).
        Requires that all items have the same dimensions (i.e. variable length
        items are not supported and segmentation should be used). If ``False``, the
        batches in the buffer are shuffled and yielded as is. Ignored if ``buffer_size``
        is ``None``.
    seed : int, optional
        Random seed for shuffling the buffer.
    **kwargs
        Passed to :class:`torch.utils.data.DataLoader`.

    """

    def __init__(self, dataset, buffer_size=None, batch_mix=False, seed=None, **kwargs):
        num_workers = kwargs.get("num_workers", 0)
        persistent_workers = kwargs.get("persistent_workers", False)
        if (
            num_workers > 0
            and not persistent_workers
            and isinstance(dataset, torch.utils.data.IterableDataset)
        ):
            raise ValueError(
                f"Set persistent_workers=True for {dataset.__class__.__name__}"
            )

        if persistent_workers and num_workers == 0:
            warnings.warn(
                "Got persistent_workers=True but num_workers is 0. "
                "Setting persistent_workers=False."
            )
            kwargs["persistent_workers"] = False

        batch_sampler = kwargs.get("batch_sampler")
        if batch_sampler is not None and isinstance(
            dataset, torch.utils.data.IterableDataset
        ):
            # for IterableDataset, DataLoader raises a ValueError when batch_sampler is
            # specified, but we want to allow it if the batch sampler is an instance of
            # TorchDefaultBatchSampler with SequentialSampler
            if not isinstance(batch_sampler, TorchDefaultBatchSampler):
                raise ValueError(
                    "AudioDataLoader with IterableDataset: specified batch sampler "
                    "must be an instance of TorchDefaultBatchSampler, "
                    f"got {batch_sampler.__class__.__name__}."
                )
            if isinstance(batch_sampler.sampler, torch.utils.data.SequentialSampler):
                # get the batch size and drop last attributes from the batch sampler to
                # pass them directly to the DataLoader and remove batch_sampler from
                # the keyword arguments to prevent the ValueError
                kwargs["batch_size"] = batch_sampler.batch_size
                kwargs["drop_last"] = batch_sampler.drop_last
                kwargs.pop("batch_sampler")
            else:
                raise ValueError(
                    "AudioDataLoader with IterableDataset: specified batch sampler "
                    "must have a SequentialSampler sampler, "
                    f"got {batch_sampler.sampler.__class__.__name__}."
                )

        self.buffer_size = buffer_size
        self.batch_mix = batch_mix
        self.seed = seed
        self._buffer = None if buffer_size is None else []
        if buffer_size is not None:
            torch.multiprocessing.set_sharing_strategy("file_system")
        self._worker_states = []
        self._random = self._get_random(epoch=0)

        super().__init__(dataset, **kwargs, collate_fn=self.collate_fn)

    def __iter__(self):
        """Iterate over the dataloader."""
        for batch, lengths, worker_state in super().__iter__():
            self._update_worker_state(worker_state)
            if self.buffer_size is None:
                yield batch, lengths
            else:
                self._buffer.append((batch, lengths))
                if len(self._buffer) >= self.buffer_size:
                    if self.batch_mix:
                        # infer batch size
                        batch_size = self.batch_sampler.batch_size
                        # concatenate batches
                        batches, lengths = zip(*self._buffer)
                        batches = zip(*batches)
                        batches = [torch.cat(b, dim=0) for b in batches]
                        lengths = torch.cat(lengths, dim=0)
                        # shuffle
                        n = lengths.shape[0]
                        indices = list(range(n))
                        self._random.shuffle(indices)
                        batches = [b[indices] for b in batches]
                        lengths = lengths[indices]
                        # split
                        for i in range(0, n - batch_size + 1, batch_size):
                            yield (
                                tuple(b[i : i + batch_size] for b in batches),
                                lengths[i : i + batch_size],
                            )
                        self._buffer.clear()
                        # push remaining items to buffer
                        i += batch_size
                        if i < n:
                            self._buffer.append(
                                (tuple(b[i:] for b in batches), lengths[i:])
                            )
                    else:
                        self._random.shuffle(self._buffer)
                        yield from self._buffer
                        self._buffer.clear()

    def _update_worker_state(self, worker_state):
        # if the dataset outputs tensors only, this does nothing
        for state_list in worker_state:
            for i, state in enumerate(state_list):
                if not isinstance(state, dict):
                    raise ValueError(
                        "Worker state must be a dictionary, "
                        f"got {state.__class__.__name__}."
                    )
                if len(self._worker_states) <= i:
                    self._worker_states.append({})
                self._worker_states[i].update(state)

    def state_dict(self):
        """Return the state dictionary."""
        # output can be empty if the dataset was not iterated over yet
        # for example a validation dataset that is not used until after some epochs
        return self._worker_states

    def load_state_dict(self, state_dict):
        """Load the state dictionary."""
        self._worker_states = state_dict
        if hasattr(self.dataset, "load_state_dict"):
            self.dataset.load_state_dict(state_dict)
        elif state_dict:
            raise ValueError(
                "Found worker states in state_dict, but "
                f"{self.dataset.__class__.__name__} has no load_state_dict method."
            )

    def set_epoch(self, epoch):
        """Set the internal epoch counter of the associated batch sampler and dataset.

        Useful for reproducibility when training is resumed.
        """
        if isinstance(self.batch_sampler, AudioBatchSampler):
            self.batch_sampler.set_epoch(epoch)
        if isinstance(self.dataset, torch.utils.data.Subset):
            dataset = self.dataset.dataset
        else:
            dataset = self.dataset
        if hasattr(dataset, "set_epoch"):
            dataset.set_epoch(epoch)
        self._random = self._get_random(epoch)

    def _get_random(self, epoch):
        # return a generator seeded from the seed attribute and the epoch
        seed_1 = random.Random(self.seed).randrange(2**32)
        seed_2 = random.Random(epoch).randrange(2**32)
        seed = seed_1 ^ seed_2
        return random.Random(seed)

    @staticmethod
    def collate_fn(unbatched, _tensors_only=False):
        """Collate variable size tensors.

        Variable size tensors are zero-padded to match the length of the longest example
        in the batch along the last dimension. Supports an arbitrary number of dataset
        outputs.

        Parameters
        ----------
        unbatched : list[torch.Tensor] or list[tuple[torch.Tensor|Any, ...]]
            Unbatched tensors. The length of the list is the batch size, while the list
            items are tensors or tuples depending on the number of outputs from the
            dataset. Tensors can have a variable size along the last dimension, in which
            case they are zero-padded to match the length of the longest example in the
            batch.

        Returns
        -------
        batched : torch.Tensor or tuple[torch.Tensor, ...]
            Batched tensors. Tensor or tuple of tensors with shape ``(batch_size, ...)``
            depending on the number of outputs from the dataset.
        lengths : torch.Tensor
            Original tensor lengths along the last dimension. Useful to ensure losses
            are not aggregated over zero-padded regions. Shape ``(batch_size,)`` or
            ``(batch_size, n_tensors)`` depending on the number of outputs from the
            dataset.
        worker_state : list[Any]
            Non-tensor outputs from the dataset.

        Example
        -------
        >>> unbatched = [[torch.rand(2, 5), torch.rand(1)],
        ...              [torch.rand(2, 3), torch.rand(1)],
        ...              [torch.rand(2, 4), torch.rand(1)]]
        >>> unbatched
        [
            [
                tensor([[0.37, 0.09, 0.51, 0.41, 0.03],
                        [0.21, 0.25, 0.26, 0.65, 0.38]]),
                tensor([0.77])
            ],
            [
                tensor([[0.99, 0.13, 0.02],
                        [0.01, 0.84, 0.48]]),
                tensor([0.14])
            ],
            [
                tensor([[0.31, 0.10, 0.31, 0.57],
                        [0.29, 0.71, 0.19, 0.34]]),
                tensor([0.42])
            ]
        ]
        >>> batched, lengths = _collate_fn(unbatched)
        >>> [x.shape for x in batched]
        [torch.Size([3, 2, 5]), torch.Size([3, 1])]
        >>> batched
        (
            tensor([
                [
                    [0.37, 0.09, 0.51, 0.41, 0.03],
                    [0.21, 0.25, 0.26, 0.65, 0.38]
                ],
                [
                    [0.99, 0.13, 0.02, 0.00, 0.00],
                    [0.01, 0.84, 0.48, 0.00, 0.00]
                ],
                [
                    [0.31, 0.10, 0.31, 0.57, 0.00],
                    [0.29, 0.71, 0.19, 0.34, 0.00]
                ]
            ]),
            tensor([[0.77],
                    [0.14],
                    [0.42]])
        )
        >>> lengths
        tensor([[5, 1],
                [3, 1],
                [4, 1]])

        """
        # convert to tuple if batch items are tensors
        inputs_are_tensors = isinstance(unbatched[0], torch.Tensor)
        unbatched = [(x,) if inputs_are_tensors else x for x in unbatched]
        lengths = torch.tensor(
            [
                [x.shape[-1] for x in inputs if isinstance(x, torch.Tensor)]
                for inputs in unbatched
            ],
            device=unbatched[0][0].device,
        )
        batched = tuple(
            torch.stack(
                [
                    F.pad(x, (0, max_length - x.shape[-1]))
                    for x in inputs
                    if isinstance(x, torch.Tensor)
                ]
            )
            for inputs, max_length in zip(zip(*unbatched), lengths.amax(dim=0))
        )
        # convert back to tensor if batch items were tensors
        if inputs_are_tensors:
            batched = batched[0]
            lengths = lengths.squeeze(-1)
        if _tensors_only:
            return batched, lengths
        # deal with non-tensor inputs
        worker_state = [
            [x for x in inputs if not isinstance(x, torch.Tensor)]
            for inputs in unbatched
        ]
        return batched, lengths, worker_state

    def __len__(self):
        """Compute the number of batches in the dataloader.

        This fixes the default calculation for instances of
        :class:`torch.utils.data.IterableDataset` when ``num_workers > 0`` which ignores
        the fact that each worker can drop the last incomplete batch. See
        `here <https://github.com/pytorch/pytorch/issues/33413>`__.
        """
        if (
            isinstance(self.dataset, torch.utils.data.IterableDataset)
            and self.num_workers > 0
        ):
            dataset_len = len(self.dataset)
            worker_len = [
                dataset_len // self.num_workers + (i < dataset_len % self.num_workers)
                for i in range(self.num_workers)
            ]
            n_batch = [
                (
                    wl // self.batch_size
                    if self.drop_last
                    else math.ceil(wl / self.batch_size)
                )
                for wl in worker_len
            ]
            output = sum(n_batch)
        else:
            output = super().__len__()
        return output

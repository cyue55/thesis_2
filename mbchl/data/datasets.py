import io
import logging
import os
import random
import re
import sys
import tarfile
import warnings

import numpy as np
import scipy.signal
import soundfile as sf
import soxr
import torch
import torch.nn.functional as F
from tqdm import tqdm

from ..filesystem import init_filesystem
from ..utils import (
    NoiseTooSmallError,
    Registry,
    find_files,
    random_audiogram,
    set_dbfs,
    set_snr,
)
from .framrir import fram_rir

DatasetRegistry = Registry("dataset")


@DatasetRegistry.register("default")
class AudioDataset(torch.utils.data.Dataset):
    """Audio file dataset class.

    A map-style dataset which reads audio files from directories or tar archives and
    optionally segments them using different strategies.

    The duration of all audio files is loaded and saved at initialization, such that
    regenerating the segments at the start of a new epoch does not require reading the
    files again.

    The :meth:`get_segment_length` method returns the length of the i-th segment without
    having to load it, which is useful for forming batches based on the length of the
    segments.

    Parameters
    ----------
    dirs : str or list[str]
        Directories containing the audio files. If ``str``, a single directory is
        scanned and outputs from :meth:`__getitem__` have shape ``(n_channels,
        n_samples)`` if the files are multi-channel or ``(n_samples,)`` if the files are
        mono and ``squeeze_channels`` is ``True``. If ``list`` of ``str`` with length
        ``n_signals``, each directory must have the same number of files with the same
        names, and outputs from :meth:`__getitem__` are tuples of ``n_signals`` tensors
        with shape ``(n_channels, n_samples)`` or ``(n_samples,)``.
    segment_length : float, optional
        Segment length in seconds to split the audio files into. If ``None``, the files
        are not segmented.
    overlap_length : float, optional
        Segment overlap in seconds. Ignored if ``segment_length`` is ``None``.
    fs : int, optional
        Sampling rate.
    segment_strategy : {"drop", "pass", "pad", "overlap", "random"}, optional
        Segmentation strategy. Ignored if ``segment_length`` is ``None``.

        - ``"drop"``: Trailing segments are discarded.
        - ``"pass"``: Trailing segments are included as is.
        - ``"pad"``: Trailing segments are zero-padded.
        - ``"overlap"``: Trailing segments overlap with the second-to-last segment.
        - ``"random"``: A segment is sampled at a random start index in each file. If
          the file is shorter than ``segment_length``, the file is padded. To generate
          new segments at the start of a new epoch, make sure to call the
          :meth:`set_epoch` method. Incompatible with preloading.
    transform : callable, optional
        Pre-processing function applied to each segment at the end of
        :meth:`__getitem__`. This is performed by the workers on CPU before batching.
        Takes as input a tensor or a tuple of tensors with shape ``(n_channels,
        n_samples)`` and returns a tensor or tuple of tensors with arbitrary shapes.
        The last dimension of each output tensor should correspond to time (e.g. STFT
        frames), such that it can be padded before batching. If ``None``, no
        pre-processing is applied.
    ext : tuple, optional
        Audio file extensions to consider when scanning input directories.
    regexp : str, optional
        Additional regular expression to filter the audio files.
    dtype : str, optional
        Data type to cast the audio files to.
    squeeze_channels : bool, optional
        Whether to squeeze the channel dimension if the audio files are mono.
    seed : int, optional
        Seed for the random segment strategy. Ignored if ``segment_strategy`` is not
        ``"random"`` and ``n_files`` is ``"all"``.
    n_files : int or "all", optional
        Number of files to load from the directories. If ``"all"``, all files are
        loaded.

    """

    def __init__(
        self,
        dirs,
        segment_length=None,
        overlap_length=0.0,
        fs=16000,
        segment_strategy="pass",
        transform=None,
        ext=(".wav", ".flac", ".mp3"),
        regexp=None,
        dtype="float32",
        squeeze_channels=False,
        seed=0,
        n_files="all",
    ):
        dirs_is_list = isinstance(dirs, list)
        if not dirs_is_list:
            dirs = [dirs]
        self.dirs = [TarArchive(d) if d.endswith(".tar") else d for d in dirs]
        segment_length = None if segment_length is None else round(segment_length * fs)
        self.segment_length = segment_length
        self.overlap_length = round(overlap_length * fs)
        self.fs = fs
        self.segment_strategy = segment_strategy
        self.transform = transform
        self.ext = ext
        self.regexp = regexp
        self.dtype = dtype
        self.squeeze_channels = squeeze_channels
        self.squeeze_signals = not dirs_is_list
        self.files = self._get_files(n_files, seed)
        self.lengths = self._get_lengths()
        self._seed = random.Random(seed).randrange(2**32)
        self._epoch = 0
        self.segments = self._get_segments()
        self.preloaded_data = None

    def _get_files(self, n_files, seed):
        for i, d in enumerate(self.dirs):
            if isinstance(d, TarArchive):
                files = filter(
                    lambda x: x.endswith(self.ext)
                    and (self.regexp is None or re.match(self.regexp, x)),
                    d.members,
                )
            else:
                if not os.path.exists(d) or not os.path.isdir(d):
                    raise FileNotFoundError(f"directory does not exist: {d}")
                files = [
                    os.path.relpath(os.path.join(root, file), d)
                    for root, _, files in os.walk(d)
                    for file in filter(
                        lambda x: x.endswith(self.ext)
                        and (self.regexp is None or re.match(self.regexp, x)),
                        files,
                    )
                ]
            if i == 0:
                # convert to set for efficient comparison
                output = set(files)
            elif set(files) != output:
                raise ValueError(
                    "directories must have the same number of files with the same names"
                )
        # sort to ensure same file order across executions and systems
        output = sorted(output)
        if isinstance(n_files, int):
            output = random.Random(seed).sample(output, k=n_files)
        elif n_files != "all":
            raise TypeError(f"n_files must be int or 'all', got {n_files}")
        return output

    def _get_segments(self):
        generator = random.Random(self._seed + self._epoch)
        output = []
        for i_file, length in enumerate(self.lengths):
            if self.segment_length is None:
                output.append((i_file, (0, length)))
            elif self.segment_strategy == "random":
                i_max = max(length - self.segment_length, 0)
                i_start = generator.randint(0, i_max)
                i_end = i_start + self.segment_length
                output.append((i_file, (i_start, i_end)))
            else:
                self._add_segments(output, i_file, length)
        self._effective_duration = sum(j - i for _, (i, j) in output) / self.fs
        return output

    def _get_lengths(self):
        output = []
        for file in self.files:
            for i, d in enumerate(self.dirs):
                file_obj = self._open_file(d, file)
                info = sf.info(file_obj)
                file_obj.close()
                if i == 0:
                    frames = info.frames
                    channels = info.channels
                    samplerate = info.samplerate
                elif (
                    info.frames != frames
                    or info.channels != channels
                    or info.samplerate != samplerate
                ):
                    raise ValueError(
                        f"file {file} must have the same length, number of channels and"
                        "sampling frequency in all directories"
                    )
            # /!\ stored length should be AFTER resampling /!\
            frames = round(frames * self.fs / samplerate)
            output.append(frames)
        self._duration = sum(output) / self.fs
        return output

    def _open_file(self, d, file):
        if isinstance(d, TarArchive):
            return d._get_file(file)
        else:
            return open(os.path.join(d, file), "rb")

    def _add_segments(self, segments, i_file, length):
        hop_length = self.segment_length - self.overlap_length
        n_segments = (length - self.segment_length) // hop_length + 1
        for i_segment in range(n_segments):
            i_start = i_segment * hop_length
            i_end = i_start + self.segment_length
            segments.append((i_file, (i_start, i_end)))
        if n_segments <= 0 or i_end != length:
            if self.segment_strategy == "pass":
                i_start = n_segments * hop_length
                segments.append((i_file, (i_start, length)))
            elif self.segment_strategy == "pad":
                i_start = n_segments * hop_length
                i_end = i_start + self.segment_length
                segments.append((i_file, (i_start, i_end)))
            elif self.segment_strategy == "overlap":
                i_start = length - self.segment_length
                if i_start >= 0:
                    segments.append((i_file, (i_start, length)))
            elif self.segment_strategy != "drop":
                raise ValueError(
                    f"invalid segment strategy, got {self.segment_strategy}"
                )

    def __getitem__(self, index):
        """Return the i-th segment of the dataset."""
        return self.get_item(index)

    def get_item(self, index, _return_metadata=False):
        """Return the i-th segment of the dataset.

        If ``_return_metadata`` is ``True``, also returns a dictionary with metadata
        about the loaded segment.
        """
        if self.preloaded_data is not None:
            signals = self.preloaded_data[index]
        else:
            if self.squeeze_signals:
                signals = self._load_segment(index, self.dirs[0], _return_metadata)
                if _return_metadata:
                    signals, metadata = signals
            else:
                signals = tuple(
                    self._load_segment(index, d, _return_metadata) for d in self.dirs
                )
                if _return_metadata:
                    signals, metadata = zip(*signals)
            if self.transform is not None:
                signals = self.transform(signals)
        if _return_metadata:
            return signals, metadata
        else:
            return signals

    def _load_segment(self, index, d, _return_metadata=False):
        i_file, (i_start, i_end) = self.segments[index]
        file = self.files[i_file]
        signal = self._read_file(d, file, i_start, i_end)
        if signal.shape[-1] < i_end - i_start:
            if self.segment_strategy not in ["pad", "random"]:
                raise ValueError(
                    "cannot load segment outside of file range when segment strategy"
                    f"is not in ['pad', 'random'], got '{self.segment_strategy}'"
                )
            signal = F.pad(signal, (0, i_end - i_start - signal.shape[-1]))
        if _return_metadata:
            metadata = {
                "file_index": i_file,
                "segment_index": index,
                "segment_start": i_start,
                "segment_end": i_end,
                "filename": file,
            }
            return signal, metadata
        else:
            return signal

    def _read_file(self, d, file, i_start, i_end):
        # /!\ i_start and i_end are the indices AFTER resampling /!\
        file_obj = self._open_file(d, file)
        info = sf.info(file_obj)
        j_start = round(i_start * info.samplerate / self.fs)
        j_end = round(i_end * info.samplerate / self.fs)
        # torchaudio is unreliable for reading from file-like objects
        # see https://github.com/pytorch/audio/issues/2356
        # use soundfile instead
        file_obj.seek(0)
        x, fs = sf.read(
            file_obj,
            start=j_start,
            stop=j_end,
            dtype=self.dtype,
            always_2d=True,
        )
        file_obj.close()
        if fs != self.fs:
            warnings.warn(
                "The file sampling rate does not match the `fs` attribute of the "
                f"dataset instance (got {fs} and {self.fs}). Files will be resampled."
            )
            x = soxr.resample(x, fs, self.fs)
        if self.segment_strategy in ["pad", "random"]:
            assert len(x) <= i_end - i_start, (len(x), i_end - i_start)
        else:
            assert len(x) == i_end - i_start, (len(x), i_end - i_start)
        if self.squeeze_channels and x.shape[1] == 1:
            x = x.squeeze(1)
        return torch.from_numpy(x.T)

    def __len__(self):
        """Return the number of segments in the dataset."""
        return len(self.segments)

    def preload(self, device, tqdm_desc=None):
        """Load the dataset to memory."""
        if self.segment_strategy == "random":
            raise ValueError("can't preload when segment_strategy is 'random'")
        preloaded_data = []
        for i in tqdm(range(len(self)), file=sys.stdout, desc=tqdm_desc):
            signals = self[i]
            if isinstance(signals, tuple):
                signals = tuple(x.to(device) for x in signals)
            else:
                signals = signals.to(device)
            preloaded_data.append(signals)
        # set the preloaded_data attribute only at the end, otherwise __getitem__ will
        # attempt to access it inside the loop, causing infinite recursion
        self.preloaded_data = preloaded_data

    def set_epoch(self, epoch):
        """Set the internal epoch counter.

        Useful for reproducibility when training is resumed.
        """
        new_epoch = epoch != self._epoch
        self._epoch = epoch
        if (
            new_epoch
            and self.segment_length is not None
            and self.segment_strategy == "random"
        ):
            self.segments = self._get_segments()

    def get_segment_length(self, i):
        """Get the length of i-th segment without loading it.

        This method is used by :class:`~mbchl.data.batching.AudioBatchSampler` to form
        batches based on the length of the segments.
        """
        if self.segment_length is not None and self.segment_strategy == "random":
            output = self.segment_length
        else:
            _, (start, end) = self.segments[i]
            output = end - start
        return output


@DatasetRegistry.register("random")
class RandomAudioDataset(torch.utils.data.IterableDataset):
    """Iterable-style random audio file dataset.

    The list of files is built at initialization and a random file is selected with
    replacement at each iteration.

    Parameters
    ----------
    dirs : str or list[str]
        Directories containing the audio files.
    regex : str, optional
        Regular expression to filter the audio files.
    n_files : int, optional
        Number of files to randomly load. If ``None``, an infinite number of files is
        loaded.
    dtype : str, optional
        Data type to cast the audio files to.
    tensor : bool, optional
        Whether to yield tensors. If ``True``, the dataset yields a tuple ``(x, fs)``
        where ``x`` is a tensor with shape ``(n_channels, n_samples)`` and ``fs`` is a
        tensor with shape ``(1,)``. If ``False``, ``x`` is a NumPy array with shape
        ``(n_samples, n_channels)`` (note the transposed shape) and ``fs`` is an int.

    """

    # TODO: make reproducible
    # TODO: add option to shuffle once per epoch and iterate sequentially instead of
    # randomly accessing a file at each iteration which is very slow

    def __init__(
        self,
        dirs,
        regex=r"^.*\.(wav|WAV|flac|FLAC)$",
        n_files=None,
        dtype="float32",
        tensor=True,
    ):
        self.n_files = float("inf") if n_files is None else n_files
        self.dtype = dtype
        self.tensor = tensor
        self.files = find_files(dirs, regex=regex, cache=True)

    def __iter__(self):
        """Iterate over the dataset."""
        _count = 0
        n_files = _get_worker_length(self.n_files)
        while _count < n_files:
            file = random.choice(self.files)
            x, fs = sf.read(file, dtype=self.dtype)
            if self.tensor:
                x, fs = torch.from_numpy(x.T), torch.tensor([fs])
            yield x, fs, {}
            _count += 1

    def __len__(self):
        """Return the length of the dataset. Can be ``inf``."""
        return self.n_files


@DatasetRegistry.register("remote")
class RemoteAudioDataset(torch.utils.data.IterableDataset):
    """Iterable-style remote audio file dataset.

    Yields audio files by reading from a stream of tar archives which can be accessed
    remotely. Since audio files are yielded as archives are downloaded, no global
    shuffling is possible. Shuffling is achieved by storing files into a buffer and
    shuffling the buffer before yielding.

    By default, at the start of a new epoch, the dataset resumes from where it left off,
    such that the same files are not yielded again. For this to work with multiple
    DataLoader workers, set ``persistent_workers=True``. To disable this behavior, set
    ``resume`` to ``False``.

    Parameters
    ----------
    url : str or list[str]
        URL of the tar archives. Must be a formattable string of the form
        ``protocol://...`` such that calling ``url.format(i)`` yields the URL of the
        i-th archive. Supported protocols are ``https``, ``s3`` and ``ssh``. For local
        files, drop the protocol and directly use a formattable path. Examples:
        ``path/to/archive-{:02d}.tar``, ``https://example.com/archive-{:02d}.tar``,
        ``s3://bucket/archive-{:02d}.tar``, ``ssh://user@host:archive-{:02d}.tar``.
        If a list of URLs is provided, all URLs must be the same except for the last
        part.
    n_archives : int
        Number of tar archives. This is required to call ``url.format(i)`` for ``i`` in
        ``range(n_archives)``. When using multiple DataLoader workers, the number of
        archives must be greater than or equal to the number of workers, and ideally a
        multiple of the number of workers. If ``url`` is a list of URLs, each URL must
        be formattable with the same number of archives.
    n_files : int, optional
        Number of files to load from the archives. If ``None``, files are yielded until
        the archives are exhausted. If greater than the number of files in the archives,
        iteration stops early or loops over depending on the ``loop`` parameter.
    loop : bool, optional
        Whether to loop over the archives. If ``True`` and ``n_files`` is ``None``, the
        dataset is infinite.
    buffer_size : int, optional
        Number of files to buffer for shuffling. If ``None``, no buffering is performed.
        Currently, if training is resumed, the training examples that were in the buffer
        before interruption are not re-downloaded.
    resume : bool, optional
        Whether to resume the stream from where it left off when calling
        :meth:`__iter__` again.
    dtype : str, optional
        Data type to cast the audio files to.
    tensor : bool, optional
        Whether to yield tensors. If ``True``, the dataset yields a tuple ``(x, fs)``
        where ``x`` is a tensor with shape ``(n_channels, n_samples)`` and ``fs`` is a
        tensor with shape ``(1,)``. If ``False``, ``x`` is a NumPy array with shape
        ``(n_samples, n_channels)`` (note the transposed shape) and ``fs`` is an int.
    seed : int, optional
        Random seem to shuffle the archives.

    """

    # TODO: fix training examples in buffer not being re-downloaded when resuming
    # TODO: fix pathological case where n_files matches the number of files in the
    # archive and loop is False; iterating again yields files from the beginning of the
    # archive even though loop is False!

    def __init__(
        self,
        url,
        n_archives,
        n_files=None,
        loop=False,
        buffer_size=None,
        resume=True,
        dtype="float32",
        tensor=True,
        seed=0,
    ):
        if isinstance(url, str):
            url = [url]
        base_urls, archive_fmts = zip(*[os.path.split(u) for u in url])
        if not all(base_url == base_urls[0] for base_url in base_urls):
            raise ValueError("all URLs must be the same except for the last part")
        self.base_url = base_urls[0]
        self.archive_fmts = archive_fmts
        self.n_archives = n_archives
        self.n_files = float("inf") if n_files is None else n_files
        self.loop = loop
        self.buffer_size = buffer_size
        self.resume = resume
        self.dtype = dtype
        self.tensor = tensor
        self.seed = seed
        self._worker_state = None
        self._buffer = None if buffer_size is None else []

    def __iter__(self):
        """Iterate over the dataset."""
        # Wrapping _iter to prevent a meaningless "KeyError: 'error'"" error caused by
        # DataLoader recalling `botocore` exception constructors with bad arguments when
        # gracefully handling errors. This causes the original traceback to be lost.
        try:
            yield from self._iter()
        except Exception as e:
            raise RemoteDatasetIterationError(
                "An error occurred while iterating over the dataset"
            ) from e

    def _iter(self):
        worker_info = torch.utils.data.get_worker_info()
        worker_id = 0 if worker_info is None else worker_info.id
        if self._worker_state is None:
            self._worker_state = {worker_id: {"archive": 0, "offset": 0}}
        num_workers = 1 if worker_info is None else worker_info.num_workers
        if num_workers > self.n_archives:
            raise ValueError(
                "number of workers must be less than or equal to the number of archives"
            )
        archives = [
            archive_fmt.format(i)
            for archive_fmt in self.archive_fmts
            for i in range(self.n_archives)
            if i % num_workers == worker_id
        ]
        self._shuffle_archives(archives, worker_id)
        n_files = _get_worker_length(self.n_files, worker_id, num_workers)
        fs = init_filesystem(self.base_url)
        _count = 0
        _stop_flag = False
        if not self.resume:
            self._worker_state[worker_id]["archive"] = 0
            self._worker_state[worker_id]["offset"] = 0
        while True:
            for i in range(self._worker_state[worker_id]["archive"], len(archives)):
                self._worker_state[worker_id]["archive"] = i
                archive = archives[i]
                # check if we coincidentally reached end of archive last epoch
                if self._worker_state[worker_id]["offset"] >= fs.size(archive):
                    # manually update offset and continue to next archive to prevent a
                    # tarfile.ReadError when attempting to read from end of archive
                    logging.debug(f"[worker{worker_id}] Reached end of {archive}")
                    self._worker_state[worker_id]["offset"] = 0
                    continue
                logging.debug(
                    f"[worker{worker_id}] Opening {archive} at "
                    f"{self._worker_state[worker_id]["offset"]}"
                )
                with (
                    fs.open(
                        archive, "rb", offset=self._worker_state[worker_id]["offset"]
                    ) as f,
                    tarfile.open(
                        mode="r|",
                        fileobj=f,
                        ignore_zeros=self._worker_state[worker_id]["offset"] != 0,
                    ) as tar,
                ):
                    # ignore_zeros=True prevents a header error when reading from a
                    # non-zero offset in the archive
                    for tarinfo in tar:
                        logging.debug(f"[worker{worker_id}] {archive}/{tarinfo.name}")
                        try:
                            x, sr = sf.read(
                                io.BytesIO(tar.extractfile(tarinfo).read()),
                                dtype=self.dtype,
                            )
                        except sf.LibsndfileError:
                            logging.error(
                                f"Error reading archive member '{tarinfo.name}' in "
                                f"'{archive}', skipping"
                            )
                            continue
                        self._worker_state[worker_id]["offset"] = f.tell()
                        if self.tensor:
                            x, sr = torch.from_numpy(x.T), torch.tensor([sr])
                        # yield the state of the current worker only, not all workers
                        _current_worker_state = {
                            worker_id: self._worker_state[worker_id]
                        }
                        if self.buffer_size is None:
                            yield x, sr, _current_worker_state
                            _count += 1
                            _stop_flag = _count >= n_files
                        else:
                            self._buffer.append((x, sr, _current_worker_state))
                            if len(self._buffer) >= self.buffer_size:
                                random.shuffle(self._buffer)
                                while self._buffer:
                                    yield self._buffer.pop()
                                    _count += 1
                                    _stop_flag = _count >= n_files
                                    if _stop_flag:
                                        break
                        if _stop_flag:
                            break
                    logging.debug(
                        f"[worker{worker_id}] Closing {archive} at "
                        f"{self._worker_state[worker_id]["offset"]}"
                    )
                if _stop_flag:
                    break
                # reset the offset for the next archive
                self._worker_state[worker_id]["offset"] = 0
            if _stop_flag:
                break
            # restart the iteration from the first archive and reset the offset
            self._worker_state[worker_id]["offset"] = 0
            self._worker_state[worker_id]["archive"] = 0
            if self.loop:
                logging.debug(f"[worker{worker_id}] Looping.")
            else:
                # handle files left in the buffer before exiting
                if self._buffer:
                    random.shuffle(self._buffer)
                    while self._buffer:
                        yield self._buffer.pop()
                        _count += 1
                        _stop_flag = _count >= n_files
                        if _stop_flag:
                            break
                break

    def _shuffle_archives(self, archives, worker_id):
        # ensure consistent and unique shuffling for each worker
        seed_seed = random.Random(self.seed).randrange(2**32)
        worker_seed = random.Random(worker_id).randrange(2**32)
        random.Random(seed_seed ^ worker_seed).shuffle(archives)

    def __len__(self):
        """Return the length of the dataset. Can be ``inf``."""
        return self.n_files

    def load_state_dict(self, state_dict):
        """Load the state of the dataset."""
        # this is loaded only once and then copied to each worker
        # so it must contain the state of all workers
        self._worker_state = state_dict


@DatasetRegistry.register("dynamic")
class DynamicAudioDataset(torch.utils.data.IterableDataset):
    """Dynamic audio dataset.

    Noisy and reverberant mixture are generated on-the-fly by mixing speech and noise
    signals convolved with simulated room impulse responses. The room impulse responses
    are simulated on-the-fly using [1].

    .. [1] Y. Luo and R. Gu, "Fast Random Approximation of Multi-Channel Room Impulse
       Response", in Proc. ICASSP, 2024.

    Parameters
    ----------
    length : int
        Length of the dataset.
    fs : int
        Sampling rate.
    speech_dataset : {"random", "remote"}
        Dataset type for speech signals. If ``"random"``, uses
        :class:`~mbchl.data.datasets.RandomAudioDataset`. If ``"remote"``, uses
        :class:`~mbchl.data.datasets.RemoteAudioDataset`.
    speech_dataset_kw : dict
        Keyword arguments for the speech dataset.
    noise_dataset : {"random", "remote"}
        Dataset type for noise signals. If ``"random"``, uses
        :class:`~mbchl.data.datasets.RandomAudioDataset`. If ``"remote"``, uses
        :class:`~mbchl.data.datasets.RemoteAudioDataset`.
    noise_dataset_kw : dict
        Keyword arguments for the noise dataset.
    segment_length : float, optional
        Segment length in seconds. If specified, speech and noise segments are randomly
        sampled from the files, and the output mixtures all have the same duration. Note
        that segments can loop around the end of the file. If ``None``, the files are
        used as is and the output mixtures can have variable duration.
    num_noises_range : tuple, optional
        Range for the number of noise sources in the mixture. Bounds are inclusive.
    room_dim_range : tuple, optional
        Range for the room dimensions in meters.
    t60_range : tuple, optional
        Range for the reverberation time in seconds.
    snr_range : tuple, optional
        Range for the signal-to-noise ratio in dB.
    dbfs_range : tuple, optional
        Range for the digital full-scale level in dB.
    transform : callable, optional
        Transform to apply to the mixture and target. The transform should take a tuple
        of two tensors (mixture and target) and return a tuple of two tensors.
    seed : int, optional
        Random seed.
    audiogram : bool, optional
        If ``True``, a random audiogram is yielded at each iteration.
    audiogram_jitter : float, optional
        Maximum jitter in dB for the audiogram.

    """

    def __init__(
        self,
        length,
        fs,
        speech_dataset,
        speech_dataset_kw,
        noise_dataset,
        noise_dataset_kw,
        segment_length=None,
        num_noises_range=(1, 3),
        room_dim_range=((3.0, 3.0, 2.5), (10.0, 10.0, 4.0)),
        t60_range=(0.1, 0.7),
        snr_range=(-10, 20),
        dbfs_range=(-20, 0),
        transform=None,
        seed=0,
        audiogram=False,
        audiogram_jitter=10.0,
        _zero_audiogram=False,
        _rir_dset=None,
    ):
        self.length = length
        self.fs = fs
        self.speech_dataset = DatasetRegistry.get(speech_dataset)(**speech_dataset_kw)
        self.noise_dataset = DatasetRegistry.get(noise_dataset)(**noise_dataset_kw)
        self.seg_len = None if segment_length is None else int(segment_length * fs)
        self.num_noises_range = num_noises_range
        self.room_dim_range = room_dim_range
        self.t60_range = t60_range
        self.snr_range = snr_range
        self.dbfs_range = dbfs_range
        self.transform = (lambda x: x) if transform is None else transform
        self.seed = seed
        self.audiogram = audiogram
        self.audiogram_jitter = audiogram_jitter
        self._zero_audiogram = _zero_audiogram
        self._rir_dset = _rir_dset
        self._epoch = 0

    def __iter__(self):
        """Iterate over the dataset."""
        worker_info = torch.utils.data.get_worker_info()
        num_workers = 1 if worker_info is None else worker_info.num_workers
        worker_id = 0 if worker_info is None else worker_info.id
        length = self.length // num_workers + (worker_id < self.length % num_workers)
        generator = self._get_generator(worker_id)
        speech_iter = iter(self.speech_dataset)
        noise_iter = iter(self.noise_dataset)
        for _ in range(length):
            # sample speech
            speech = 0
            while np.equal(speech, 0).all():
                speech, speech_fs, speech_worker_state = next(speech_iter)
                speech = self._segment_and_resample(
                    speech, self.seg_len, speech_fs, self.fs, generator
                )
            # sample noises
            num_noise = generator.integers(
                self.num_noises_range[0], self.num_noises_range[1] + 1
            )
            noises = []
            while len(noises) < num_noise:
                noise, noise_fs, noise_worker_state = next(noise_iter)
                noise = self._segment_and_resample(
                    noise, len(speech), noise_fs, self.fs, generator
                )
                # check noise power to prevent error in set_snr
                if noise.var() < np.finfo(noise.dtype).tiny:
                    warnings.warn("Noise power is too small. Skipping.")
                else:
                    noises.append(noise)
            mix, target = _make_random_scene(
                speech=speech,
                noises=noises,
                fs=self.fs,
                room_dim_range=self.room_dim_range,
                t60_range=self.t60_range,
                snr_range=self.snr_range,
                dbfs_range=self.dbfs_range,
                generator=generator,
                _rir_dset=self._rir_dset,
            )
            mix, target = torch.from_numpy(mix), torch.from_numpy(target)
            mix, target = mix.unsqueeze(0), target.unsqueeze(0)
            if self.audiogram:
                audiogram = random_audiogram(generator, jitter=self.audiogram_jitter)
                if self._zero_audiogram:
                    audiogram[:, 1] = 0
                audiogram = torch.from_numpy(audiogram).to(mix.dtype)
                mix, target, audiogram = self.transform((mix, target, audiogram))
                yield mix, target, audiogram, speech_worker_state, noise_worker_state
            else:
                mix, target = self.transform((mix, target))
                yield mix, target, speech_worker_state, noise_worker_state

    def __len__(self):
        """Return the length of the dataset."""
        return self.length

    def _segment_and_resample(self, x, seg_len, fs, target_fs, generator):
        if x.ndim != 1:
            warnings.warn("Found a file that is not mono. Using first channel.")
            x = x[:, 0]
        if seg_len is not None:
            seg_len_pre = round(seg_len * fs / target_fs)
            x = self._sample_random_segment(x, seg_len_pre, generator)
        if fs != target_fs:
            warnings.warn(f"Resampling from {fs} Hz to {target_fs} Hz")
            x = soxr.resample(x, fs, target_fs)
        if seg_len is not None:
            assert len(x) == seg_len
        return x

    def _sample_random_segment(self, x, n, generator, i_min=0, i_max=None):
        if i_max is None:
            i_max = len(x)
        i_start = generator.integers(i_min, i_max)
        indices = (np.arange(n) + i_start) % (i_max - i_min) + i_min
        return x[indices]

    def set_epoch(self, epoch):
        """Set the internal epoch counter.

        Useful for reproducibility when training is resumed.
        """
        self._epoch = epoch

    def _get_generator(self, worker_id):
        # initialize a unique seeded NumPy generator for each worker
        seed_1 = random.Random(self.seed).randrange(2**32)
        seed_2 = random.Random(worker_id).randrange(2**32)
        seed_3 = random.Random(self._epoch).randrange(2**32)
        seed = seed_1 ^ seed_2 ^ seed_3
        return np.random.default_rng(seed)

    def load_state_dict(self, state_dict):
        """Load the state dictionary."""
        # state_dict can be empty if the dataset was not iterated over in previous run
        # for example a validation dataset that is not used until after some epochs
        # TODO: check if dataset was iterated and raise error if state_dict is empty
        if not state_dict:
            return
        speech_worker_state, noise_worker_state = state_dict
        self.speech_dataset.load_state_dict(speech_worker_state)
        self.noise_dataset.load_state_dict(noise_worker_state)


class TarArchive:
    """Tar archive interface compatible with multiple DataLoader workers."""

    # Copyright (c) 2021 Joao F. Henriques
    # https://github.com/jotaf98/simple-tar-dataset
    # BSD 3-Clause License

    def __init__(self, archive):
        worker = torch.utils.data.get_worker_info()
        worker = worker.id if worker else None
        self.tar_obj = {worker: tarfile.open(archive)}
        self.archive = archive
        self.members = {m.name: m for m in self.tar_obj[worker].getmembers()}

    def _get_file(self, name):
        # ensure a unique file handle per worker
        worker = torch.utils.data.get_worker_info()
        worker = worker.id if worker else None
        if worker not in self.tar_obj:
            self.tar_obj[worker] = tarfile.open(self.archive)
        return self.tar_obj[worker].extractfile(self.members[name])


class RemoteDatasetIterationError(Exception):
    """Error raised when iterating over a :class:`RemoteAudioDataset` fails."""


class DNS5RIRDataset(torch.utils.data.Dataset):
    """DNS5 room impulse response dataset."""

    def __init__(self, path, fs, dtype="float32"):
        self.rirs = [
            find_files(f"{path}/{relpath}", regex=regex, cache=True)
            for relpath, regex in [
                *[
                    (f"SLR26/simulated_rirs_48k/{room}/Room{i + 1:03d}", r"^.*\.wav$")
                    for room in ["smallroom", "mediumroom", "largeroom"]
                    for i in range(200)
                ],
                *[
                    ("SLR28/RIRS_NOISES/real_rirs_isotropic_noises/", regex)
                    for regex in [
                        r"^.*air_type1_air_binaural_aula_carolina_1_.*\.wav$",
                        r"^.*air_type1_air_binaural_booth_0_.*\.wav$",
                        r"^.*air_type1_air_binaural_booth_1_.*\.wav$",
                        r"^.*air_type1_air_binaural_lecture_0_.*\.wav$",
                        r"^.*air_type1_air_binaural_lecture_1_.*\.wav$",
                        r"^.*air_type1_air_binaural_meeting_0_.*\.wav$",
                        r"^.*air_type1_air_binaural_meeting_1_.*\.wav$",
                        r"^.*air_type1_air_binaural_office_0_.*\.wav$",
                        r"^.*air_type1_air_binaural_office_1_.*\.wav$",
                        r"^.*air_type1_air_binaural_stairway_1_.*\.wav$",
                        r"^.*air_type1_air_phone_bathroom_hfrp\.wav$",
                        r"^.*air_type1_air_phone_bathroom_hhp\.wav$",
                        r"^.*air_type1_air_phone_bt_corridor_hhp\.wav$",
                        r"^.*air_type1_air_phone_bt_office_hhp\.wav$",
                        r"^.*air_type1_air_phone_bt_stairway_hhp\.wav$",
                        r"^.*air_type1_air_phone_corridor_hfrp\.wav$",
                        r"^.*air_type1_air_phone_corridor_hhp\.wav$",
                        r"^.*air_type1_air_phone_kitchen_hfrp\.wav$",
                        r"^.*air_type1_air_phone_kitchen_hhp\.wav$",
                        r"^.*air_type1_air_phone_lecture1_hfrp\.wav$",
                        r"^.*air_type1_air_phone_lecture1_hhp\.wav$",
                        r"^.*air_type1_air_phone_lecture_hfrp\.wav$",
                        r"^.*air_type1_air_phone_lecture_hhp\.wav$",
                        r"^.*air_type1_air_phone_meeting_hfrp\.wav$",
                        r"^.*air_type1_air_phone_meeting_hhp\.wav$",
                        r"^.*air_type1_air_phone_office_hfrp\.wav$",
                        r"^.*air_type1_air_phone_office_hhp\.wav$",
                        r"^.*air_type1_air_phone_stairway1_hfrp\.wav$",
                        r"^.*air_type1_air_phone_stairway1_hhp\.wav$",
                        r"^.*air_type1_air_phone_stairway2_hfrp\.wav$",
                        r"^.*air_type1_air_phone_stairway2_hhp\.wav$",
                        r"^.*air_type1_air_phone_stairway_hfrp\.wav$",
                        r"^.*air_type1_air_phone_stairway_hhp\.wav$",
                        r"^.*RWCP_type1_rir_circle_ane_.*\.wav$",
                        r"^.*RWCP_type1_rir_circle_e1a_.*\.wav$",
                        r"^.*RWCP_type1_rir_circle_e1b_.*\.wav$",
                        r"^.*RWCP_type1_rir_circle_e1c_.*\.wav$",
                        r"^.*RWCP_type1_rir_cirline_e2a_.*\.wav$",
                        r"^.*RWCP_type1_rir_cirline_e2b_.*\.wav$",
                        r"^.*RWCP_type1_rir_cirline_jr1_.*\.wav$",
                        r"^.*RWCP_type1_rir_cirline_jr2_.*\.wav$",
                        r"^.*RWCP_type1_rir_cirline_ofc_.*\.wav$",
                        r"^.*RWCP_type2_rir_cirline_e2a_.*\.wav$",
                        r"^.*RWCP_type2_rir_cirline_e2b_.*\.wav$",
                        r"^.*RWCP_type2_rir_cirline_jr1_.*\.wav$",
                        r"^.*RWCP_type2_rir_cirline_ofc_.*\.wav$",
                        r"^.*RWCP_type3_rir_cirline_ofc_.*\.wav$",
                        r"^.*RWCP_type4_rir_.*l\.wav$",
                        r"^.*RWCP_type4_rir_.*r\.wav$",
                    ]
                ],
            ]
        ]
        n_files = 60248
        n_found = sum(len(rirs) for rirs in self.rirs)
        if n_found != n_files:
            raise ValueError(f"expected {n_files} impulse responses, found {n_found}")
        self.fs = fs
        self.dtype = dtype

    def get_item(self, i_room, n, generator=None):
        """Get a number of impulse responses from the same room."""
        if generator is None:
            generator = random.Random()
        elif isinstance(generator, np.random.Generator):
            generator = random.Random(generator.integers(2**32).item())
        rirs = self.rirs[i_room]
        rirs = generator.choices(rirs, k=n)
        rirs = np.stack([self._load_rir(rir, generator) for rir in rirs])
        rirs_early, rirs_late = split_rir(rirs)
        return rirs, rirs_early

    def _load_rir(self, path, generator):
        x, fs = sf.read(path, dtype=self.dtype, always_2d=True)
        # pick one random channel
        # TODO: this might not be the best way to handle multi-channel RIRs
        x = x[:, generator.randrange(x.shape[1])]
        if fs != self.fs:
            warnings.warn(f"Resampling RIR from {fs} Hz to {self.fs} Hz")
            x = soxr.resample(x, fs, self.fs)
        return x


def _get_worker_length(dataset_length, worker_id=None, num_workers=None):
    if dataset_length == float("inf"):
        return float("inf")
    elif not isinstance(dataset_length, int):
        raise TypeError(f"dataset length must be int or inf, got {dataset_length}")
    elif dataset_length < 0:
        raise ValueError(f"dataset length must be non-negative, got {dataset_length}")
    if worker_id is None or num_workers is None:
        worker_info = torch.utils.data.get_worker_info()
        worker_id = 0 if worker_info is None else worker_info.id
        num_workers = 1 if worker_info is None else worker_info.num_workers
    quotient, remainder = divmod(dataset_length, num_workers)
    return quotient + (worker_id < remainder)


def _make_random_scene(
    speech,
    noises,
    fs,
    room_dim_range,
    t60_range,
    snr_range,
    dbfs_range,
    generator,
    _rir_dset,
):
    num_noise = len(noises)
    if _rir_dset is None:
        rir, early_rir = fram_rir(
            sr=fs,
            num_src=1 + num_noise,
            num_mic=1,
            room_dim_range=room_dim_range,
            t60_range=t60_range,
            generator=generator,
        )
        rir, early_rir = rir.squeeze(0), early_rir.squeeze(0)
    else:
        rir, early_rir = _rir_dset.get_item(0, 1 + num_noise, generator)
    target = scipy.signal.oaconvolve(speech, early_rir[0])
    mix = scipy.signal.oaconvolve(speech, rir[0])
    target, mix = target[: len(speech)], mix[: len(speech)]
    for i in range(num_noise):
        noise = noises[i]
        noise = scipy.signal.oaconvolve(noise, rir[1 + i])
        noise = noise[: len(speech)]
        snr = generator.uniform(*snr_range)
        try:
            noise = set_snr(target, noise, snr)
        except NoiseTooSmallError:
            warnings.warn("Noise power too small after convolving with RIR. Skipping.")
            continue
        mix = mix + noise
    dbfs = generator.uniform(*dbfs_range)
    mix, factor = set_dbfs(mix, dbfs, mode="peak", return_factor=True)
    return mix, target * factor


def split_rir(rir, reflection_boundary=50, fs=16000):
    """Split a room impulse response into early and late reflection.

    Parameters
    ----------
    rir : numpy.ndarray
        Room impulse response. Shape `(n_channels, rir_length)`.
    reflection_boundary : float, optional
        Reflection boundary in milliseconds.
    fs : int, optional
        Sampling frequency.

    Returns
    -------
    rir_early: numpy.ndarray
        Early reflection part. Shape `(n_channels, rir_length)`.
    rir_late: numpy.ndarray
        Late reflection part. Shape `(n_channels, rir_length)`.

    """
    i_max = np.argmax(np.abs(rir), axis=1)
    win_early = np.zeros(rir.shape)
    for i_ch in range(rir.shape[0]):
        win_early[i_ch, : i_max[i_ch] + round(1e-3 * reflection_boundary * fs)] = 1
    rir_early = win_early * rir
    rir_late = (1 - win_early) * rir
    return rir_early, rir_late

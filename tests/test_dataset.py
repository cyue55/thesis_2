import os
import shutil
import tarfile

import numpy as np
import pytest
import soundfile as sf

from mbchl.data.dataloader import AudioDataLoader
from mbchl.data.datasets import AudioDataset

FS = 8000
BATCH_SIZE = 2
SEGMENT_LENGTH = 2.0


def create_test_dirs(
    tmp_path,
    np_rng,
    dirnames="audio",
    n_files=10,
    n_channels=1,
    min_length=1.0,
    max_length=4.0,
    fs=8000,
    tar=True,
):
    min_length = round(min_length * fs)
    max_length = round(max_length * fs)
    dirnames_is_list = isinstance(dirnames, list)
    if not dirnames_is_list:
        dirnames = [dirnames]
    dirpaths = []
    lengths = []
    for i_dir, dirname in enumerate(dirnames):
        dirpath = tmp_path / dirname
        if i_dir == 0:
            os.mkdir(dirpath)
            for i_file in range(n_files):
                length = np_rng.integers(min_length, max_length)
                x = np_rng.standard_normal((length, n_channels))
                sf.write(dirpath / f"{i_file}.wav", x, fs)
                lengths.append(length)
        else:
            shutil.copytree(tmp_path / dirnames[0], dirpath)
        dirpaths.append(dirpath)
    if tar:
        for i_dir, dirpath in enumerate(dirpaths):
            tarpath = dirpaths[i_dir].with_suffix(".tar")
            with tarfile.open(tarpath, "w") as f:
                f.add(dirpath, arcname=".")
            shutil.rmtree(dirpaths[i_dir])
            dirpaths[i_dir] = tarpath
    dirpaths = [dirpath.as_posix() for dirpath in dirpaths]
    if not dirnames_is_list:
        dirpaths = dirpaths[0]
    return dirpaths, lengths, fs


@pytest.fixture(scope="module")
def default_test_dir(tmp_path_factory):
    tmp_path = tmp_path_factory.mktemp("default_test_dir")
    # cannot use np_rng fixture because its scope is "function"
    np_rng = np.random.default_rng(seed=0)
    return create_test_dirs(tmp_path, np_rng)


@pytest.mark.parametrize("dirnames", ["audio", ["audio"], ["audio_0", "audio_1"]])
@pytest.mark.parametrize("n_channels", [1, 2])
@pytest.mark.parametrize("tar", [False, True])
def test_ndim(tmp_path, np_rng, dirnames, n_channels, tar):
    dirs, _, _ = create_test_dirs(
        tmp_path,
        np_rng,
        dirnames=dirnames,
        n_channels=n_channels,
        fs=FS,
        tar=tar,
    )
    for squeeze_channels in [False, True]:
        dset = AudioDataset(dirs, fs=FS, squeeze_channels=squeeze_channels)
        ndim = 1 + (n_channels > 1 or not squeeze_channels)
        for x in dset:
            if isinstance(dirs, list):
                assert isinstance(x, tuple)
                for x_ in x:
                    assert x_.ndim == ndim
            else:
                assert x.ndim == ndim
        dloader = AudioDataLoader(dset, batch_size=2)
        for x, lengths in dloader:
            if isinstance(dirs, list):
                assert isinstance(x, tuple)
                for x_ in x:
                    assert x_.ndim == ndim + 1
                assert lengths.shape == (BATCH_SIZE, len(dirs))
            else:
                assert x.ndim == ndim + 1
                assert lengths.shape == (BATCH_SIZE,)


@pytest.mark.parametrize(
    "segment_strategy", ["drop", "pass", "pad", "overlap", "random"]
)
@pytest.mark.parametrize(
    "segment_length, overlap_length",
    [
        (None, 0.0),
        (1.5, 0.0),
        (1.5, 0.5),
    ],
)
def test_segmentation(
    default_test_dir, segment_strategy, segment_length, overlap_length
):
    d, lengths, fs = default_test_dir
    dset = AudioDataset(
        d,
        fs=fs,
        segment_strategy=segment_strategy,
        segment_length=segment_length,
        overlap_length=overlap_length,
    )
    if segment_length is not None:
        segment_length = round(segment_length * fs)
        hop_length = segment_length - round(overlap_length * fs)
    if segment_length is None or segment_strategy == "random":
        n_segments = len(lengths)
    elif segment_strategy == "drop":
        n_segments = sum((l_ - segment_length) // hop_length + 1 for l_ in lengths)
    elif segment_strategy == "overlap":
        n_segments = sum(
            np.ceil((l_ - segment_length) / hop_length) + 1
            for l_ in lengths
            if l_ >= segment_length
        )
    else:
        n_segments = sum(
            np.ceil((l_ - segment_length) / hop_length) + 1 for l_ in lengths
        )
    assert len(dset) == n_segments
    for i, x in enumerate(dset):
        if segment_length is None:
            assert x.shape[-1] == lengths[i]
        elif segment_strategy == "pass":
            assert x.shape[-1] <= segment_length
        else:
            assert x.shape[-1] == segment_length


def test_random_segment_reproducibility(default_test_dir):
    d, _, fs = default_test_dir

    # two datasets with the same seed
    dset_1 = AudioDataset(
        d,
        fs=fs,
        segment_length=SEGMENT_LENGTH,
        segment_strategy="random",
        seed=0,
    )
    dset_2 = AudioDataset(
        d,
        fs=fs,
        segment_length=SEGMENT_LENGTH,
        segment_strategy="random",
        seed=0,
    )
    dloader_1 = AudioDataLoader(dset_1, batch_size=2)
    dloader_2 = AudioDataLoader(dset_2, batch_size=2)

    # iterating the first time should yield the same
    x_1, _ = next(iter(dloader_1))
    x_2, _ = next(iter(dloader_2))
    assert (x_1 == x_2).all()

    # iterating after setting the same epoch should yield the same
    dloader_1.set_epoch(42)
    dloader_2.set_epoch(42)
    x_1, _ = next(iter(dloader_1))
    x_2, _ = next(iter(dloader_2))
    assert (x_1 == x_2).all()

    # iterating after setting a different epoch should yield different results
    dloader_1.set_epoch(0)
    dloader_2.set_epoch(1)
    x_1, _ = next(iter(dloader_1))
    x_2, _ = next(iter(dloader_2))
    assert (x_1 != x_2).any()

    # two datasets with the different seeds
    dset_1 = AudioDataset(
        d,
        fs=fs,
        segment_length=SEGMENT_LENGTH,
        segment_strategy="random",
        seed=7,
    )
    dset_2 = AudioDataset(
        d,
        fs=fs,
        segment_length=SEGMENT_LENGTH,
        segment_strategy="random",
        seed=11,
    )
    dloader_1 = AudioDataLoader(dset_1, batch_size=2)
    dloader_2 = AudioDataLoader(dset_2, batch_size=2)

    # iterating the first time should yield different results
    x_1, _ = next(iter(dloader_1))
    x_2, _ = next(iter(dloader_2))
    assert (x_1 != x_2).any()

    # results should be different even when setting the same epoch
    dloader_1.set_epoch(42)
    dloader_2.set_epoch(42)
    x_1, _ = next(iter(dloader_1))
    x_2, _ = next(iter(dloader_2))
    assert (x_1 != x_2).any()

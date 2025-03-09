Streaming remote datasets
=========================

It it possible to read datasets of audio files that are stored remotely, e.g. via HTTPS or SSH, without the need to download them in their entirety into the local filesystem. This has a few advantages:

* No large volume space is required on the local machine, as the data is streamed directly from the remote server.
* Training can be started immediately, without the need for waiting for the dataset to be downloaded.
* By reading from the same remote dataset across machines, reproducibility is ensured.

These advantages are particularly useful for debugging, and for training models with cloud services, where a virtual machine is started for each training run, local storage is limited, and billing is by the minute.

The major limitation with streaming remote datasets is that global shuffling is not possible. Shuffling is performed in different ways described further below.

Unless you are training a very small model and using a single worker, data loading is usually not the bottleneck. So streaming data from a remote server should not increase training times.

The solution implemented here is similar to `WebDataset <https://github.com/webdataset/webdataset>`_, but I was never able to wrap my head around it, so I implemented my own solution.


Splitting and storing files as shards
-------------------------------------

Storing files on the remote server individually and opening a connection for each file would be extremely inefficient due to latency. Instead, files should be stored in large TAR archives, such that a connection is opened only once per archive and files are unpacked as the archive is downloaded.

To leverage multiple workers, we can split the files into multiple archives or "shards". Each worker can then open a connection and stream a different shard. This significantly speeds up data loading, but makes the results dependent on the number of workers, since the order in which files are presented to the model is not preserved. The set of shards each worker reads are disjoint.

The first step is thus to split the files into shards. This can be done with the ``scripts/split_tar.py`` script. The script has many options:

.. code-block::

  $ python scripts/split_tar.py --help
  usage: split_tar.py [-h] [-o OUTPUT] [-e EXT] [-S] [-c] [-r REGEX] [-C CONVERT] [-f [FFMPEG]]
                      [-y] [-R RESAMPLE] [-s SEGMENT] [-M] [-d MIN_DURATION] [-l MIN_RMS]
                      [--opus_resample]
                      indir n_splits

  positional arguments:
    indir                 Input directory.
    n_splits              Number of splits.

  options:
    -h, --help            show this help message and exit
    -o OUTPUT, --output OUTPUT
                          Output tarfile formatting string.
    -e EXT, --ext EXT     Extension of files to consider in input directory. Default is '.wav'.
    -S, --shuffle         Whether to shuffle files before archiving. If toggled, files are
                          renamed.
    -c, --check           Whether to check if input files are corrupted.
    -r REGEX, --regex REGEX
                          Regular expression to filter files in input directory.
    -C CONVERT, --convert CONVERT
                          Output audio format.
    -f [FFMPEG], --ffmpeg [FFMPEG]
                          FFmpeg command. If not provided, libsndfile is used (recommended).
    -y, --skip            Skip corrupted files.
    -R RESAMPLE, --resample RESAMPLE
                          Output sampling frequency. If not provided, files are not resampled.
    -s SEGMENT, --segment SEGMENT
                          Segment duration. If not provided, files are not segmented.
    -M, --mono            Whether to convert multi-channel files to mono.
    -d MIN_DURATION, --min_duration MIN_DURATION
                          Minimum output file duration in seconds. Useful with --segment to skip
                          short trailing segments.
    -l MIN_RMS, --min_rms MIN_RMS
                          Minimum output file RMS.
    --opus_resample       Whether to resample to the next supported Opus sampling rate. Ignored
                          unless '--convert opus' is used.

For example, the following command copies the WAV files in ``data/external/Clarity/`` into 16 shards named ``data/shards/clarity-00.tar``, ``data/shards/clarity-01.tar``, ..., ``data/shards/clarity-15.tar``, shuffles the files, converts them to mono Opus files, resamples them to the next supported Opus sampling rate, skips corrupted files, and skips files with an RMS below 0.001:

.. code-block:: bash

  python scripts/split_tar.py data/external/Clarity/ 16 -o data/shards/clarity-{}.tar -S -C opus -y -M -l 0.001 --opus_resample

The following command copies the WAV files in ``data/external/TUT/`` into 16 shards named ``data/shards/tut-00.tar``, ``data/shards/tut-01.tar``, ..., ``data/shards/tut-15.tar``, shuffles the files, converts them to mono Opus files, resamples them to the next supported Opus sampling rate, skips corrupted files, segments files into segments with a maximum duration of 15.01 seconds, discards trailing segments and files with a duration below 1 second, and skips segments with an RMS below 0.0001:

.. code-block:: bash

  python scripts/split_tar.py data/external/TUT/ 16 -o data/shards/tut-{}.tar -S -C opus -y -M -s 15.01 -d 1.0 -l 0.0001 --opus_resample


Iterating over shards
---------------------

The :class:`~mbchl.data.datasets.RemoteAudioDataset` class allows to iterate over a set of shards that was created with the ``scripts/split_tar.py`` script. For example:

.. code-block:: python

  from mbchl.data.datasets import RemoteAudioDataset

  dataset = RemoteAudioDataset(
      url="data/shards/clarity-{:02d}.tar",
      n_archives=16,
  )

  for x, fs, worker_state in dataset:
      pass  # do stuff

The dataset yields the audio data ``x``, the sampling frequency ``fs``, and a dictionary ``worker_state`` containing worker state information intended for :class:`~mbchl.data.dataloader.AudioDataLoader` internal use that should not be used by the user.

The ``url`` argument can also be a list of strings to read from multiple sets of shards. For example:

.. code-block:: python

  from mbchl.data.datasets import RemoteAudioDataset

  dataset = RemoteAudioDataset(
      url=[
          "data/shards/clarity-{:02d}.tar",
          "data/shards/vctk-{:02d}.tar",
      ],
      n_archives=16,
  )

  for x, fs, worker_state in dataset:
      pass  # do stuff

In the examples above, the shards are still stored in the local filesystem. The :class:`~mbchl.data.datasets.RemoteAudioDataset` supports reading from shards that are stored remotely. Supported protocols are S3, HTTPS and SSH.

For example, assuming you uploaded the shards to an S3 bucket and you are authenticated in your environment, you can stream the shards as follows:

.. code-block:: python

  from mbchl.data.datasets import RemoteAudioDataset

  dataset = RemoteAudioDataset(
      url="s3://bucket_name/path/to/clarity-{:02d}.tar",
      n_archives=16,
  )

  for x, fs, worker_state in dataset:
      pass  # do stuff

If the shards are accessible via HTTPS, you can stream them as follows:

.. code-block:: python

  from mbchl.data.datasets import RemoteAudioDataset

  dataset = RemoteAudioDataset(
      url="https://example.com/path/to/clarity-{:02d}.tar",
      n_archives=16,
  )

  for x, fs, worker_state in dataset:
      pass  # do stuff

Finally, assuming you have a configured SSH client, you can stream the shards via SSH as follows (much slower than S3 or HTTPS in my experience):

.. code-block:: python

  from mbchl.data.datasets import RemoteAudioDataset

  dataset = RemoteAudioDataset(
      url="ssh://user@host:path/to/clarity-{:02d}.tar",
      n_archives=16,
  )

  for x, fs, worker_state in dataset:
      pass  # do stuff

A :class:`~mbchl.data.datasets.RemoteAudioDataset` instance can be wrapped around a :class:`~mbchl.data.dataloader.AudioDataLoader` instance to perform batching and leverage multiple workers:

.. code-block:: python

  from mbchl.data.dataloader import AudioDataLoader
  from mbchl.data.datasets import RemoteAudioDataset

  dataset = RemoteAudioDataset(
      url="data/shards/clarity-{:02d}.tar",
      n_archives=16,
  )

  dataloader = AudioDataLoader(
      dataset,
      batch_size=32,
      num_workers=4,
      persistent_workers=True,
      buffer_size=4,
  )

  for (x, fs), length in dataloader:
      pass  # do stuff


Dynamic mixing from remote datasets
-----------------------------------

The :class:`~mbchl.data.datasets.DynamicAudioDataset` supports remote datasets. It wraps two :class:`~mbchl.data.datasets.RemoteAudioDataset` instances, one for the clean speech and one for the noise segments, and creates mixtures on-the-fly. For example:

.. code-block:: python

  from mbchl.data.dataloader import AudioDataLoader
  from mbchl.data.datasets import DynamicAudioDataset

  dataset = DynamicAudioDataset(
      length=200,
      fs=16000,
      speech_dataset="remote",
      speech_dataset_kw={
          "url": "data/shards/clarity-{:02d}.tar",
          "n_archives": 16,
          "loop": True,
          "tensor": False,
      },
      noise_dataset="remote",
      noise_dataset_kw={
          "url": "data/shards/tut-{:02d}.tar",
          "n_archives": 16,
          "loop": True,
          "tensor": False,
      },
      segment_length=4.0,
  )

  dataloader = AudioDataLoader(
      dataset,
      batch_size=32,
      num_workers=4,
      persistent_workers=True,
      buffer_size=4,
      batch_mix=True,
  )

  for (noisy, clean), length in dataloader:
      pass  # do stuff


Data shuffling
--------------

Even though global shuffling at the start of each epoch is not supported when streaming remote datasets, shuffling still happens on different levels:

* The files or segments are globally shuffled once when the shards are created with the ``-S`` option in the ``scripts/split_tar.py`` script.
* The order in which each worker reads its assigned set of shards is randomized.
* Files read by :class:`~mbchl.data.datasets.RemoteAudioDataset` can be placed in a buffer and shuffled before yielding. This is enabled via the ``buffer_size`` option of :class:`~mbchl.data.datasets.RemoteAudioDataset`.
* Batches created by :class:`~mbchl.data.dataloader.AudioDataLoader` can be placed in a buffer and shuffled before yielding. This is enabled via the ``buffer_size`` option of :class:`~mbchl.data.dataloader.AudioDataLoader`, and is recommended in conjunction with ``batch_mix=True`` to ensure each batch contains items from different workers (and thus different shards).

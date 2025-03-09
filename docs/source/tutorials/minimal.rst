Minimal working code examples
=============================

This page provides minimal working code examples for training systems for speech enhancement, or joint speech enhancement and hearing loss compensation. Normally, one would use the ``train.py`` script or the :class:`~mbchl.training.trainer.AudioTrainer` class, which handles logging, validation, checkpoint saving, etc... But if you would rather write training loops from scratch, the following examples might be helpful.

Training a system for speech enhancement
----------------------------------------

The following code uses the provided :class:`~mbchl.data.datasets.AudioDataset` and :class:`~mbchl.data.dataloader.AudioDataLoader` classes to load batches of noisy and clean speech files in pairs, and train a :class:`~mbchl.has.BSRNNHA` hearing aid instance.

.. code-block:: python

  from mbchl.data.dataloader import AudioDataLoader
  from mbchl.data.datasets import AudioDataset
  from mbchl.has import BSRNNHA

  noisy_dir = "path/to/noisy/dir/"
  clean_dir = "path/to/clean/dir/"

  model = BSRNNHA(loss="snr", optimizer="Adam")

  dataset = AudioDataset([noisy_dir, clean_dir], transform=model.transform)
  dataloader = AudioDataLoader(dataset, batch_size=4)

  for epoch in range(100):
      for batch, lengths in dataloader:
          model.train_step(batch, lengths)

The PyTorch optimization routine happens inside the :meth:`~mbchl.has.base.BaseHA.train_step` method of the hearing aid. The :meth:`~mbchl.has.base.BaseHA.transform` method of the hearing aid is passed to the dataset and defines data pre-processing steps performed by workers on CPU before batching. The ``lengths`` tensor contains the original length of the transformed noisy and clean speech before they were padded to the same length to create a batch.

If this is is still too abstract for you, the code below should be equivalent:

.. code-block:: python

  from torch.optim import Adam

  from mbchl.data.dataloader import AudioDataLoader
  from mbchl.data.datasets import AudioDataset
  from mbchl.nets import BSRNN
  from mbchl.signal.stft import STFT
  from mbchl.training.losses import SNRLoss

  noisy_dir = "path/to/noisy/dir/"
  clean_dir = "path/to/clean/dir/"

  stft = STFT()
  model = BSRNN()
  loss = SNRLoss()
  optimizer = Adam(model.parameters(), lr=1e-3)

  dataset = AudioDataset([noisy_dir, clean_dir])
  dataloader = AudioDataLoader(dataset, batch_size=4)

  for epoch in range(100):
      for (noisy, clean), lengths in dataloader:
          optimizer.zero_grad()
          noisy = stft(noisy)
          enhanced = model(noisy)
          enhanced = stft.inverse(enhanced, length=clean.shape[-1])
          loss_ = loss(enhanced, clean, lengths[:, -1])
          loss_.mean().backward()
          optimizer.step()

Key differences include:

* The short-time Fourier transform (STFT), raw neural network, loss and optimizer instances are manually created. Before, this was all handled inside the hearing aid constructor.
* The ``transform`` option is not passed to the dataset. Instead, the STFT is manually applied to the noisy speech before feeding it to the model.
* The signal-to-noise ratio (SNR) loss is calculated in the time domain, so the inverse STFT is applied to the enhanced speech before calculating the loss.

Training a system for joint speech enhancement and hearing loss compensation
----------------------------------------------------------------------------

The code below adapts the previous code to perform joint speech enhancement and hearing loss compensation. The ``emb_dim`` option is passed to the raw neural network constructor to enable embedding layers that can process the audiogram (the embedding size is 10 frequencies + 10 thresholds = 20). An auditory model-based loss :class:`~mbchl.training.losses.AuditoryLoss` is used. For each training iteration, a random audiogram is generated and fed to the hearing aid a loss instances.

.. code-block:: python

  from torch.optim import Adam

  from mbchl.data.dataloader import AudioDataLoader
  from mbchl.data.datasets import AudioDataset
  from mbchl.nets import BSRNN
  from mbchl.signal.stft import STFT
  from mbchl.training.losses import AuditoryLoss
  from mbchl.utils import random_audiogram

  noisy_dir = "path/to/noisy/dir/"
  clean_dir = "path/to/clean/dir/"

  stft = STFT()
  model = BSRNN(emb_dim=20)
  loss = AuditoryLoss()
  optimizer = Adam(model.parameters(), lr=1e-3)

  batch_size = 4

  dataset = AudioDataset([noisy_dir, clean_dir])
  dataloader = AudioDataLoader(dataset, batch_size=batch_size)

  for epoch in range(100):
      for (noisy, clean), lengths in dataloader:
          optimizer.zero_grad()
          noisy = stft(noisy)
          audiogram = random_audiogram(batch_size=batch_size, tensor=True)
          enhanced = model(noisy, emb=audiogram.reshape(batch_size, -1))
          enhanced = stft.inverse(enhanced, length=clean.shape[-1])
          loss_ = loss(enhanced, clean, lengths[:, -1], audiogram=audiogram)
          loss_.mean().backward()
          optimizer.step()

The next code uses the :meth:`~mbchl.has.base.BaseHA.train_step` method to add abstraction. Additionally, the :class:`~mbchl.data.datasets.DynamicAudioDataset` class is used to generate mixtures on-the-fly from clean speech utterances and noise segments. Passing ``audiogram=True`` allows ``batch`` to directly contain the generated random audiograms.

.. code-block:: python

  from mbchl.data.dataloader import AudioDataLoader
  from mbchl.data.datasets import DynamicAudioDataset
  from mbchl.has import BSRNNHA

  speech_archive = "path/to/speech/archive.tar"
  noise_archive = "path/to/noise/archive.tar"

  model = BSRNNHA(loss="auditory", optimizer="Adam", audiogram=True, emb_dim=20)

  dataset = DynamicAudioDataset(
      length=200,
      fs=16000,
      speech_dataset="remote",
      speech_dataset_kw={"url": speech_archive, "n_archives": 1, "tensor": False},
      noise_dataset="remote",
      noise_dataset_kw={"url": noise_archive, "n_archives": 1, "tensor": False},
      audiogram=True,
      transform=model.transform,
  )
  dataloader = AudioDataLoader(dataset, batch_size=4)

  for epoch in range(100):
      for batch, lengths in dataloader:
          model.train_step(batch, lengths)

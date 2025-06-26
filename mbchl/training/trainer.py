import io
import itertools
import logging
import operator
import os
import sys
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.distributed as dist
import wandb
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm import tqdm

from ..data.batching import BatchSamplerRegistry, DistributedBatchSamplerWrapper
from ..data.dataloader import AudioDataLoader
from ..filesystem import init_filesystem
from ..metrics import MetricRegistry
from ..utils import MathDict
from .ema import EMARegistry
from .losses import ControllableNoiseReductionHearingLossCompensationLoss, ControllableNoiseReductionHearingLossCompensationLosseFixedWeights, ControllableNoiseReductionHearingLossCompensationLosseSpeechUncertainties, ControllableNoiseReductionHearingLossCompensationLosseEnvUncertainties, ControllableNoiseReductionHearingLossCompensationLosseAllUncertainties, LossPhilippeStyleTG2, ControllableNoiseReductionHearingLossCompensationLosseSixUncertainties, ControllableNoiseReductionHearingLossCompensationLoss7, CNRHLCLossC8, CNRHLCLossC9, CNRHLCLossC10


class AudioTrainer:
    """Trainer class for training models.

    Parameters
    ----------
    model : BaseHA
        Model to train.
    train_dataset : AudioDataset
        Training dataset.
    val_dataset : AudioDataset or list[AudioDataset]
        Validation dataset(s).
    model_dirpath : str
        Model directory path. Must contain a ``config.yaml`` file. Checkpoints are saved
        in ``<model_dirpath>/checkpoints``.
    workers : int, optional
        Number of workers.
    epochs : int, optional
        Number of epochs.
    device : int or str, optional
        Device to use.
    train_batch_sampler : str, optional
        Batch sampler for training.
    train_batch_sampler_kw : dict, optional
        Keyword arguments for the training batch sampler.
    val_batch_sampler : str, optional
        Batch sampler for validation.
    val_batch_sampler_kw : dict, optional
        Keyword arguments for the validation batch sampler.
    ignore_checkpoint : bool, optional
        If ``False`` and a checkpoint is found, then training is resumed.
    preload : bool, optional
        Whether to load the training and validation datasets in memory.
    ddp : bool, optional
        Whether the application is running in distributed mode.
    rank : int, optional
        Rank of the process in distributed mode.
    use_wanb : bool, optional
        Whether to use Weights and Biases.
    profile : bool, optional
        Whether to profile training using ``torch.profiler``. Use only for short runs.
    val_metrics : dict, optional
        Metrics used for validation. Must be a dictionary of the form
        ``{metric_name: metric_kw}``, where ``metric_name`` is the name of the metric in
        the ``MetricRegistry`` and ``metric_kw`` is a dictionary of keyword arguments
        for the metric.
    val_period : int, optional
        Validation period. Validation is performed every ``val_period`` epochs. If
        ``val_period`` is ``0``, then no validation is performed.
    use_amp : bool, optional
        Whether to use automatic mixed precision.
    save_on_epochs : list[int], optional
        List of epochs indexes on which to save additional checkpoints.
    ema : str, optional
        Exponential moving average to apply to the model.
    ema_kw : dict, optional
        Keyword arguments for the exponential moving average.
    filesystem_url : str, optional
        URL of the filesystem to save checkpoints. If ``None``, the local filesystem is
        used. For an S3 bucket, use ``"s3://<bucket_name>/<path>"``.
    persistent_workers : bool, optional
        Whether to keep the workers alive between epochs.
    buffer_size : int, optional
        Buffer size for the training dataloader. See
        :class:`~mbchl.data.dataloader.AudioDataLoader`.
    batch_mix : bool, optional
        Whether the training dataloader should mix batches before yielding them. See
        :class:`~mbchl.data.dataloader.AudioDataLoader`.

    """

    def __init__(
        self,
        model,
        train_dataset,
        val_dataset,
        model_dirpath,
        workers=0,
        epochs=100,
        device="cuda",
        train_batch_sampler="random",
        train_batch_sampler_kw={
            "batch_size": 4,
            "dynamic": False,
            "fs": 16000,
            "seed": None,
        },
        val_batch_sampler="random",
        val_batch_sampler_kw={
            "batch_size": 4,
            "dynamic": False,
            "fs": 16000,
            "seed": None,
        },
        ignore_checkpoint=False,
        preload=False,
        ddp=False,
        rank=0,
        use_wandb=False,
        profile=False,
        val_metrics={},
        val_period=10,
        use_amp=False,
        save_on_epochs=[],
        ema=None,
        ema_kw=None,
        filesystem_url=None,
        persistent_workers=False,
        buffer_size=None,
        batch_mix=False,
    ):
        # set workers to 0 if preloading
        if preload and workers > 0:
            logging.warning(
                "Cannot use workers > 0 with preload=True. Forcing workers=0."
            )
            workers = 0

        # move model to device and cast to DDP
        model = model.to(device)
        if ddp:
            model = DDP(model, device_ids=[device])

        self.model = model
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.model_dirpath = model_dirpath
        self.epochs = epochs
        self.device = device
        self.ignore_checkpoint = ignore_checkpoint
        self.preload = preload
        self.rank = rank
        self.use_wandb = use_wandb
        self.profile = profile
        self.val_metrics = val_metrics
        self.val_period = val_period
        self.save_on_epochs = save_on_epochs

        self.last_ckpt_name = "last.ckpt"
        self.epochs_ran = 0
        self.max_memory_allocated = 0
        self.profiler = None

        val_datasets = val_dataset if isinstance(val_dataset, list) else [val_dataset]

        # batch samplers
        train_batch_sampler_cls = BatchSamplerRegistry.get(train_batch_sampler)
        self.train_batch_sampler = train_batch_sampler_cls(
            dataset=train_dataset,
            **train_batch_sampler_kw,
        )
        val_batch_sampler_cls = BatchSamplerRegistry.get(val_batch_sampler)
        self.val_batch_samplers = [
            val_batch_sampler_cls(
                dataset=dataset,
                **val_batch_sampler_kw,
            )
            for dataset in val_datasets
        ]

        # distributed samplers
        if dist.is_initialized():
            self.train_batch_sampler = DistributedBatchSamplerWrapper(
                self.train_batch_sampler
            )
            self.val_batch_samplers = [
                DistributedBatchSamplerWrapper(batch_sampler)
                for batch_sampler in self.val_batch_samplers
            ]

        # dataloaders
        self.train_dataloader = AudioDataLoader(
            dataset=train_dataset,
            batch_sampler=self.train_batch_sampler,
            num_workers=workers,
            persistent_workers=persistent_workers,
            buffer_size=buffer_size,
            batch_mix=batch_mix,
        )
        self.val_dataloaders = [
            # do not pass buffer_size and batch_mix to validation dataloader since
            # shuffling batches is useless during validation
            # plus validation sequences should have variable length so batch_mix is not
            # supported anyway
            AudioDataLoader(
                dataset=dataset,
                batch_sampler=batch_sampler,
                num_workers=workers,
                persistent_workers=persistent_workers,
            )
            for dataset, batch_sampler in zip(val_datasets, self.val_batch_samplers)
        ]

        # loss logger
        self.loss_logger = LossLogger(model_dirpath)

        # filesystem object
        self.filesystem = init_filesystem(filesystem_url or "")

        # checkpoint saver
        self.checkpoint_saver = CheckpointSaver(
            filesystem=self.filesystem, dirpath=self._checkpoints_dir
        )

        # timer
        self.timer = TrainingTimer(epochs, val_period)

        # automatic mixed precision scaler
        self.scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

        # exponential moving average
        if ema is not None:
            ema = EMARegistry.get(ema)(model, **(ema_kw or {}))
        self.ema = ema

    def run(self):
        """Start training.

        Produces logs, loads a checkpoint if found, preloads data if ``preload`` is
        ``True``, and starts the profiler if ``profile`` is ``True`` before starting the
        training loop.
        """

        def _rank_zero_log(msg):
            if self.rank == 0:
                logging.info(msg)

        # print number of parameters and dataset duration
        _rank_zero_log(f"Number of parameters: {self.model.count_params():_}")
        for dset, dset_name in [
            (self.train_dataset, "Training dataset"),
            *(
                [
                    (dataset, f"Validation dataset {i}")
                    for i, dataset in enumerate(self.val_dataset)
                ]
                if isinstance(self.val_dataset, list)
                else [(self.val_dataset, "Validation dataset")]
            ),
        ]:
            for duration_attr, duration_name in [
                ("_duration", "duration"),
                ("_effective_duration", "duration per epoch"),
            ]:
                if hasattr(dset, duration_attr):
                    duration = getattr(dset, duration_attr)
                else:
                    continue
                if duration == float("inf"):
                    fmt_time = "inf"
                else:
                    h, m = divmod(int(duration), 3600)
                    m, s = divmod(m, 60)
                    fmt_time = f"{h} h {m} m {s} s"
                _rank_zero_log(f"{dset_name} {duration_name}: {fmt_time}")

        # check for a checkpoint
        checkpoint_loaded = False
        if not self.ignore_checkpoint and self.filesystem.exists(self._last_ckpt_path):
            _rank_zero_log("Checkpoint found")
            if isinstance(self.device, int):
                map_location = f"cuda:{self.device}"
            else:
                map_location = self.device
            buffer = self.filesystem.get(self._last_ckpt_path)
            state = torch.load(buffer, map_location=map_location, weights_only=True)
            self.load_state_dict(state)
            # if training was interrupted then resume training
            if self.epochs_ran < self.epochs:
                _rank_zero_log(f"Resuming training at epoch {self.epochs_ran}")
            else:
                _rank_zero_log("Model is already trained")
                return
            checkpoint_loaded = True

        # preload data
        if self.preload:
            _rank_zero_log("Preloading data")
            self.train_dataset.preload(self.device, tqdm_desc="train")
            if isinstance(self.val_dataset, list):
                for i, dataset in enumerate(self.val_dataset):
                    dataset.preload(self.device, tqdm_desc=f"val {i}")
            else:
                self.val_dataset.preload(self.device, tqdm_desc="  val")

        # pre-training instructions
        if not checkpoint_loaded:
            _rank_zero_log("Pre-training model instructions")
            self._get_model().pre_train(
                self.train_dataset, self.train_dataloader, self.epochs
            )

        # start profiler
        if self.profile:

            def trace_handler(profiler):
                _rank_zero_log(
                    "\n"
                    + profiler.key_averages().table(
                        sort_by="self_cuda_time_total", row_limit=-1
                    )
                )

            _rank_zero_log("Starting profiler")
            self.profiler = torch.profiler.profile(
                activities=[
                    torch.profiler.ProfilerActivity.CPU,
                    torch.profiler.ProfilerActivity.CUDA,
                ],
                schedule=torch.profiler.schedule(
                    wait=1,
                    warmup=1,
                    active=2,
                    repeat=1,
                ),
                on_trace_ready=trace_handler,
            )
            self.profiler.start()

        # start training loop
        try:
            _rank_zero_log("Starting training loop")
            self.training_loop()
        except Exception:
            raise
        finally:
            if self.profiler is not None:
                self.profiler.stop()

    def training_loop(self):
        """Start the training loop."""
        if self.rank == 0:
            self.timer.start()
        for epoch in range(self.epochs_ran, self.epochs):
            self.train_dataloader.set_epoch(epoch)
            # TODO: for the val datasets set_epoch should not change the yielded items!
            for dataloader in self.val_dataloaders:
                dataloader.set_epoch(epoch)
            # train
            train_loss = self.routine(epoch, train=True)
            self._get_model().on_train()
            # evaluate
            validate = self.val_period != 0 and (epoch + 1) % self.val_period == 0
            if validate:
                val_loss, val_metrics = {}, {}
                for i in range(len((self.val_dataloaders))):
                    with torch.no_grad():
                        val_loss_, val_metrics_ = self.routine(
                            epoch,
                            train=False,
                            val_idx=i,
                        )
                    if i == 0:
                        # TODO: on_validate is called only after the first val dataset
                        self._get_model().on_validate(val_loss_)
                    val_loss.update({f"{k}_{i}": v for k, v in val_loss_.items()})
                    val_metrics.update({f"{k}_{i}": v for k, v in val_metrics_.items()})
            else:
                val_loss, val_metrics = {}, {}
            # ddp reduce
            if dist.is_initialized():
                # TODO: reduce expects tensors but val_metrics can have numpy scalars!
                self._reduce(train_loss, val_loss, val_metrics)
            # log and save best model
            self.epochs_ran += 1
            if self.rank == 0:
                self.loss_logger.add(epoch, train_loss, val_loss, val_metrics)
                self.loss_logger.log(epoch)
                if self.use_wandb:
                    self._wandb_log(train_loss, val_loss, val_metrics)
                # save best model
                self.checkpoint_saver(epoch, val_loss, val_metrics, self.state_dict())
                # save last model
                self.save_checkpoint()
                # save additional checkpoints
                if epoch in self.save_on_epochs:
                    self.save_checkpoint(f"epoch={epoch}.ckpt")
            # ddp sync
            if dist.is_initialized():
                dist.barrier()
        # plot and save losses
        if self.rank == 0:
            self.timer.final_log()
            self.loss_logger.plot_and_save()

    def routine(self, epoch, train=True, val_idx=None):
        """Single training or validation epoch.

        Parameters
        ----------
        epoch : int
            Epoch index.
        train : bool, optional
            Whether to train or validate.
        val_idx : int, optional
            Index of the validation dataset to use. Must be provided when
            ``train`` is ``False``.

        Returns
        -------
        loss : dict[str, torch.Tensor]
            Losses averaged over the epoch. Values are 0-dimensional tensors.
        metrics : dict[str, torch.Tensor or np.generic]
            Metrics averaged over the epoch. Returned only when ``train`` is ``False``.
            Values are 0-dimensional tensors or NumPy scalars.

        """
        # TODO: make all metrics return consistent output type
        if train:
            self.model.train()
            dataloader = self.train_dataloader
        else:
            self.model.eval()
            dataloader = self.val_dataloaders[val_idx]
            if dataloader is None:
                raise ValueError("dataloader must be provided for validation")
            if val_idx is None:
                raise ValueError("val_idx must be provided for validation")
            if self.ema is not None:
                self.ema.store()
                self.ema.apply()
            avg_metrics = MathDict()
        avg_loss = MathDict()
        tqdm_desc = (
            f"epoch {epoch}, "
            + (
                "train"
                if train
                else (
                    f"val {val_idx}" if isinstance(self.val_dataset, list) else "  val"
                )
            )
            + (f", rank {self.rank}" if dist.is_initialized() else "")
        )
        for batch, lengths in tqdm(
            dataloader, file=sys.stdout, desc=tqdm_desc, position=self.rank
        ):
            if isinstance(batch, tuple):
                batch = tuple(x.to(self.device) for x in batch)
            else:
                batch = batch.to(self.device)
            lengths = lengths.to(self.device)
            model = self._get_model()
            use_amp = self.scaler.is_enabled()
            if train:
                loss = model.train_step(batch, lengths, use_amp, self.scaler)
                assert not loss.isnan(), "loss is nan"
                if self.ema is not None:
                    self.ema.update()
            else:
                # the validation dataset yields raw waveforms; manually apply the model
                # transform and recreate the batch to calculate calculate the val loss
                if isinstance(batch, tuple):
                    transformed, trans_lengths = AudioDataLoader.collate_fn(
                        [
                            model.transform(
                                tuple(x_[..., :l__] for x_, l__ in zip(x, l_))
                            )
                            for x, l_ in zip(zip(*batch), lengths)
                        ],
                        _tensors_only=True,
                    )
                else:
                    transformed, trans_lengths = AudioDataLoader._collate_fn(
                        [model.transform(x[..., :l_]) for x, l_ in zip(batch, lengths)]
                    )
                loss = model.val_step(transformed, trans_lengths, use_amp)
                # finally compute metrics
                metrics = self.compute_metrics(batch, lengths, use_amp)
                avg_metrics += metrics
            if isinstance(loss, torch.Tensor):
                loss = {"loss": loss}
            elif not isinstance(loss, dict):
                raise ValueError(
                    "train_step and val_step must return a tensor or a "
                    f"dict, got {loss.__class__.__name__}"
                )
            loss = {k: v.detach() for k, v in loss.items()}
            avg_loss += loss
            if self.profiler is not None:
                self.profiler.step()
        avg_loss /= len(dataloader)
        if train:
            output = avg_loss
        else:
            if self.ema is not None:
                self.ema.restore()
            avg_metrics /= len(dataloader)
            output = avg_loss, avg_metrics
        if dist.is_initialized():
            dist.barrier()
        # update time spent
        if self.rank == 0:
            if train or val_idx == len(self.val_dataloaders) - 1:
                self.timer.step(is_validation_step=not train)
            self.timer.log()
        return output

    def compute_metrics(self, batch, lengths, use_amp):
        """Calculate validation metrics on a batch.

        Parameters
        ----------
        batch : tuple[torch.Tensor, ...]
            Input batch. Each item has shape ``(batch, ...)``.
        lengths : torch.Tensor
            Input lengths along the last dimension before batching. Shape ``(batch_size,
            len(batch))``.
        use_amp : bool
            Whether to use automatic mixed precision.

        Returns
        -------
        metrics : dict[str, torch.Tensor or np.generic]
            Metrics averaged over the batch. Values are 0-dimensional tensors or NumPy
            scalars.

        """
        if not self.val_metrics:
            return {}
        input_, target, extra_inputs = batch[0], batch[1], batch[2:]
        output = self._get_model().enhance(
            input_,
            use_amp=use_amp,
            extra_inputs=extra_inputs,
        )
        if not self._get_model()._labels_include_clean:
            target = torch.tensor([], device=target.device, dtype=target.dtype)
        if self._get_model()._labels_include_noisy:
            target = torch.cat([input_, target], dim=1)
        # output and target have shape (batch_size, output_channels, time)
        # reshape to calculate score in each channel and average
        lengths = lengths[:, 1].tile(target.shape[1], 1).T.flatten()
        output = output.reshape(-1, output.shape[-1])
        target = target.reshape(-1, target.shape[-1])
        metrics = {}
        for metric_name, metric_kw in self.val_metrics.items():
            metric_cls = MetricRegistry.get(metric_name)
            metric_obj = metric_cls(**(metric_kw or {}))
            metric_values = metric_obj(output, target, lengths=lengths)
            metrics[metric_name] = metric_values.mean()
        return metrics

    def save_checkpoint(self, filename=None):
        """Save a checkpoint.

        Parameters
        ----------
        filename : str, optional
            Filename for the checkpoint. If ``None``, the checkpoint is saved as
            ``last.ckpt``.

        """
        if filename is None:
            filename = self.last_ckpt_name
        path = self.filesystem.join(self._checkpoints_dir, filename)
        self.filesystem.makedirs(self._checkpoints_dir)
        state = self.state_dict()
        with io.BytesIO() as buffer:
            torch.save(state, buffer)
            buffer.seek(0)
            self.filesystem.put(buffer, path)

    def state_dict(self):
        """Return a state dictionary."""
        state = {
            "epochs": self.epochs_ran,
            "model": self._get_model().state_dict(),
            "scaler": self.scaler.state_dict(),
            "losses": {
                "train": self.loss_logger.train_loss,
                "val": self.loss_logger.val_loss,
            },
            "max_memory_allocated": max(
                torch.cuda.max_memory_allocated(),
                self.max_memory_allocated,
            ),
            "timer": self.timer.state_dict(),
            "best_ckpts": self.checkpoint_saver.best,
            "train_dataloader": self.train_dataloader.state_dict(),
            "val_dataloaders": [
                dataloader.state_dict() for dataloader in self.val_dataloaders
            ],
        }
        if self.ema is not None:
            state["ema"] = self.ema.state_dict()
        return state

    def load_state_dict(self, state):
        """Load a state dictionary."""
        self._get_model().load_state_dict(state["model"])
        self.scaler.load_state_dict(state["scaler"])
        self.loss_logger.train_loss = state["losses"]["train"]
        self.loss_logger.val_loss = state["losses"]["val"]
        self.epochs_ran = state["epochs"]
        self.max_memory_allocated = state["max_memory_allocated"]
        self.timer.load_state_dict(state["timer"])
        self.checkpoint_saver.best = state["best_ckpts"]
        self.train_dataloader.load_state_dict(state["train_dataloader"])
        for dataloader, dataloader_state in zip(
            self.val_dataloaders, state["val_dataloaders"]
        ):
            dataloader.load_state_dict(dataloader_state)
        if self.ema is not None:
            self.ema.load_state_dict(state["ema"])
        elif "ema" in state.keys():
            raise ValueError("found 'ema' in state dict but EMA is not enabled")

    def _get_model(self):
        if isinstance(self.model, DDP):
            model = self.model.module
        else:
            model = self.model
        return model

    @staticmethod
    def _reduce(*tensor_dicts):
        for tensor_dict in tensor_dicts:
            for key, tensor in tensor_dict.items():
                dist.reduce(tensor, 0)
                tensor /= dist.get_world_size()

    def _wandb_log(self, train_loss, val_loss, val_metrics):
        log = {
            "train": train_loss,
            "val": {
                **val_loss,
                "metrics": val_metrics,
            },
        }
        # TODO: find a cleaner way to log the uncertainties
        # do not log them inside the loss compute method since calling item() at each
        # training iteration is expensive
        # logging them here has the advantage of doing it only once per epoch
        _loss = self._get_model()._loss
        if isinstance(_loss, ControllableNoiseReductionHearingLossCompensationLoss):
            log["uncertainty_denoising"] = _loss.log_uncertainty_denoising.item()
            log["uncertainty_compensation"] = _loss.log_uncertainty_compensation.item()

            log['denoising_loss'] = _loss.denoising_loss.detach().mean().item()
            log['compensation_loss'] = _loss.compensation_loss.detach().mean().item()
        elif isinstance(_loss, ControllableNoiseReductionHearingLossCompensationLosseFixedWeights):
            log["uncertainty_denoising"] = _loss.log_uncertainty_denoising.item()
            log["uncertainty_compensation"] = _loss.log_uncertainty_compensation.item()

            log['denoising_loss'] = _loss.denoising_loss.detach().mean().item()
            log['compensation_loss'] = _loss.compensation_loss_base.detach().mean().item()
            log['augmented_compensation_loss'] = _loss.augmented_compensation_loss.detach().mean().item()
            log['modulation_speech_loss'] = _loss.modulation_speech_loss.detach().mean().item()
            log['modulation_env_loss'] = _loss.modulation_env_loss.mean().item()
        elif isinstance(_loss, ControllableNoiseReductionHearingLossCompensationLosseSpeechUncertainties):
            log["uncertainty_denoising"] = _loss.log_uncertainty_denoising.item()
            log["uncertainty_compensation"] = _loss.log_uncertainty_compensation.item()
            log["uncertainty_modulation_speech"] = _loss.log_uncertainty_modulation_speech.item() 

            log['denoising_loss'] = _loss.denoising_loss.detach().mean().item()
            log['compensation_loss'] = _loss.compensation_loss.detach().mean().item()
            log['modulation_speech_loss'] = _loss.modulation_speech_loss.detach().mean().item()
        elif isinstance(_loss, ControllableNoiseReductionHearingLossCompensationLosseEnvUncertainties):
            log["uncertainty_denoising"] = _loss.log_uncertainty_denoising.item()
            log["uncertainty_compensation"] = _loss.log_uncertainty_compensation.item()
            log["uncertainty_modulation_env"] = _loss.log_uncertainty_modulation_env.item()

            log['denoising_loss'] = _loss.denoising_loss.detach().mean().item()
            log['compensation_loss'] = _loss.compensation_loss.detach().mean().item()
            log['modulation_env_loss'] = _loss.modulation_env_loss.detach().mean().item()
        elif isinstance(_loss, ControllableNoiseReductionHearingLossCompensationLosseAllUncertainties):
            log["uncertainty_denoising"] = _loss.log_uncertainty_denoising.item()
            log["uncertainty_compensation"] = _loss.log_uncertainty_compensation.item()
            log["uncertainty_modulation_speech"] = _loss.log_uncertainty_modulation_speech.item() 
            log["uncertainty_modulation_env"] = _loss.log_uncertainty_modulation_env.item()

            log['denoising_loss'] = _loss.denoising_loss.detach().mean().item()
            log['compensation_loss'] = _loss.compensation_loss.detach().mean().item()
            log['modulation_speech_loss'] = _loss.modulation_speech_loss.detach().mean().item()
            log['modulation_env_loss'] = _loss.modulation_env_loss.detach().mean().item()
        elif isinstance(_loss, LossPhilippeStyleTG2):
            log["log_uncertainty_nr_speech"] = _loss.log_uncertainty_nr_speech.item()
            log["uncertainty_compensation"] = _loss.log_uncertainty_nr_env.item()
            log["uncertainty_modulation_speech"] = _loss.log_uncertainty_hlc_speech.item() 
            log["uncertainty_modulation_env"] = _loss.log_uncertainty_hlc_env.item()

            log['loss_nr_speech'] = _loss.loss_nr_speech.detach().mean().item()
            log['loss_nr_env'] = _loss.loss_nr_env.detach().mean().item()
            log['loss_hlc_speech'] = _loss.loss_hlc_speech.detach().mean().item()
            log['loss_hlc_env'] = _loss.loss_hlc_env.detach().mean().item()
        elif isinstance(_loss, ControllableNoiseReductionHearingLossCompensationLosseSixUncertainties):
            log["uncertainty_denoising"] = _loss.log_uncertainty_denoising.item()
            log["uncertainty_compensation"] = _loss.log_uncertainty_compensation.item()
            log["uncertainty_nr_speech"] = _loss.log_uncertainty_nr_speech.item() 
            log["uncertainty_nr_env"] = _loss.log_uncertainty_nr_env.item()
            log["uncertainty_hlc_speech"] = _loss.log_uncertainty_hlc_speech.item()
            log["uncertainty_hlc_env"] = _loss.log_uncertainty_hlc_env.item()

            log['denoising_loss'] = _loss.denoising_loss.detach().mean().item()
            log['compensation_loss'] = _loss.compensation_loss.detach().mean().item()
            log['nr_speech_loss'] = _loss.loss_nr_speech.detach().mean().item()
            log['nr_env_loss'] = _loss.loss_nr_env.detach().mean().item()
            log['hlc_speech_loss'] = _loss.loss_hlc_speech.detach().mean().item()
            log['hlc_env_loss'] = _loss.loss_hlc_env.detach().mean().item() # 
        elif isinstance(_loss, ControllableNoiseReductionHearingLossCompensationLoss7):
            log['denoising_loss'] = _loss.denoising_loss.detach().mean().item()
            log['compensation_loss'] = _loss.compensation_loss.detach().mean().item()
            log['modulation_speech_loss'] = _loss.modulation_speech_loss.detach().mean().item()
            log['modulation_env_loss'] = _loss.modulation_env_loss.detach().mean().item()

            log['uncertainty_denoising'] = _loss.log_uncertainty_denoising.item()
            log['uncertainty_compensation'] = _loss.log_uncertainty_compensation.item()
            log['uncertainty_modulation_speech'] = _loss.log_uncertainty_modulation_speech.item()
            log['uncertainty_modulation_env'] = _loss.log_uncertainty_modulation_env.item()
        elif isinstance(_loss, CNRHLCLossC8):
            log['denoising_loss'] = _loss.denoising_loss.detach().mean().item()
            log['hlc_speech_loss'] = _loss.loss_hlc_speech.detach().mean().item()
            log['hlc_env_loss'] = _loss.loss_hlc_env.detach().mean().item()
            log['uncertainty_denoising'] = _loss.log_uncertainty_denoising.item()
            log['uncertainty_hlc_speech'] = _loss.log_uncertainty_hlc_speech.item()
            log['uncertainty_hlc_env'] = _loss.log_uncertainty_hlc_env.item()
        elif isinstance(_loss, CNRHLCLossC9):
            log['denoising_loss'] = _loss.compensation_loss.detach().mean().item()
            log['hlc_speech_loss'] = _loss.loss_nr_speech.detach().mean().item()
            log['hlc_env_loss'] = _loss.loss_nr_env.detach().mean().item()
            log['uncertainty_denoising'] = _loss.log_uncertainty_compensation.item()
            log['uncertainty_hlc_speech'] = _loss.log_uncertainty_nr_speech.item()
            log['uncertainty_hlc_env'] = _loss.log_uncertainty_nr_env.item()
        elif isinstance(_loss, CNRHLCLossC10):
            log['denoising_loss'] = _loss.denoising_loss.detach().mean().item()
            log['hlc_speech_loss'] = _loss.compensation_loss.detach().mean().item()
            log['uncertainty_denoising'] = _loss.log_uncertainty_compensation.item()
            log['uncertainty_hlc_speech'] = _loss.log_uncertainty_denoising.item()
            # # 不确定性指标组
            # log["uncertainties"] = {
            #     "denoising": _loss.log_uncertainty_denoising.item(),
            #     "compensation": _loss.log_uncertainty_compensation.item(),
            #     "modulation_speech": _loss.log_uncertainty_modulation_speech.item() if hasattr(_loss, 'log_uncertainty_modulation_speech') else None,
            #     "modulation_env": _loss.log_uncertainty_modulation_env.item() if hasattr(_loss, 'log_uncertainty_modulation_env') else None
            # }

            # # 损失指标组
            # log["losses"] = {
            #     "denoising": _loss.denoising_loss.item(),
            #     "compensation": _loss.compensation_loss.item(),
            #     "modulation_speech": _loss.modulation_speech_loss.item() if hasattr(_loss, 'modulation_speech_loss') else None,
            #     "modulation_env": _loss.modulation_env_loss.item() if hasattr(_loss, 'modulation_env_loss') else None
            # }

            # # 添加学习率监控
            # if hasattr(self.optimizer, 'param_groups'):
            #     for i, param_group in enumerate(self.optimizer.param_groups):
            #         log[f"learning_rate_group_{i}"] = param_group['lr']

            # # 添加模型参数统计
            # model = self._get_model()
            # for name, param in model.named_parameters():
            #     if param.requires_grad:
            #         log[f"param_stats/{name}/mean"] = param.data.mean().item()
            #         log[f"param_stats/{name}/std"] = param.data.std().item()
            #         if param.grad is not None:
            #             log[f"grad_stats/{name}/mean"] = param.grad.mean().item()
            #             log[f"grad_stats/{name}/std"] = param.grad.std().item()

        wandb.log(log)

    @property
    def _checkpoints_dir(self):
        return self.filesystem.join(self.model_dirpath, "checkpoints")

    @property
    def _last_ckpt_path(self):
        return self.filesystem.join(self._checkpoints_dir, self.last_ckpt_name)


class TrainingTimer:
    """Training time of arrival estimator.

    Parameters
    ----------
    epochs : int
        Total number of epochs.
    val_period : int
        Validation period.

    """

    def __init__(self, epochs, val_period):
        self.epochs = epochs
        self.val_period = val_period
        self.train_steps_taken = 0
        self.val_steps_taken = 0
        self.train_steps_measured = 0
        self.val_steps_measured = 0
        self.avg_train_duration = None
        self.avg_val_duration = None
        self.start_time = None
        self.step_start_time = None
        self.resume_offset = 0
        self.first_session_step = True

        if val_period == 0:
            self.avg_val_duration = 0

    def load_state_dict(self, state):
        """Load a state dictionary."""
        self.train_steps_taken = state["train_steps_taken"]
        self.val_steps_taken = state["val_steps_taken"]
        self.train_steps_measured = state["train_steps_measured"]
        self.val_steps_measured = state["val_steps_measured"]
        self.avg_train_duration = state["avg_train_duration"]
        self.avg_val_duration = state["avg_val_duration"]
        self.resume_offset = state["resume_offset"]

    def state_dict(self):
        """Return a state dictionary."""
        return dict(
            train_steps_taken=self.train_steps_taken,
            val_steps_taken=self.val_steps_taken,
            train_steps_measured=self.train_steps_measured,
            val_steps_measured=self.val_steps_measured,
            avg_train_duration=self.avg_train_duration,
            avg_val_duration=self.avg_val_duration,
            resume_offset=self.total_elapsed_time,
        )

    def start(self):
        """Start the timer."""
        start_time = time.time()
        self.start_time = start_time
        self.step_start_time = start_time

    def step(self, is_validation_step=False):
        """Update the timer after a training or a validation epoch."""
        step_end_time = time.time()
        step_duration = step_end_time - self.step_start_time
        if is_validation_step:
            if not self.first_session_step:
                self._update_avg_val_duration(step_duration)
            self.val_steps_taken += 1
        else:
            if not self.first_session_step:
                self._update_avg_train_duration(step_duration)
            self.train_steps_taken += 1
        self.first_session_step = False
        self.step_start_time = step_end_time

    def log(self):
        """Display the estimated time left."""
        logging.info(
            ", ".join(
                [
                    f"Avg train time: {self._fmt_time(self.avg_train_duration)}",
                    f"Avg val time: {self._fmt_time(self.avg_val_duration)}",
                    f"ETA: {self._fmt_time(self.estimated_time_left)}",
                ]
            )
        )

    def final_log(self):
        """Display the total time spent."""
        total_time = self.total_elapsed_time
        logging.info(
            f"Time spent: {int(total_time/3600)} h "
            f"{int(total_time%3600/60)} m {int(total_time%60)} s"
        )

    @property
    def total_elapsed_time(self):
        """Total time spent."""
        return self.session_elapsed_time + self.resume_offset

    @property
    def session_elapsed_time(self):
        """Time spent since resuming."""
        return time.time() - self.start_time

    @property
    def train_steps(self):
        """Total number of training steps."""
        return self.epochs

    @property
    def val_steps(self):
        """Total number of validation steps."""
        return 0 if self.val_period == 0 else self.epochs // self.val_period

    @property
    def train_steps_left(self):
        """Number of training steps left."""
        return self.train_steps - self.train_steps_taken

    @property
    def val_steps_left(self):
        """Number of validation steps left."""
        return self.val_steps - self.val_steps_taken

    @property
    def estimated_time_left(self):
        """Estimated time left."""
        if self.avg_train_duration is None or self.avg_val_duration is None:
            return None
        else:
            return (
                self.avg_train_duration * self.train_steps_left
                + self.avg_val_duration * self.val_steps_left
            )

    def _update_avg_train_duration(self, duration):
        if self.avg_train_duration is None:
            self.avg_train_duration = duration
        else:
            self.avg_train_duration = (
                self.avg_train_duration * self.train_steps_measured + duration
            ) / (self.train_steps_measured + 1)
        self.train_steps_measured += 1

    def _update_avg_val_duration(self, duration):
        if self.avg_val_duration is None:
            self.avg_val_duration = duration
        else:
            self.avg_val_duration = (
                self.avg_val_duration * self.val_steps_measured + duration
            ) / (self.val_steps_measured + 1)
        self.val_steps_measured += 1

    @staticmethod
    def _fmt_time(time):
        if time is None:
            return "--"
        h, m, s = int(time // 3600), int((time % 3600) // 60), int(time % 60)
        output = f"{s} s"
        if time >= 60:
            output = f"{m} m {output}"
        if time >= 3600:
            output = f"{h} h {output}"
        return output


class LossLogger:
    """Loss and metric logger.

    Parameters
    ----------
    dirpath : str
        Director where to save losses and plots.

    """

    def __init__(self, dirpath):
        self.train_loss = {}
        self.val_loss = {}
        self.val_metrics = {}
        self.dirpath = dirpath

    def add(self, epoch, train_loss, val_loss, val_metrics):
        """Add losses and metrics to the logger.

        Parameters
        ----------
        epoch : int
            Current epoch.
        train_loss : dict[str, torch.Tensor]
            Training losses averaged over the epoch. Values are 0-dimensional tensors.
        val_loss : dict[str, torch.Tensor]
            Validation losses averaged over the epoch. Values are 0-dimensional tensors.
        val_metrics : dict[str, torch.Tensor or np.generic]
            Validation metrics averaged over the epoch.  Values are 0-dimensional
            tensors or NumPy scalars.

        """
        for self_loss, new_loss in [
            (self.train_loss, train_loss),
            (self.val_loss, val_loss),
            (self.val_metrics, val_metrics),
        ]:
            for key, item in new_loss.items():
                if key not in self_loss:
                    self_loss[key] = []
                self_loss[key].append((epoch, item))

    def log(self, epoch):
        """Log the losses.

        Parameters
        ----------
        epoch : int
            Current epoch.

        """
        logging.info(
            f"Epoch {epoch}: "
            + ", ".join(
                itertools.chain(
                    (f"train_{k}: {v[-1][1]:.2e}" for k, v in self.train_loss.items()),
                    (f"val_{k}: {v[-1][1]:.2e}" for k, v in self.val_loss.items()),
                    (f"val_{k}: {v[-1][1]:.2e}" for k, v in self.val_metrics.items()),
                )
            )
        )

    def plot_and_save(self):
        """Plot and save the losses when training is finished."""
        losses = self._to_numpy_dict()
        self._plot(losses)
        self._save(losses)

    def _to_numpy_dict(self):
        output = dict()
        for loss, tag in [
            (self.train_loss, "train"),
            (self.val_loss, "val"),
            (self.val_metrics, "metrics"),
        ]:
            for k, v in loss.items():
                out_key = f"{tag}_{k}"
                output[out_key] = torch.tensor(v).cpu().numpy()
        return output

    def _plot(self, losses):
        plt.rc("axes", facecolor="#E6E6E6", edgecolor="none", axisbelow=True)
        plt.rc("grid", color="w", linestyle="solid")
        fig, ax = plt.subplots()
        for k, v in losses.items():
            if not k.startswith("metrics"):
                ax.plot(v[:, 0], v[:, 1], label=k)
        ax.legend()
        ax.set_xlabel("epoch")
        ax.set_ylabel("error")
        ax.grid(True)
        plot_output_path = os.path.join(self.dirpath, "training_curve.png")
        fig.tight_layout()
        fig.savefig(plot_output_path)
        plt.close(fig)

    def _save(self, losses):
        loss_path = os.path.join(self.dirpath, "losses.npz")
        np.savez(loss_path, **losses)


class CheckpointSaver:
    """Checkpoint saver.

    Parameters
    ----------
    dirpath : str
        Directory where to save checkpoints.

    """

    def __init__(self, filesystem, dirpath):
        self.filesystem = filesystem
        self.dirpath = dirpath
        self.best = {}

    def __call__(self, epoch, loss, metrics, state):
        """Save the model if loss or metrics are better than the previous best.

        To be used as a callback in the training loop.

        Parameters
        ----------
        epoch : int
            Current epoch.
        loss : dict[str, torch.Tensor]
            Validation losses averaged over the epoch. Values are 0-dimensional tensors.
        metrics : dict[str, torch.Tensor or np.generic]
            Validation metrics averaged over the epoch. Values are 0-dimensional tensors
            or NumPy scalars.
        state : dict
            State dictionary of the model to save.

        """
        for d, op in [
            (loss, operator.lt),
            (metrics, operator.gt),
        ]:
            for name, val in d.items():
                first_time = name not in self.best
                if first_time or op(val, self.best[name]["val"]):
                    filename = f"epoch={epoch}_{name}={self._fmt_float(val)}.ckpt"
                    filepath = self.filesystem.join(self.dirpath, filename)
                    self.save(state, filepath)
                    logging.info(f"New best {name}, saving {filepath}")
                    if not first_time:
                        if self.filesystem.exists(self.best[name]["filepath"]):
                            self.filesystem.remove(self.best[name]["filepath"])
                        else:
                            logging.warning(
                                f'Previous best {name} checkpoint '
                                f'{self.best[name]["filepath"]} does not exist. '
                                'Skipping removal.'
                            )
                    # convert numpy scalar to builtin to prevent errors when loading
                    if isinstance(val, np.generic):
                        val = val.item()
                    self.best[name] = {"val": val, "filepath": filepath}

    def save(self, state, path):
        """Save a state dictionary."""
        self.filesystem.makedirs(self.filesystem.dirname(path))
        with io.BytesIO() as buffer:
            torch.save(state, buffer)
            buffer.seek(0)
            self.filesystem.put(buffer, path)

    @staticmethod
    def _fmt_float(x):
        return f"{x:.2e}" if abs(x) < 0.1 or abs(x) >= 100 else f"{x:.2f}"

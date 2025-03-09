import torch
import torch.nn as nn
import torch.nn.functional as F

from ..nets import NetRegistry
from ..signal.stft import STFT
from ..training.losses import LossRegistry
from ..utils import count_params


class BaseHA(nn.Module):
    """Base class for all hearing aid modules.

    Sub-classes might re-implement some of the methods, most likely :meth:`loss`,
    :meth:`_enhance`, :meth:`forward` and :meth:`transform`.

    All models should call the :meth:`post_init` method at the end of :meth:`__init__`
    to assign the ``_optimizer``, ``_loss``, ``_scheduler``, ``_grad_clip``, ``stft``,
    ``spk_adapt_net``, ``spk_adapt_stft``, ``_wav_norm`` and ``_norm_clean`` attributes.

    """

    def post_init(
        self,
        optimizer,
        optimizer_kw,
        loss,
        loss_kw=None,
        scheduler=None,
        scheduler_kw=None,
        grad_clip=None,
        stft_kw=None,
        stft_future_frames=0,
        spk_adapt_net=None,
        spk_adapt_net_kw=None,
        spk_adapt_stft_kw=None,
        wav_norm=None,
        norm_clean=True,
        audiogram=False,
        _labels_include_clean=True,
        _labels_include_noisy=False,
    ):
        """Post-initialization.

        Assigns optimizer, loss, scheduler, gradient clipping, STFT and speaker
        adaptation network  attributes. Must be called at the end of the
        :meth:`__init__` method.

        Parameters
        ----------
        optimizer : str or torch.optim.Optimizer
            Optimizer to use. If ``str``, a callable is loaded from the ``torch.optim``
            module.
        optimizer_kw : dict
            Keyword arguments passed to the optimizer constructor.
        loss : str
            Loss function. Must be in ``mbchl.losses.LossRegistry``.
        loss_kw : dict, optional
            Keyword arguments passed to the loss function constructor.
        scheduler : str or torch.optim.lr_scheduler._LRScheduler, optional
            Scheduler to use. If ``str``, a callable is loaded from the
            ``torch.optim.lr_scheduler`` module. If ``None``, no scheduler is used.
        scheduler_kw : dict, optional
            Keyword arguments passed to the scheduler constructor.
        grad_clip : float, optional
            Gradient clipping value. If ``None``, no clipping is performed.
        stft_kw : dict, optional
            Keyword arguments passed to the STFT constructor. If ``None``, the ``stft``
            attribute is set to ``None``.
        stft_future_frames : int, optional
            Number of future STFT frames to predict.
        spk_adapt_net : str, optional
            Speaker adaptation network. Must be in ``mbchl.nets.NetRegistry``. If
            ``None``, the ``spk_adapt_net`` attribute is set to ``None``.
        spk_adapt_net_kw : dict, optional
            Keyword arguments passed to the speaker adaptation network constructor.
        spk_adapt_stft_kw : dict, optional
            Keyword arguments passed to the speaker adaptation STFT constructor. If
            ``None``, the ``spk_adapt_stft`` attribute is set to ``None``.
        wav_norm : {"peak", "rms"}, optional
            Waveform normalization. If ``None``, no normalization is performed.
        norm_clean : bool, optional
            Whether to apply the normalization factor to the clean signal as well
            during training.
        audiogram : bool, optional
            Whether the hearing aid takes an audiogram as an input.

        """
        if audiogram and spk_adapt_net is not None:
            raise ValueError(
                "Audiogram and speaker adaptation are currently not supported together"
            )
        self.stft = None if stft_kw is None else STFT(**stft_kw)
        self.stft_future_frames = stft_future_frames
        self.spk_adapt_net = (
            None
            if spk_adapt_net is None
            else NetRegistry.get(spk_adapt_net)(**(spk_adapt_net_kw or {}))
        )
        self.spk_adapt_stft = (
            None if spk_adapt_stft_kw is None else STFT(**spk_adapt_stft_kw)
        )
        self._loss = LossRegistry.get(loss)(**(loss_kw or {}))
        # initialize optimizer after all the network parameters are defined!!!
        if isinstance(optimizer, str):
            optimizer = getattr(torch.optim, optimizer)
        self._optimizer = optimizer(self.parameters(), **optimizer_kw)
        # initialize scheduler after optimizer
        if isinstance(scheduler, str):
            scheduler = getattr(torch.optim.lr_scheduler, scheduler)
        if scheduler is None:
            self._scheduler = None
        else:
            self._scheduler = scheduler(self._optimizer, **(scheduler_kw or {}))
            if not isinstance(
                self._scheduler,
                (
                    torch.optim.lr_scheduler.LambdaLR,
                    torch.optim.lr_scheduler.MultiplicativeLR,
                    torch.optim.lr_scheduler.StepLR,
                    torch.optim.lr_scheduler.MultiStepLR,
                    torch.optim.lr_scheduler.ConstantLR,
                    torch.optim.lr_scheduler.LinearLR,
                    torch.optim.lr_scheduler.ExponentialLR,
                    torch.optim.lr_scheduler.PolynomialLR,
                    torch.optim.lr_scheduler.CosineAnnealingLR,
                    torch.optim.lr_scheduler.ReduceLROnPlateau,
                ),
            ):
                raise ValueError(
                    f"Unsupported scheduler {self._scheduler.__class__.__name__}"
                )
        self._grad_clip = grad_clip
        self._wav_norm = wav_norm
        self._norm_clean = norm_clean
        self._audiogram = audiogram
        self._labels_include_clean = _labels_include_clean
        self._labels_include_noisy = _labels_include_noisy

    def transform(self, signals):
        """Input pre-processing.

        Pre-processing that can be separated from inference in :meth:`forward`. Executed
        by workers when loading the data before making mini-batches, but also during
        validation to pre-process waveforms already moved to device. This means this
        should be able to run on CPU or GPU depending on the input device, even if the
        model was moved to GPU!

        Parameters
        ----------
        signals : torch.Tensor or tuple[torch.Tensor, ...]
            Input signals.

        Returns
        -------
        torch.Tensor or tuple[torch.Tensor, ...]
            Pre-processed signals with arbitrary shapes. The last dimension should be
            homogeneous to time (e.g. STFT frames), such that tensors can be padded to
            form mini-batches.

        """
        assert self.spk_adapt_net is None or not self._audiogram
        if self._audiogram:
            x, y, audiogram = signals
        elif self.spk_adapt_net is not None:
            x, y, spk_adapt = signals
            if self.spk_adapt_stft is not None:
                spk_adapt = self.spk_adapt_stft(spk_adapt)
        else:
            x, y = signals
        x, factor = self.normalize(x)
        if self._norm_clean:
            y = y / factor
        if not self._labels_include_clean:
            y = torch.tensor([], device=y.device, dtype=y.dtype)
        if self._labels_include_noisy:
            y = torch.cat([x, y], dim=0)
        if self.stft is not None:
            x = self.stft(x)
            x = self.pad_stft_future_frames(x)
        if self._audiogram:
            return x, y, audiogram
        elif self.spk_adapt_net is not None:
            return x, y, spk_adapt
        else:
            return x, y

    def enhance(self, x, use_amp=False, extra_inputs=None):
        """Signal enhancement.

        Estimates a clean speech signal given a noisy input mixture. Supports batched
        inputs. Models should not overwrite this method and should overwrite
        :meth:`_enhance` instead.

        Parameters
        ----------
        x : torch.Tensor
            Noisy mixture. Shape ``(n_input_channels, n_samples)`` or
            ``(batch_size, n_input_channels, n_samples)``.
        use_amp : bool, optional
            Whether to use automatic mixed precision.
        extra_inputs : tuple[torch.Tensor, ...], optional
            Extra inputs for the model, e.g. speaker adaptations. Assumed batched or
            unbatched depending on ``x``.

        Returns
        -------
        torch.Tensor
            Enhanced signal. Shape ``(n_output_channels, n_samples)`` if unbatched
            input, else shape ``(batch_size, n_output_channels, n_samples)``.

        """
        unbatched = x.ndim == 2
        if unbatched:
            x = x.unsqueeze(0)
            if extra_inputs is not None:
                extra_inputs = tuple(e.unsqueeze(0) for e in extra_inputs)
        elif x.ndim != 3:
            raise ValueError(f"input must be 2 or 3 dimensional, got {x.ndim}")
        output = self._enhance(x, use_amp, extra_inputs)
        if not output.ndim == 3:
            raise ValueError(
                f"output from _enhance must be 3 dimensional, got {output.ndim}"
            )
        if unbatched:
            output = output.squeeze(0)
        return output

    def _enhance(self, x, use_amp, extra_inputs):
        """Batched signal enhancement.

        Same as :meth:`enhance` but assumes batched inputs. Called inside
        :meth:`enhance`. All models must overwrite this method.

        Parameters
        ----------
        x : torch.Tensor
            Batched noisy mixture. Shape ``(batch_size, n_input_channels, n_samples)``.
        use_amp : bool
            Whether to use automatic mixed precision.
        extra_inputs : tuple[torch.Tensor, ...]
            Extra inputs for the model, e.g. speaker adaptations.

        Returns
        -------
        torch.Tensor
            Batched enhanced signal. Shape ``(batch_size, n_output_channels,
            n_samples)``.

        """
        length = x.shape[-1]
        device = x.device.type
        dtype = torch.bfloat16 if device == "cpu" else torch.float16
        assert extra_inputs is None or len(extra_inputs) <= 1
        assert self.spk_adapt_net is None or not self._audiogram
        if extra_inputs and self.spk_adapt_net is not None:
            spk_adapt = extra_inputs[0]
        else:
            spk_adapt = None
        if extra_inputs and self._audiogram:
            audiogram = extra_inputs[0]
        else:
            audiogram = None
        with torch.autocast(device_type=device, dtype=dtype, enabled=use_amp):
            x, factor = self.normalize(x)
            if self.stft is not None:
                x = self.stft(x)
                x = self.pad_stft_future_frames(x)
            if self.spk_adapt_stft is not None:
                spk_adapt = self.spk_adapt_stft(spk_adapt)
            x = self(x, spk_adapt=spk_adapt, audiogram=audiogram)
            if self.stft is not None:
                x = self.stft.inverse(x, length=length)
            x = x * factor
        return x

    def train_step(self, batch, lengths, use_amp=False, scaler=None):
        """Training step.

        Sets gradients to zero, calculates the loss and backpropagates given batched
        observations from the dataloader. Called inside the training loop.

        Parameters
        ----------
        batch : torch.Tensor or tuple[torch.Tensor, ...]
            Batched inputs from the dataloader. ``batch`` is a tensor if the dataset's
            :meth:`__getitem__` returns a single output, or a tuple of tensors if it
            returns multiple outputs.
        lengths : torch.Tensor
            Original input lengths along the last dimension. Can be important for
            post-processing, e.g. to ensure sample-wise losses are not aggregated over
            the zero-padded regions. Shape ``(batch_size,)`` if the dataset's
            :meth:`__getitem__` returns a single output, else shape ``(batch_size,
            n_signals)``.
        use_amp : bool, optional
            Whether to use automatic mixed precision.
        scaler : torch.amp.GradScaler, optional
            Gradient scaler for automatic mixed precision.

        Returns
        -------
        torch.Tensor or dict
            Loss to backpropagate. Can be a dict if multiple losses are calculated like
            with GANs.

        """
        self._optimizer.zero_grad()
        loss = self.loss(batch, lengths, use_amp)
        if scaler is None:
            scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
        self.update(loss, scaler)
        return loss

    def val_step(self, batch, lengths, use_amp):
        """Validate step.

        Calculates the loss to log given batched observations from the dataloader in a
        ``torch.no_grad()`` context manager. Called inside the validation loop.

        Parameters
        ----------
        batch : torch.Tensor or tuple[torch.Tensor, ...]
            Batched inputs from the dataloader. ``batch`` is a tensor if the dataset's
            :meth:`__getitem__` returns a single output, or a tuple of tensors if it
            returns multiple outputs.
        lengths : torch.Tensor
            Original input lengths along the last dimension. Can be important for
            post-processing, e.g. to ensure sample-wise losses are not aggregated over
            the zero-padded regions. Shape ``(batch_size,)`` if the dataset's
            :meth:`__getitem__` returns a single output, else shape ``(batch_size,
            n_signals)``.
        use_amp : bool
            Whether to use automatic mixed precision.

        Returns
        -------
        torch.Tensor or dict
            Loss to log. Can be a dict if multiple losses are calculated like with GANs.

        """
        return self.loss(batch, lengths, use_amp)

    def loss(self, batch, lengths, use_amp):
        """Loss calculation.

        Calculates the loss given batched observations from the dataloader. Called in
        the training and validation loops.

        Parameters
        ----------
        batch : torch.Tensor or tuple[torch.Tensor, ...]
            Batched inputs from the dataloader. ``batch`` is a tensor if the dataset's
            :meth:`__getitem__` returns a single output, or a tuple of tensors if it
            returns multiple outputs.
        lengths : torch.Tensor
            Original input lengths along the last dimension. Can be important for
            post-processing, e.g. to ensure sample-wise losses are not aggregated over
            the zero-padded regions. Shape ``(batch_size,)`` if the dataset's
            :meth:`__getitem__` returns a single output, else shape ``(batch_size,
            n_signals)``.
        use_amp : bool
            Whether to use automatic mixed precision.

        Returns
        -------
        torch.Tensor
            Loss. Single-element tensor.

        """
        assert self.spk_adapt_net is None or not self._audiogram
        if self._audiogram:
            x, y, audiogram = batch
            spk_adapt = None
        elif self.spk_adapt_net is not None:
            x, y, spk_adapt = batch
            audiogram = None
        else:
            x, y = batch
            spk_adapt = None
            audiogram = None
        device = x.device.type
        dtype = torch.bfloat16 if device == "cpu" else torch.float16
        with torch.autocast(device_type=device, dtype=dtype, enabled=use_amp):
            x = self(x, spk_adapt=spk_adapt, audiogram=audiogram)
            if self.stft is not None:
                x = self.stft.inverse(x, length=y.shape[-1])
            loss = self._loss(x, y, lengths[:, 1], audiogram=audiogram)
        return loss.mean()

    def update(
        self, loss, scaler, net=None, optimizer=None, grad_clip=None, retain_graph=None
    ):
        """Network weight update.

        Backpropagates the loss and updates the network weights. Usually called in
        ``train_step``, after the gradients were zeroed and the loss was calculated.

        Parameters
        ----------
        loss : torch.Tensor
            Loss to backpropagate.
        scaler : torch.amp.GradScaler
            Gradient scaler for automatic mixed precision.
        net : torch.nn.Module, optional
            Network to update. If ``None``, uses ``self``.
        optimizer : torch.optim.Optimizer, optional
            Optimizer to use. If ``None``, uses ``self._optimizer``.
        grad_clip : float, optional
            Gradient clipping value. If ``None``, uses ``self._grad_clip``.
        retain_graph : bool, optional
            Whether to retain the computational graph. Passed to the :meth:`backward`
            method.

        """
        net = self if net is None else net
        optimizer = self._optimizer if optimizer is None else optimizer
        scaler.scale(loss).backward(retain_graph=retain_graph)
        grad_clip = self._grad_clip if grad_clip is None else grad_clip
        if grad_clip is not None:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(net.parameters(), grad_clip)
        scaler.step(optimizer)
        scaler.update()

    def forward(self, x, spk_adapt=None, audiogram=None):
        """Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor.
        spk_adapt : torch.Tensor, optional
            Speaker adaptation.
        audiogram : torch.Tensor, optional
            Audiogram.

        Returns
        -------
        torch.Tensor
            Output tensor.

        """
        assert not (spk_adapt is None and self.spk_adapt_net is not None)
        assert not (spk_adapt is not None and self.spk_adapt_net is None)
        assert not (audiogram is None and self._audiogram)
        assert not (audiogram is not None and not self._audiogram)
        assert self.spk_adapt_net is None or not self._audiogram
        assert spk_adapt is None or x.shape[0] == spk_adapt.shape[0]
        assert audiogram is None or x.shape[0] == audiogram.shape[0]
        spk_emb = None if spk_adapt is None else self.spk_adapt_net(spk_adapt)
        if audiogram is not None:
            audiogram = torch.cat(
                [audiogram[..., 0] / 10000, audiogram[..., 1] / 100], dim=-1
            )
        emb = spk_emb if spk_emb is not None else audiogram
        assert emb is None or emb.ndim == 2  # (batch_size, emb_dim)
        return self.net(x, emb=emb)

    def pre_train(self, dataset, dataloader, epochs):
        """Pre-training instructions.

        Contains instructions that need to be run once before the training loop, e.g.
        calculate normalization statistics of input features.

        Parameters
        ----------
        dataset : torch.utils.data.Dataset
            The training dataset.
        dataloader : torch.utils.data.DataLoader
            The associated training dataloader.
        epochs : int
            Number of epochs. Useful for setting learning rate schedules.

        """
        pass

    def on_validate(self, val_loss):
        """Validate hook.

        Called after each validation epoch. Can be used to schedule learning rates. No
        need to save checkpoints here as this is done in the trainer class.

        Parameters
        ----------
        val_loss : torch.Tensor or dict
            Validation loss. Can be a dict if ``val_step`` returns a dict.

        Example
        -------
        If ``self._scheduler`` is a ``torch.optim.lr_scheduler.ReduceLROnPlateau``, the
        learning rate can be scheduled based on the validation loss:

        def on_validate(self, val_loss):
            self._scheduler.step(val_loss)

        """
        if isinstance(self._scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            if isinstance(val_loss, dict):
                if len(val_loss) == 1:
                    val_loss = next(iter(val_loss.values()))
                elif len(val_loss) > 1:
                    raise ValueError(
                        "ReduceLROnPlateau scheduler expects a single loss value"
                    )
                elif len(val_loss) == 0:
                    raise ValueError("ReduceLROnPlateau scheduler got empty loss dict")
            self._scheduler.step(val_loss)

    def on_train(self):
        """Train hook.

        Called after each training epoch. Can be used to schedule learning rates. No
        need to save checkpoints here as this is done in the trainer class.

        Example
        -------
        If ``self._scheduler`` is a ``torch.optim.lr_scheduler.ExponentialLR``, the
        learning rate can be scheduled based on the training epoch:

        >>> def on_train(self):
        ...     self._scheduler.step()

        """
        if isinstance(
            self._scheduler,
            (
                torch.optim.lr_scheduler.LambdaLR,
                torch.optim.lr_scheduler.MultiplicativeLR,
                torch.optim.lr_scheduler.StepLR,
                torch.optim.lr_scheduler.MultiStepLR,
                torch.optim.lr_scheduler.ConstantLR,
                torch.optim.lr_scheduler.LinearLR,
                torch.optim.lr_scheduler.ExponentialLR,
                torch.optim.lr_scheduler.PolynomialLR,
                torch.optim.lr_scheduler.CosineAnnealingLR,
            ),
        ):
            self._scheduler.step()

    def count_params(self, requires_grad=True, unique=True):
        """Count parameters.

        Parameters
        ----------
        requires_grad : bool, optional
            Whether to count trainable parameters only.
        unique : bool, optional
            Whether to count shared parameters only once.

        Returns
        -------
        int
            Number of trainable parameters.

        """
        return count_params(self, requires_grad, unique)

    @torch.no_grad()
    def set_all_weights(self, val=1e-3, requires_grad=True, buffers=False):
        """Set all weights to a given value.

        Useful for unit testing.

        Parameters
        ----------
        val : float, optional
            Value to fill the weights with.
        requires_grad : bool, optional
            Whether to set the trainable weights only.
        buffers : bool, optional
            Whether to set the buffers as well.

        """
        for p in self.parameters():
            if p.requires_grad or not requires_grad:
                p.fill_(val)
        if buffers:
            for b in self.buffers():
                b.fill_(val)

    def pad_stft_future_frames(self, x):
        """Pad STFT future frames.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor.

        Returns
        -------
        torch.Tensor
            Padded tensor.

        """
        return F.pad(
            x[..., : x.shape[-1] - self.stft_future_frames],
            (self.stft_future_frames, 0),
        )

    def normalize(self, x):
        """Normalize waveform."""
        # x has shape (channels, time) or (batch, channels, time)
        if self._wav_norm is None:
            factor = 1.0
        elif self._wav_norm == "peak":
            factor = x.abs().amax(axis=(-1, -2), keepdims=True)
        elif self._wav_norm == "rms":
            factor = x.std(axis=(-1, -2), keepdims=True)
        else:
            raise ValueError(f"Invalid waveform normalization: {self._wav_norm}")
        return x / factor, factor

    def state_dict(self, *args, **kwargs):
        """Return the model state dict."""
        output = {
            "model": super().state_dict(*args, **kwargs),
            "optimizer": self._optimizer.state_dict(),
        }
        if self._scheduler is not None:
            output["scheduler"] = self._scheduler.state_dict()
        return output

    def load_state_dict(self, state_dict):
        """Load the model state dict."""
        super().load_state_dict(state_dict["model"])
        self._optimizer.load_state_dict(state_dict["optimizer"])
        if self._scheduler is not None:
            self._scheduler.load_state_dict(state_dict["scheduler"])

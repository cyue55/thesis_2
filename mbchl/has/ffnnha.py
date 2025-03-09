from typing import override

import torch

from ..nets import FFNN
from ..signal.mel import MelFilterbank
from .base import BaseHA
from .registry import HARegistry


@HARegistry.register("ffnn")
class FFNNHA(BaseHA):
    """Feed-forward neural network-based hearing aid.

    Predicts a real-valued mask on a mel-frequency scale.
    """

    def __init__(
        self,
        input_channels=1,
        reference_channels=[0],
        fs=16000,
        stacks=5,
        n_mels=64,
        hidden_sizes=[1024, 1024],
        dropout=0.0,
        aggregate=False,
        fusion_layer=None,
        emb_dim=None,
        eps=1e-7,
        optimizer="Adam",
        optimizer_kw={"lr": 1e-4},
        loss="mse",
        loss_kw=None,
        scheduler=None,
        scheduler_kw=None,
        grad_clip=None,
        stft_kw={"frame_length": 512, "hop_length": 128},
        stft_future_frames=0,
        spk_adapt_net=None,
        spk_adapt_net_kw=None,
        spk_adapt_stft_kw=None,
        wav_norm=None,
        norm_clean=True,
        audiogram=False,
    ):
        super().__init__()
        self.reference_channels = reference_channels
        self.stacks = stacks
        self.eps = eps
        self.mel_fb = MelFilterbank(
            n_filters=n_mels,
            n_fft=stft_kw.get("n_fft") or stft_kw["frame_length"],
            fs=fs,
        )
        self.net = FFNN(
            input_size=n_mels * (stacks + 1) * input_channels,
            output_size=emb_dim if aggregate else n_mels,
            hidden_sizes=hidden_sizes,
            dropout=dropout,
            aggregate=aggregate,
            fusion_layer=fusion_layer,
            emb_dim=emb_dim,
        )
        self.post_init(
            optimizer=optimizer,
            optimizer_kw=optimizer_kw,
            loss=loss,
            loss_kw=loss_kw,
            scheduler=scheduler,
            scheduler_kw=scheduler_kw,
            grad_clip=grad_clip,
            stft_kw=stft_kw,
            stft_future_frames=stft_future_frames,
            spk_adapt_net=spk_adapt_net,
            spk_adapt_net_kw=spk_adapt_net_kw,
            spk_adapt_stft_kw=spk_adapt_stft_kw,
            wav_norm=wav_norm,
            norm_clean=norm_clean,
            audiogram=audiogram,
        )

    @override
    def transform(self, signals):
        assert self.spk_adapt_net is None or not self._audiogram
        if self._audiogram:
            x, y, audiogram = signals
        elif self.spk_adapt_net is not None:
            x, y, spk_adapt = signals
            if self.spk_adapt_stft is not None:
                spk_adapt = self.spk_adapt_stft(spk_adapt)
        else:
            x, y = signals
        x, y = self.stft(x), self.stft(y)
        x = self.pad_stft_future_frames(x)
        x, y = self._features(x), self._irm(x, y)
        if self._audiogram:
            return x, y, audiogram
        elif self.spk_adapt_net is not None:
            return x, y, spk_adapt
        else:
            return x, y

    @override
    def loss(self, batch, lengths, use_amp):
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
            loss = self._loss(x, y, lengths[:, 1], audiogram=audiogram)
        return loss.mean()

    def _enhance(self, x, use_amp, extra_inputs):
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
            x = self.stft(x)
            x = self.pad_stft_future_frames(x)
            if self.spk_adapt_stft is not None:
                spk_adapt = self.spk_adapt_stft(spk_adapt)
            features = self._features(x)
            mask = self(features, spk_adapt=spk_adapt, audiogram=audiogram)
            mask_extrapolated = self.mel_fb.inverse(mask).unsqueeze(1)
            x = mask_extrapolated * x[:, self.reference_channels, :, :]
            x = self.stft.inverse(x, length=length)
        return x

    def _features(self, x):
        x = x.abs()
        x = self.mel_fb(x)
        x = torch.log(x + self.eps)
        x = x.reshape(*x.shape[:-3], -1, x.shape[-1])
        x = self._stack(x)
        return x

    def _irm(self, x, y):
        n = x[self.reference_channels, :, :] - y
        y, n = y.abs().pow(2).mean(0), n.abs().pow(2).mean(0)
        y, n = self.mel_fb(y), self.mel_fb(n)
        return torch.sqrt(y / (n + y + self.eps))

    def _stack(self, x):
        output = [x]
        for i in range(self.stacks):
            rolled = x.roll(i + 1, -1)
            rolled[..., : i + 1] = x[..., :1]
            output.append(rolled)
        return torch.cat(output, dim=-2)

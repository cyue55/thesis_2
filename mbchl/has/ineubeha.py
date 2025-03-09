from typing import override

import torch

from ..nets import NetRegistry
from .base import BaseHA
from .registry import HARegistry


@HARegistry.register("ineube")
class iNeuBeHA(BaseHA):
    """Iterative neural beamforming (iNeuBe) hearing aid.

    iNeuBe was proposed in [1].

    .. [1] Y.-J. Lu, S. Cornell, X. Chang, W. Zhang, C. Li, Z. Ni, Z.-Q. Wang and S.
       Watanabe, "Towards Low-Distortion Multi-Channel Speech Enhancement: The ESPNET-SE
       Submission to the L3DAS22 Challenge", in Proc. ICASSP, 2022.
    """

    def __init__(
        self,
        net1_cls,
        net1_kw,
        net2_cls,
        net2_kw,
        optimizer="Adam",
        optimizer_kw={"lr": 1e-3},
        loss="snr",
        loss_kw=None,
        scheduler="ExponentialLR",
        scheduler_kw={"gamma": 0.99},
        grad_clip=5.0,
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
        self.net_1 = NetRegistry.get(net1_cls)(**net1_kw)
        self.net_2 = NetRegistry.get(net2_cls)(**net2_kw)
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
    def forward(self, x, spk_adapt=None, audiogram=None):
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
        est = self.net_1(x, emb=emb)
        beam = self._mcwf(x, est)
        x = torch.cat([beam, est, x], dim=1)
        output = self.net_2(x, emb=emb)
        return est, output

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
            est, out = self(x, spk_adapt=spk_adapt, audiogram=audiogram)
            if self.stft is not None:
                est = self.stft.inverse(est, length=y.shape[-1])
                out = self.stft.inverse(out, length=y.shape[-1])
            loss = self._loss(est, y, lengths[:, 1], audiogram=audiogram) + self._loss(
                out, y, lengths[:, 1], audiogram=audiogram
            )
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
            if self.stft is not None:
                x = self.stft(x)
                x = self.pad_stft_future_frames(x)
            if self.spk_adapt_stft is not None:
                spk_adapt = self.spk_adapt_stft(spk_adapt)
            _, x = self(x, spk_adapt=spk_adapt, audiogram=audiogram)
            if self.stft is not None:
                x = self.stft.inverse(x, length=length)
        return x

    def _mcwf(self, mixture, estimate, eps=1e-7):
        # mixture has shape (B, M, F, T)
        # estimate has shape (B, output_channels, F, T)
        z = torch.einsum("bmft,boft->bmoft", mixture, estimate.conj())
        z = z.cumsum(-1).to(torch.complex128)
        scm = torch.einsum("bmft,bnft->bmnft", mixture, mixture.conj())
        scm = scm.cumsum(-1).to(torch.complex128)
        scm = scm.permute(0, 3, 4, 1, 2)  # (B, F, T, M, M)
        eye = torch.eye(scm.shape[-1], device=scm.device, dtype=scm.dtype)
        inv = torch.inverse(scm + eps * eye)
        inv = inv.permute(0, 3, 4, 1, 2)  # (B, M, M, F, T)
        w = torch.einsum("bmnft,bnoft->bmoft", inv, z)
        out = torch.einsum("bmoft,bnft->boft", w.conj(), mixture)
        return out.to(mixture.dtype)

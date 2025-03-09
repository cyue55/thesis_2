import torch
import torch.nn.functional as F

from ...nets import NetRegistry
from ...signal.mel import MelFilterbank
from ..base import BaseHA
from ..registry import HARegistry
from .hifigan import init_hifigan
from .preconditioning import Preconditioning
from .sdes import SDERegistry
from .solvers import SolverRegistry


class _BaseDiffusionHA(BaseHA):
    def __init__(
        self,
        input_channels=1,
        reference_channels=[0],
        net="diffunet",
        net_kw=None,
        sde_training="edm-training",
        sde_training_kw=None,
        sde_sampling="edm-sampling",
        sde_sampling_kw=None,
        solver="edm",
        solver_kw=None,
        preconditioning="edm",
        sigma_data=0.5,
        t_eps=0.01,
        optimizer="Adam",
        optimizer_kw={"lr": 1e-4},
        loss="mse",
        loss_kw=None,
        scheduler=None,
        scheduler_kw=None,
        grad_clip=None,
        stft_kw={"frame_length": 512, "hop_length": 128, "compression_factor": 0.5},
        stft_future_frames=0,
        spk_adapt_net=None,
        spk_adapt_net_kw=None,
        spk_adapt_stft_kw=None,
        wav_norm="peak",
        norm_clean=True,
        audiogram=False,
        mel_domain=False,
        mel_fb_kw=None,
        mel_power=1,
        mel_log=True,
        mel_log_eps=1e-7,
        hifigan_ckpt=None,
        hifigan_json=None,
    ):
        super().__init__()
        self.reference_channels = reference_channels
        self.t_eps = t_eps
        self.mel_domain = mel_domain
        self.sde_training = SDERegistry.get(sde_training)(**(sde_training_kw or {}))
        self.sde_sampling = SDERegistry.get(sde_sampling)(**(sde_sampling_kw or {}))
        self.solver = SolverRegistry.get(solver)(**(solver_kw or {}))
        if mel_domain:
            self.mel_fb = MelFilterbank(**(mel_fb_kw or {}))
            self.mel_power = mel_power
            self.mel_log = mel_log
            self.mel_log_eps = mel_log_eps
            self.hifigan = init_hifigan(ckpt_path=hifigan_ckpt, json_path=hifigan_json)
        self.net = Preconditioning(
            raw_net=NetRegistry.get(net)(**(net_kw or {})),
            cskip=preconditioning,
            cout=preconditioning,
            cin=preconditioning,
            cnoise=preconditioning,
            cshift=preconditioning,
            weight=preconditioning,
            sigma_data=sigma_data,
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
        x, factor = self.normalize(x)
        if self._norm_clean:
            y = y / factor
        x, y = self.stft(x), self.stft(y)
        if self.mel_domain:
            x, y = x.abs().pow(self.mel_power), y.abs().pow(self.mel_power)
            x, y = self.mel_fb(x), self.mel_fb(y)
            if self.mel_log:
                x = x.clamp(min=self.mel_log_eps).log()
                y = y.clamp(min=self.mel_log_eps).log()
        else:
            x, y = x[..., :-1, :], y[..., :-1, :]  # discard nyquist frequency
        x = self.pad_stft_future_frames(x)
        if self.spk_adapt_net is None:
            return x, y
        else:
            return x, y, spk_adapt

    def forward(self, x, y, y_score, sigma, t, sde=None, spk_emb=None, audiogram=None):
        # y is the center of the prior distribution which lives in same space as x
        # y_score is the input of the score model which can have more channels
        assert not (spk_emb is None and self.spk_adapt_net is not None)
        assert not (spk_emb is not None and self.spk_adapt_net is None)
        assert not (audiogram is None and self._audiogram)
        assert not (audiogram is not None and not self._audiogram)
        assert self.spk_adapt_net is None or not self._audiogram
        assert spk_emb is None or x.shape[0] == spk_emb.shape[0]
        assert audiogram is None or x.shape[0] == audiogram.shape[0]
        if audiogram is not None:
            audiogram = torch.cat(
                [audiogram[..., 0] / 10000, audiogram[..., 1] / 100], dim=-1
            )
        emb = spk_emb if spk_emb is not None else audiogram
        assert emb is None or emb.ndim == 2  # (batch_size, emb_dim)
        if sde is None:
            sde = self.sde_training if self.training else self.sde_sampling
        return self.net(x, y, y_score, sigma, t, sde, emb)

    def loss(self, batch, lengths, use_amp):
        # y_score is noisy, x_0 is clean
        assert self.spk_adapt_net is None or not self._audiogram
        if self._audiogram:
            y_score, x_0, audiogram = batch
            spk_adapt = None
        elif self.spk_adapt_net is not None:
            y_score, x_0, spk_adapt = batch
            audiogram = None
        else:
            y_score, x_0 = batch
            spk_adapt = None
            audiogram = None
        y = y_score[:, self.reference_channels, :, :]
        # y is the center of the prior distribution which lives in same space as x
        # y_score is the input of the score model which can have more channels
        t = (
            torch.rand(x_0.shape[0], 1, 1, 1, device=y.device) * (1 - self.t_eps)
            + self.t_eps
        )
        sde = self.sde_training if self.training else self.sde_sampling
        sigma = sde.sigma(t)
        n = sigma * torch.randn_like(x_0)
        weight = self.net.weight(sigma)
        device = y.device.type
        dtype = torch.bfloat16 if device == "cpu" else torch.float16
        with torch.autocast(device_type=device, dtype=dtype, enabled=use_amp):
            spk_emb = None if spk_adapt is None else self.spk_adapt_net(spk_adapt)
            d = self(
                x_0 - y + n,
                y,
                y_score,
                sigma,
                t,
                sde=sde,
                spk_emb=spk_emb,
                audiogram=audiogram,
            )
            loss = self._loss(
                d, x_0 - y, lengths[:, 1], weight=weight, audiogram=audiogram
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
            x, factor = self.normalize(x)
            x = self.stft(x)
            if self.mel_domain:
                x = x.abs().pow(self.mel_power)
                x = self.mel_fb(x)
                if self.mel_log:
                    x = x.clamp(min=self.mel_log_eps).log()
            else:
                x = x[..., :-1, :]  # discard nyquist frequency
            x = self.pad_stft_future_frames(x)
            if self.spk_adapt_stft is not None:
                spk_adapt = self.spk_adapt_stft(spk_adapt)
            x_ref = x[:, self.reference_channels, :, :]
            spk_emb = None if spk_adapt is None else self.spk_adapt_net(spk_adapt)
            emb = spk_emb if spk_emb is not None else audiogram
            x, _ = self.solver(self.sde_sampling, x_ref, x, self.net, emb)
            if self.mel_domain:
                batch, channels, freqs, time = x.shape
                x = x.reshape(batch * channels, freqs, time)
                if not self.mel_log:
                    # hifigan input is log so force it even if mel_log is False
                    x = x.clamp(min=self.mel_log_eps).log()
                x = self.hifigan(x).squeeze(1)
                x = x.reshape(batch, channels, -1)
                x = x[..., :length]
            else:
                x = F.pad(x, (0, 0, 0, 1))  # pad nyquist frequency
                x = self.stft.inverse(x, length=length)
            x = x * factor
        return x

    def state_dict(self, *args, **kwargs):
        # override to avoid pickling hifigan
        if self.mel_domain:
            hifigan = self.hifigan
            self.hifigan = None
        output = super().state_dict(*args, **kwargs)
        if self.mel_domain:
            self.hifigan = hifigan
        return output

    def load_state_dict(self, state_dict):
        # override to avoid loading hifigan
        if self.mel_domain:
            hifigan = self.hifigan
            self.hifigan = None
        super().load_state_dict(state_dict)
        if self.mel_domain:
            self.hifigan = hifigan


@HARegistry.register("sgmsep")
class SGMSEpHA(_BaseDiffusionHA):
    """SGMSE+ hearing aid.

    SGMSE+ was proposed in [1].

    .. [1] J. Richter, S. Welker, J.-M. Lemercier, B. Lay and T. Gerkmann, "Speech
       Enhancement and Dereverberation with Diffusion-Based Generative Models", in
       IEEE/ACM Trans. Audio, Speech, Lang. Process., 2023.
    """

    def __init__(
        self,
        net_kw={
            "in_channels": 4,  # (2 - mel_domain) * (input_channels + output_channels)
            "out_channels": 2,  # (2 - mel_domain) * output_channels,
            "aux_out_channels": 4,  # 4 * output_channels
            "num_freqs": 256,  # n_fft // 2
            "base_channels": 128,
            "channel_mult": [1, 1, 2, 2, 2, 2, 2],
            "num_blocks_per_res": 2,
            "noise_channel_mult": 2,
            "emb_channel_mult": 4,
            "fir_kernel": [1, 3, 3, 1],
            "attn_resolutions": [16],
            "attn_bottleneck": True,
            "encoder_type": "skip",
            "decoder_type": "skip",
            "block_type": "ncsn",
            "skip_scale": 0.5**0.5,
            "dropout": 0.0,
        },
        sde_training="richter-ouve",
        sde_sampling="richter-ouve",
        solver="pc",
        preconditioning="richter",
        stft_kw={
            "frame_length": 512,
            "hop_length": 128,
            "scale_factor": 0.15,
            "compression_factor": 0.5,
        },
        **kwargs,
    ):
        super().__init__(
            net_kw=net_kw,
            sde_training=sde_training,
            sde_sampling=sde_sampling,
            solver=solver,
            preconditioning=preconditioning,
            stft_kw=stft_kw,
            **kwargs,
        )


@HARegistry.register("sgmsepm")
class SGMSEpMHA(_BaseDiffusionHA):
    """SGMSE+M hearing aid.

    SGMSE+M was proposed in [1].

    .. [1] J.-M. Lemercier, J. Richter, S. Welker and T. Gerkmann, "Analysing
       Diffusion-Based Generative Approaches Versus Discriminative Approaches for Speech
       Restoration", in Proc. ICASSP, 2023.
    """

    def __init__(
        self,
        net_kw={
            "in_channels": 4,  # (2 - mel_domain) * (input_channels + output_channels)
            "out_channels": 2,  # (2 - mel_domain) * output_channels,
            "aux_out_channels": 4,  # 4 * output_channels
            "num_freqs": 256,  # n_fft // 2
            "base_channels": 128,
            "channel_mult": [1, 2, 2, 2],
            "num_blocks_per_res": 1,
            "noise_channel_mult": 2,
            "emb_channel_mult": 4,
            "fir_kernel": [1, 3, 3, 1],
            "attn_resolutions": [],
            "attn_bottleneck": True,
            "encoder_type": "skip",
            "decoder_type": "skip",
            "block_type": "ncsn",
            "skip_scale": 0.5**0.5,
            "dropout": 0.0,
        },
        sde_training="richter-ouve",
        sde_sampling="richter-ouve",
        solver="pc",
        preconditioning="richter",
        stft_kw={
            "frame_length": 512,
            "hop_length": 128,
            "scale_factor": 0.15,
            "compression_factor": 0.5,
        },
        **kwargs,
    ):
        super().__init__(
            net_kw=net_kw,
            sde_training=sde_training,
            sde_sampling=sde_sampling,
            solver=solver,
            preconditioning=preconditioning,
            stft_kw=stft_kw,
            **kwargs,
        )


@HARegistry.register("sgmsepheun")
class SGMSEpHeunHA(_BaseDiffusionHA):
    """SGMSE+ with Heun-based solver and cosine schedule."""

    def __init__(
        self,
        net_kw={
            "in_channels": 4,  # (2 - mel_domain) * (input_channels + output_channels)
            "out_channels": 2,  # (2 - mel_domain) * output_channels,
            "aux_out_channels": 4,  # 4 * output_channels
            "num_freqs": 256,  # n_fft // 2
            "base_channels": 128,
            "channel_mult": [1, 1, 2, 2, 2, 2, 2],
            "num_blocks_per_res": 2,
            "noise_channel_mult": 2,
            "emb_channel_mult": 4,
            "fir_kernel": [1, 3, 3, 1],
            "attn_resolutions": [16],
            "attn_bottleneck": True,
            "encoder_type": "skip",
            "decoder_type": "skip",
            "block_type": "ncsn",
            "skip_scale": 0.5**0.5,
            "dropout": 0.0,
        },
        sde_training="brever-oucosine",
        sde_training_kw={"shift": 3.0},
        sde_sampling="brever-oucosine",
        sde_sampling_kw={"shift": 3.0},
        solver="edm",
        preconditioning="edm",
        sigma_data=0.1,
        stft_kw={
            "frame_length": 512,
            "hop_length": 128,
            "scale_factor": 0.15,
            "compression_factor": 0.5,
        },
        **kwargs,
    ):
        super().__init__(
            net_kw=net_kw,
            sde_training=sde_training,
            sde_training_kw=sde_training_kw,
            sde_sampling=sde_sampling,
            sde_sampling_kw=sde_sampling_kw,
            solver=solver,
            preconditioning=preconditioning,
            sigma_data=sigma_data,
            stft_kw=stft_kw,
            **kwargs,
        )


@HARegistry.register("sgmsepmheun")
class SGMSEpMHeunHA(_BaseDiffusionHA):
    r"""SGMSE+M\ :sup:`cos`\ :sub:`Heun` hearing aid.

    SGMSE+M\ :sup:`cos`\ :sub:`Heun` was proposed in [1].

    .. [1] P. Gonzalez, Z.-H. Tan, J. Østergaard, J. Jensen, T. S. Alstrøm and T. May.
       "The Effect of Training Dataset Size on Discriminative and Diffusion-Based Speech
       Enhancement Systems", in IEEE Signal Process. Lett., 2024.
    """

    def __init__(
        self,
        net_kw={
            "in_channels": 4,  # (2 - mel_domain) * (input_channels + output_channels)
            "out_channels": 2,  # (2 - mel_domain) * output_channels,
            "aux_out_channels": 4,  # 4 * output_channels
            "num_freqs": 256,  # n_fft // 2
            "base_channels": 128,
            "channel_mult": [1, 2, 2, 2],
            "num_blocks_per_res": 1,
            "noise_channel_mult": 2,
            "emb_channel_mult": 4,
            "fir_kernel": [1, 3, 3, 1],
            "attn_resolutions": [],
            "attn_bottleneck": True,
            "encoder_type": "skip",
            "decoder_type": "skip",
            "block_type": "ncsn",
            "skip_scale": 0.5**0.5,
            "dropout": 0.0,
        },
        sde_training="brever-oucosine",
        sde_training_kw={"shift": 3.0},
        sde_sampling="brever-oucosine",
        sde_sampling_kw={"shift": 3.0},
        solver="edm",
        preconditioning="edm",
        sigma_data=0.1,
        stft_kw={
            "frame_length": 512,
            "hop_length": 128,
            "scale_factor": 0.15,
            "compression_factor": 0.5,
        },
        **kwargs,
    ):
        super().__init__(
            net_kw=net_kw,
            sde_training=sde_training,
            sde_training_kw=sde_training_kw,
            sde_sampling=sde_sampling,
            sde_sampling_kw=sde_sampling_kw,
            solver=solver,
            preconditioning=preconditioning,
            sigma_data=sigma_data,
            stft_kw=stft_kw,
            **kwargs,
        )


@HARegistry.register("idmse")
class IDMSEHA(_BaseDiffusionHA):
    """Uses ADM, the Heun-based solver, and the EDM training and sampling schedules."""

    def __init__(
        self,
        net_kw={
            "in_channels": 4,  # (2 - mel_domain) * (input_channels + output_channels)
            "out_channels": 2,  # (2 - mel_domain) * output_channels,
            "aux_out_channels": 4,  # 4 * output_channels
            "num_freqs": 256,  # n_fft // 2
            "base_channels": 64,
            "channel_mult": [1, 2, 3, 4],
            "num_blocks_per_res": 1,
            "noise_channel_mult": 1,
            "emb_channel_mult": 4,
            "fir_kernel": [1, 1],
            "attn_resolutions": [],
            "attn_bottleneck": True,
            "encoder_type": "standard",
            "decoder_type": "standard",
            "block_type": "adm",
            "skip_scale": 0.5**0.5,
            "dropout": 0.0,
        },
        **kwargs,
    ):
        super().__init__(net_kw=net_kw, **kwargs)

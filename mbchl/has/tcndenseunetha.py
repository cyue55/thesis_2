from ..nets import TCNDenseUNet
from .base import BaseHA
from .registry import HARegistry


@HARegistry.register("tcndenseunet")
class TCNDenseUNetHA(BaseHA):
    """TCNDenseUNet hearing aid.

    TCNDenseUNet was proposed in [1] and [2].

    .. [1] Z.-Q. Wang, G. Wichern and J. Le Roux, "Leveraging Low-Distortion Target
       Estimates for Improved Speech Enhancement", arXiv preprint  arXiv:2110.00570,
       2021.
    .. [2] Y.-J. Lu, S. Cornell, X. Chang, W. Zhang, C. Li, Z. Ni, Z.-Q. Wang and S.
       Watanabe, "Towards Low-Distortion Multi-Channel Speech Enhancement: The ESPNET-SE
       Submission to the L3DAS22 Challenge", in Proc. ICASSP, 2022.
    """

    def __init__(
        self,
        input_channels=1,
        output_channels=1,
        hidden_channels=32,
        hidden_channels_dense=32,
        kernel_size_dense=(3, 3),
        kernel_size_tcn=3,
        tcn_repeats=4,
        tcn_blocks=7,
        tcn_channels=384,
        no_decoder=False,
        emb_dim=None,
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
        self.net = TCNDenseUNet(
            n_spk=output_channels,
            in_freqs=(stft_kw.get("n_fft") or stft_kw["frame_length"]) // 2 + 1,
            mic_channels=input_channels,
            hid_chans=hidden_channels,
            hid_chans_dense=hidden_channels_dense,
            ksz_dense=kernel_size_dense,
            ksz_tcn=kernel_size_tcn,
            tcn_repeats=tcn_repeats,
            tcn_blocks=tcn_blocks,
            tcn_channels=tcn_channels,
            no_decoder=no_decoder,
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

from ..nets import ConvTasNet
from .base import BaseHA
from .registry import HARegistry


@HARegistry.register("convtasnet")
class ConvTasNetHA(BaseHA):
    """Conv-TasNet hearing aid.

    Conv-TasNet was proposed in [1]. Multi-channel version proposed in [2].

    Note this multi-channel version is not the same as in [3] which was proposed for
    binaural-input-binaural-output speech separation.

    .. [1] Y. Luo and N. Mesgarani, "Conv-TasNet: Surpassing ideal time-frequency
       magnitude masking for speech separation", in IEEE/ACM Trans. Audio, Speech, Lang.
       Process., 2019.
    .. [2] J. Zhang, C. Zorila, R. Doddipatla and J. Barker, "On end-to-end
       multi-channel time domain speech separation in reverberant environments", in
       Proc. ICASSP, 2020.
    .. [3] C. Han, Y. Luo and N. Mesgarani, "Real-time binaural speech separation with
       preserved spatial cues", in Proc. ICASSP, 2020.
    """

    def __init__(
        self,
        input_channels=1,
        reference_channels=[0],
        filters=512,
        filter_length=32,
        bottleneck_channels=128,
        hidden_channels=512,
        skip_channels=128,
        kernel_size=3,
        layers=8,
        repeats=3,
        causal=False,
        fusion_layer=None,
        shared_fusion=True,
        emb_dim=None,
        optimizer="Adam",
        optimizer_kw={"lr": 1e-3},
        loss="snr",
        loss_kw=None,
        scheduler=None,
        scheduler_kw=None,
        grad_clip=5.0,
        spk_adapt_net=None,
        spk_adapt_net_kw=None,
        spk_adapt_stft_kw=None,
        wav_norm=None,
        norm_clean=True,
        audiogram=False,
    ):
        super().__init__()
        self.net = ConvTasNet(
            input_channels=input_channels,
            reference_channels=reference_channels,
            filters=filters,
            filter_length=filter_length,
            bottleneck_channels=bottleneck_channels,
            hidden_channels=hidden_channels,
            skip_channels=skip_channels,
            kernel_size=kernel_size,
            layers=layers,
            repeats=repeats,
            causal=causal,
            fusion_layer=fusion_layer,
            shared_fusion=shared_fusion,
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
            spk_adapt_net=spk_adapt_net,
            spk_adapt_net_kw=spk_adapt_net_kw,
            spk_adapt_stft_kw=spk_adapt_stft_kw,
            wav_norm=wav_norm,
            norm_clean=norm_clean,
            audiogram=audiogram,
        )

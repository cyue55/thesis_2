from ..nets import TFGridNet
from .base import BaseHA
from .registry import HARegistry


@HARegistry.register("tfgridnet")
class TFGridNetHA(BaseHA):
    """TF-GridNet hearing aid.

    TF-GridNet was proposed in [1] and [2].

    .. [1] Z.-Q. Wang, S. Cornell, S. Choi, Y. Lee, B.-Y. Kim and S. Watanabe,
       "TF-GridNet: Integrating Full- and Sub-Band Modeling for Speech Separation", in
       IEEE/ACM Trans. Audio, Speech, Lang. Process., 2023.
    .. [2] Z.-Q. Wang, S. Cornell, S. Choi, Y. Lee, B.-Y. Kim and S. Watanabe,
       "TF-GridNet: Making Time-Frequency Domain Models Great Again for Monaural Speaker
       Separation", in Proc. ICASSP, 2023.
    """

    def __init__(
        self,
        output_channels=1,
        input_channels=1,
        layers=6,
        lstm_hidden_units=128,
        attn_heads=4,
        attn_approx_qk_dim=512,
        _emb_dim=32,  # internal emb
        _emb_ks=4,
        _emb_hs=4,
        activation="PReLU",
        eps=1e-5,
        emb_dim=None,  # external emb e.g. speaker emb or audiogram
        optimizer="Adam",
        optimizer_kw={"lr": 1e-3},
        loss="snr",
        loss_kw=None,
        scheduler="ReduceLROnPlateau",
        scheduler_kw={"mode": "min", "factor": 0.5, "patience": 3},
        grad_clip=5.0,
        stft_kw={"frame_length": 512, "hop_length": 128},
        stft_future_frames=0,
        spk_adapt_net=None,
        spk_adapt_net_kw=None,
        spk_adapt_stft_kw=None,
        wav_norm="rms",
        norm_clean=True,
        audiogram=False,
    ):
        super().__init__()
        self.net = TFGridNet(
            output_channels=output_channels,
            input_channels=input_channels,
            n_fft=stft_kw.get("n_fft") or stft_kw["frame_length"],
            layers=layers,
            lstm_hidden_units=lstm_hidden_units,
            attn_heads=attn_heads,
            attn_approx_qk_dim=attn_approx_qk_dim,
            _emb_dim=_emb_dim,
            _emb_ks=_emb_ks,
            _emb_hs=_emb_hs,
            activation=activation,
            eps=eps,
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

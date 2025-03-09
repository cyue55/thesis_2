from ..nets import BSRNN
from .base import BaseHA
from .registry import HARegistry


@HARegistry.register("bsrnn")
class BSRNNHA(BaseHA):
    """Band-Split RNN (BSRNN) hearing aid.

    BSRNN was proposed in [1], [2] and [3]. This implementations includes the residual
    spectrogram proposed in [3] in addition to the mask.

    .. [1] Y. Luo and J. Yu, "Music source separation with band-split RNN", in
       IEEE/ACM Trans. Audio, Speech, Lang. Process., 2023.
    .. [2] J. Yu and Y. Luo, "Efficient monaural speech enhancement with universal
       sample rate band-split RNN", in Proc. ICASSP, 2023.
    .. [3] J. Yu, H. Chen, Y. Luo, R. Gu and C. Weng, "High fidelity speech
       enhancement with band-split RNN", in Proc. INTERSPEECH, 2023.
    """

    def __init__(
        self,
        input_channels=1,
        reference_channels=[0],
        fs=16000,
        base_channels=64,
        layers=6,
        causal=True,
        subband_right_limits=None,
        emb_dim=None,
        aggregate=False,
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
        _labels_include_clean=True,
        _labels_include_noisy=False,
    ):
        super().__init__()
        self.net = BSRNN(
            input_channels=input_channels,
            reference_channels=reference_channels,
            n_fft=stft_kw.get("n_fft") or stft_kw["frame_length"],
            fs=fs,
            base_channels=base_channels,
            layers=layers,
            causal=causal,
            subband_right_limits=subband_right_limits,
            emb_dim=emb_dim,
            aggregate=aggregate,
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
            _labels_include_clean=_labels_include_clean,
            _labels_include_noisy=_labels_include_noisy,
        )

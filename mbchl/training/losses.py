import warnings
from typing import override

import torch
import torch.nn as nn

from ..signal.auditory import AuditoryModel
from ..signal.stft import STFT
from ..utils import Registry, apply_mask, snr

LossRegistry = Registry("loss")


class BaseLoss(nn.Module):
    """Base class for all losses.

    Losses are always computed along the last dimension of the input tensors. If there
    are remaining dimensions other that the batch dimension after reducing the last
    dimension, then these are reduced by taking the mean.

    Subclasses must implement the :meth:`compute` method.

    """

    def forward(self, x, y, lengths, weight=None, audiogram=None):
        """Compute the loss.

        Performs checks before calling the :meth:`compute` method.

        Parameters
        ----------
        x : torch.Tensor
            Predictions. Shape ``(batch_size, ..., time)``.
        y : torch.Tensor or float
            Targets. Can be a float or a tensor with shape ``(batch_size, ..., time)``.
        lengths : torch.Tensor
            Length of tensors along last axis before batching. Shape ``(batch_size,)``.
        weight : torch.Tensor, optional
            Weight for each tensor. Shape ``(batch_size,)``.
        audiogram : torch.Tensor, optional
            Audiogram.

        Returns
        -------
        torch.Tensor
            Loss. Shape ``(batch_size,)``.

        """
        if isinstance(y, float):
            y = torch.full_like(x, y)
        assert x.shape == y.shape
        assert x.ndim >= 2
        return self.compute(x, y, lengths, weight=weight, audiogram=audiogram)

    def compute(self, x, y, lengths, weight=None, audiogram=None):
        """Compute the loss.

        Parameters
        ----------
        x : torch.Tensor
            Predictions. Shape ``(batch_size, ..., length)``.
        y : torch.Tensor
            Targets. Shape ``(batch_size, ..., length)``.
        lengths : torch.Tensor
            Length of tensors along last axis before batching. Shape ``(batch_size,)``.
        weight : torch.Tensor, optional
            Weight for each tensor. Shape ``(batch_size,)``.
        audiogram : torch.Tensor, optional
            Audiogram.

        Returns
        -------
        torch.Tensor
            Loss. Shape ``(batch_size,)``.

        """
        raise NotImplementedError


class BaseSNRLoss(BaseLoss):
    """Base class for :class:`SNRLoss` and :class:`SISNRLoss`.

    Parameters
    ----------
    scale_invariant : bool, optional
        Whether to make the loss scale-invariant.
    zero_mean : bool, optional
        Whether to subtract the mean to the signals before calculating the loss.
    eps : float, optional
        Small value to avoid division by zero.

    """

    def __init__(self, scale_invariant=False, zero_mean=True, eps=1e-7):
        super().__init__()
        self.scale_invariant = scale_invariant
        self.zero_mean = zero_mean
        self.eps = eps

    @override
    def compute(self, x, y, lengths, weight=None, audiogram=None):
        assert (
            weight is None
        ), f"weight argument not supported for {self.__class__.__name__} loss"
        assert (
            audiogram is None
        ), f"audiogram argument not supported for {self.__class__.__name__} loss"
        return -snr(
            x,
            y,
            scale_invariant=self.scale_invariant,
            zero_mean=self.zero_mean,
            eps=self.eps,
            lengths=lengths,
        )


@LossRegistry.register("snr")
class SNRLoss(BaseSNRLoss):
    """Signal-to-noise ratio (SNR).

    Parameters
    ----------
    zero_mean : bool, optional
        Whether to subtract the mean to the signals before calculating the loss.
    eps : float, optional
        Small value to avoid division by zero.

    """

    def __init__(self, zero_mean=True, eps=1e-7):
        super().__init__(scale_invariant=False, zero_mean=zero_mean, eps=eps)


@LossRegistry.register("sisnr")
class SISNRLoss(BaseSNRLoss):
    """Scale-invariant signal-to-noise ratio (SI-SNR).

    Parameters
    ----------
    zero_mean : bool, optional
        Whether to subtract the mean to the signals before calculating the loss.
    eps : float, optional
        Small value to avoid division by zero.

    """

    def __init__(self, zero_mean=True, eps=1e-7):
        super().__init__(scale_invariant=True, zero_mean=zero_mean, eps=eps)


@LossRegistry.register("pnorm")
class PNormLoss(BaseLoss):
    r"""P-norm loss.

    Calculated as

    .. math::

        \frac{1}{N} \left( \sum_{i=1}^{N} |x_i - y_i|^p \right)^{1/p}.

    Parameters
    ----------
    order : int, optional
        Order of the norm :math:`p`.
    norm : bool, optional
        If ``False``, then the :math:`1/p` exponent is not applied.

    """

    def __init__(self, order=2, norm=True):
        super().__init__()
        self.order = order
        self.norm = norm

    @override
    def compute(self, x, y, lengths, weight=None, audiogram=None):
        assert (
            audiogram is None
        ), f"audiogram argument not supported for {self.__class__.__name__} loss"
        x, y = apply_mask(x, y, lengths=lengths)
        loss = (x - y).abs().pow(self.order).sum(-1)
        if self.norm:
            loss = loss.pow(1 / self.order)
        loss /= lengths.view(-1, *[1] * (x.ndim - 2))
        if weight is not None:
            loss *= weight.view(-1, *[1] * (x.ndim - 2))
        dims = tuple(range(1, x.ndim - 1))
        if dims:
            loss = loss.mean(dims)
        return loss


@LossRegistry.register("l1")
class L1Loss(PNormLoss):
    """L1 loss."""

    def __init__(self):
        super().__init__(order=1)


@LossRegistry.register("l2")
class L2Loss(PNormLoss):
    """L2 loss."""

    def __init__(self):
        super().__init__(order=2)


@LossRegistry.register("mse")
class MSELoss(PNormLoss):
    """Mean squared error (MSE) loss."""

    def __init__(self):
        super().__init__(order=2, norm=False)


@LossRegistry.register("normpnorm")
class NormalizedPNormLoss(PNormLoss):
    """Normalized P-norm loss.

    The P-norm loss is normalized by the P-norm of the target.

    Parameters
    ----------
    order : int, optional
        Order of the norm :math:`p`.
    norm : bool, optional
        If ``False``, then the :math:`1/p` exponent is not applied.
    eps : float, optional
        Small value to avoid division by zero.

    """

    def __init__(self, order=2, norm=True, eps=1e-7):
        super().__init__(order=order, norm=norm)
        self.eps = eps

    @override
    def compute(self, x, y, lengths, weight=None, audiogram=None):
        assert (
            audiogram is None
        ), f"audiogram argument not supported for {self.__class__.__name__} loss"
        loss = super().compute(x, y, lengths, weight=weight)
        norm = super().compute(y, 0, lengths)  # do not weight normalization factor
        return loss / (norm + self.eps)


@LossRegistry.register("norml1")
class NormalizedL1Loss(NormalizedPNormLoss):
    """Normalized L1 loss."""

    def __init__(self, eps=1e-7):
        super().__init__(order=1, eps=eps)


@LossRegistry.register("norml2")
class NormalizedL2Loss(NormalizedPNormLoss):
    """Normalized L2 loss."""

    def __init__(self, eps=1e-7):
        super().__init__(order=2, eps=eps)


@LossRegistry.register("normmse")
class NormalizedMSELoss(NormalizedPNormLoss):
    """Normalized mean squared error (MSE) loss."""

    def __init__(self, eps=1e-7):
        super().__init__(order=2, norm=False, eps=eps)


@LossRegistry.register("mss")
class MultiScaleSpectralLoss(BaseLoss):
    """Multi-scale spectral loss (MSS).

    A great overview of multi scale spectral losses can be found in [1].

    .. [1] S. Schwär and M. Müller, "Multi-Scale Spectral Loss Revisited", in IEEE
       Signal Processing Letters, 2023.

    Parameters
    ----------
    frame_lengths : list[int], optional
        STFT frame lengths.
    hop_lengths : list[int], optional
        STFT hop lengths. If ``None``, defaults to half the frame lengths.
    window : str, optional
        STFT window. If ``None``, a rectangular window is used.
    order : int, optional
        Loss order.
    norm : bool, optional
        Whether to make the loss a norm.
    log : bool, optional
        Whether to log-compress the STFT magnitude.
    eps : float, optional
        Small value to avoid division by zero.

    """

    def __init__(
        self,
        frame_lengths=[512],
        hop_lengths=None,
        window="hann",
        order=2,
        norm=True,
        log=False,
        eps=1e-7,
    ):
        super().__init__()
        if hop_lengths is None:
            hop_lengths = [x // 2 for x in frame_lengths]
        self.stfts = [
            STFT(frame_length=frame_length, hop_length=hop_length, window=None)
            for frame_length, hop_length in zip(frame_lengths, hop_lengths)
        ]
        self.order = order
        self.log = log
        self.eps = eps
        self.pnorm = PNormLoss(order=order, norm=norm)

    @override
    def compute(self, x, y, lengths, weight=None, audiogram=None):
        assert (
            weight is None
        ), f"weight argument not supported for {self.__class__.__name__} loss"
        assert (
            audiogram is None
        ), f"audiogram argument not supported for {self.__class__.__name__} loss"
        x, y = apply_mask(x, y, lengths=lengths)
        output = 0
        for stft in self.stfts:
            x_mag = stft(x).abs()
            y_mag = stft(y).abs()
            if self.log:
                x_mag = torch.log(x_mag + self.eps)
                y_mag = torch.log(y_mag + self.eps)
            output += self.pnorm(x_mag, y_mag, lengths)
        return output / len(self.stfts)


@LossRegistry.register("mssl1td")
class MSSL1TD(BaseLoss):
    """Multi-scale spectral loss + L1 time-domain loss (MSS-L1TD).

    Proposed in [1].

    .. [1] Y.-J. Lu, S. Cornell, X. Chang, W. Zhang, C. Li, Z. Ni, Z.-Q. Wang and S.
       Watanabe, "Towards Low-Distortion Multi-Channel Speech Enhancement: The ESPNET-SE
       Submission to the L3DAS22 Challenge", in Proc. ICASSP, 2022.

    Parameters
    ----------
    frame_lengths : list[int], optional
        STFT frame lengths.
    hop_lengths : list[int], optional
        STFT hop lengths. If ``None``, defaults to half the frame lengths.
    window : str, optional
        STFT window. If ``None``, a rectangular window is used.
    order : int, optional
        Order for the MSS loss.
    norm : bool, optional
        Whether to make the MSS loss a norm.
    log : bool, optional
        Whether to log-compress the STFT magnitude in the MSS loss.
    time_domain_weight : float, optional
        Weight for the time-domain loss.
    spectral_weight : float, optional
        Weight for the spectral loss.
    scale_invariant : bool, optional
        Whether to make the loss scale-invariant.
    eps : float, optional
        Small value to avoid division by zero.

    """

    def __init__(
        self,
        frame_lengths=[512],
        hop_lengths=None,
        window=None,
        order=1,
        norm=True,
        log=False,
        time_domain_weight=1.0,
        spectral_weight=1.0,
        scale_invariant=True,
        eps=1e-7,
    ):
        super().__init__()
        if hop_lengths is None:
            hop_lengths = [x // 2 for x in frame_lengths]
        self.l1 = L1Loss()
        self.mss = MultiScaleSpectralLoss(
            frame_lengths=frame_lengths,
            hop_lengths=hop_lengths,
            window=window,
            order=order,
            norm=norm,
            log=log,
            eps=eps,
        )
        self.time_domain_weight = time_domain_weight
        self.spectral_weight = spectral_weight
        self.scale_invariant = scale_invariant
        self.eps = eps

    @override
    def compute(self, x, y, lengths, weight=None, audiogram=None):
        assert (
            weight is None
        ), f"weight argument not supported for {self.__class__.__name__} loss"
        assert (
            audiogram is None
        ), f"audiogram argument not supported for {self.__class__.__name__} loss"
        x, y = apply_mask(x, y, lengths=lengths)
        if self.scale_invariant:
            alpha = (x * y).sum(-1, keepdim=True) / (
                x.pow(2).sum(-1, keepdim=True) + self.eps
            )
            x = alpha * x
        time_domain_loss = self.time_domain_weight * self.l1(x, y, lengths)
        spectral_loss = self.spectral_weight * self.mss(x, y, lengths)
        return time_domain_loss + spectral_loss


@LossRegistry.register("auditory")
class AuditoryLoss(BaseLoss):
    """Auditory model-based loss.

    Parameters
    ----------
    am_kw : dict, optional
        Keyword arguments for the auditory model.
    am_kw_enhanced : dict, optional
        Keyword arguments for the auditory model for the enhanced signals. If ``None``,
        defaults to ``am_kw``.
    am_kw_clean : dict, optional
        Keyword arguments for the auditory model for the clean signals. If ``None``,
        defaults to ``am_kw``.
    loss : str, optional
        Loss function between the auditory model outputs.
    loss_kw : dict, optional
        Keyword arguments for the loss function.

    """

    def __init__(
        self,
        am_kw=None,
        am_kw_enhanced=None,
        am_kw_clean=None,
        loss="mse",
        loss_kw=None,
    ):
        super().__init__()
        self.am_enhanced = AuditoryModel(**(am_kw_enhanced or am_kw or {}))
        self.am_clean = AuditoryModel(**(am_kw_clean or am_kw or {}))
        self.loss = LossRegistry.get(loss)(**(loss_kw or {}))

    @override
    def compute(self, x, y, lengths, weight=None, audiogram=None):
        x = self.am_enhanced(x, audiogram=audiogram)
        y = self.am_clean(y, audiogram=None)
        return self.loss(x, y, lengths, weight=weight, audiogram=None)


@LossRegistry.register("cnrhlc")
class ControllableNoiseReductionHearingLossCompensationLoss(AuditoryLoss):
    """Controllable noise reduction and hearing loss compensation loss."""

    def __init__(
        self,
        am_kw=None,
        am_kw_hi=None,
        am_kw_nh=None,
        loss="mse",
        loss_kw=None,
        nh_denoising=True,
    ):
        super().__init__()
        self.am_hi = AuditoryModel(**(am_kw_hi or am_kw or {}))
        self.am_nh = AuditoryModel(**(am_kw_nh or am_kw or {}))
        self.loss = LossRegistry.get(loss)(**(loss_kw or {}))
        self.nh_denoising = nh_denoising
        self.log_uncertainty_denoising = nn.Parameter(torch.zeros(1))
        self.log_uncertainty_compensation = nn.Parameter(torch.zeros(1))
        self.denoising_loss = torch.tensor(0.0)
        self.compensation_loss = torch.tensor(0.0)


    @override
    def compute(self, x, y, lengths, weight=None, audiogram=None):
        assert x.ndim == y.ndim == 3  # (batch_size, n_channels, time)
        compensated, denoised = x.unbind(1)
        noisy, clean = y.unbind(1)
        if self.nh_denoising:
            self.denoising_loss = self.loss(
                self.am_nh(denoised, audiogram=None),
                self.am_nh(clean, audiogram=None),
                lengths,
                weight=weight,
                audiogram=None,
            )
        else:
            self.denoising_loss = self.loss(
                self.am_hi(denoised, audiogram=audiogram),
                self.am_hi(clean, audiogram=audiogram),
                lengths,
                weight=weight,
                audiogram=None,
            )
        self.compensation_loss = self.loss(
            self.am_hi(compensated, audiogram=audiogram),
            self.am_nh(noisy, audiogram=None),
            lengths,
            weight=weight,
            audiogram=None,
        )
        return (
            torch.exp(-self.log_uncertainty_denoising) * self.denoising_loss
            + torch.exp(-self.log_uncertainty_compensation) * self.compensation_loss
            + self.log_uncertainty_denoising
            + self.log_uncertainty_compensation
        )

@LossRegistry.register("cnrhlc-c2-tg2")
class ControllableNoiseReductionHearingLossCompensationLosseFixedWeights(AuditoryLoss):
    """Controllable noise reduction and hearing loss compensation loss."""

    def __init__(
        self,
        am_kw=None,
        am_kw_hi=None,
        am_kw_nh=None,
        loss="mse",
        loss_kw=None,
        nh_denoising=True,
        # Fixed weights for modulation losses
        alpha_mod_speech: float = 0.1, # Weight for modulation_speech_loss
        beta_mod_env: float = 0.1      # Weight for modulation_env_loss
    ):
        super().__init__()
        self.am_hi = AuditoryModel(**(am_kw_hi or am_kw or {}))
        self.am_nh = AuditoryModel(**(am_kw_nh or am_kw or {}))
        self.loss = LossRegistry.get(loss)(**(loss_kw or {}))
        self.nh_denoising = nh_denoising
        self.alpha_mod_speech = alpha_mod_speech
        self.beta_mod_env = beta_mod_env
        self.log_uncertainty_denoising = nn.Parameter(torch.zeros(1))
        self.log_uncertainty_compensation = nn.Parameter(torch.zeros(1))
        self.denoising_loss = torch.tensor(0.0)
        self.compensation_loss_base = torch.tensor(0.0)
        self.modulation_speech_loss = torch.tensor(0.0)
        self.modulation_env_loss = torch.tensor(0.0)
        self.augmented_compensation_loss = torch.tensor(0.0)

    @override
    def compute(self, x, y, lengths, weight=None, audiogram=None):
        assert x.ndim == y.ndim == 3  # (batch_size, n_channels, time)
        compensated, denoised = x.unbind(1)
        noisy, clean = y.unbind(1)
        if self.nh_denoising:
            am_nh_denoised, am_nh_denoised_speech_mod, am_nh_denoised_env_mod = (
                self.am_nh(denoised, audiogram=None)
            )
            am_nh_clean, am_nh_clean_speech_mod, am_nh_clean_env_mod = (
                self.am_nh(clean, audiogram=None)
            )
            self.denoising_loss = self.loss(
                am_nh_denoised,
                am_nh_clean,
                lengths,
                weight=weight,
                audiogram=None,
            )
        else:
            am_hi_denoised, am_hi_denoised_speech_mod, am_hi_denoised_env_mod = (
                self.am_hi(denoised, audiogram=audiogram)
            )
            am_hi_clean, am_hi_clean_speech_mod, am_hi_clean_env_mod = (
                self.am_hi(clean, audiogram=audiogram)
            )
            self.denoising_loss = self.loss(
                am_hi_denoised,
                am_hi_clean,
                lengths,
                weight=weight,
                audiogram=None,
            )
        am_hi_compensated, am_hi_compensated_speech_mod, am_hi_compensated_env_mod = (
            self.am_hi(compensated, audiogram=audiogram)
        )
        am_nh_noisy, am_nh_noisy_speech_mod, am_nh_noisy_env_mod = (
            self.am_nh(noisy, audiogram=None)
        )
        self.compensation_loss_base = self.loss(
            am_hi_compensated,
            am_nh_noisy,
            lengths,
            weight=weight,
            audiogram=None,
        )
        self.modulation_speech_loss = self.loss(
            am_hi_compensated_speech_mod,
            am_nh_noisy_speech_mod,
            lengths,
            weight=weight,
            audiogram=None,
        )
        self.modulation_env_loss = self.loss(
            am_hi_compensated_env_mod,
            am_nh_noisy_env_mod,
            lengths,
            weight=weight,
            audiogram=None,
        )
        self.augmented_compensation_loss = (
            self.compensation_loss_base
            + self.alpha_mod_speech * self.modulation_speech_loss
            + self.beta_mod_env * self.modulation_env_loss
        )
        return (
            torch.exp(-self.log_uncertainty_denoising) * self.denoising_loss
            + torch.exp(-self.log_uncertainty_compensation) * self.augmented_compensation_loss
            + self.log_uncertainty_denoising
            + self.log_uncertainty_compensation
        )

@LossRegistry.register("cnrhlc-c3a-tg2")
class ControllableNoiseReductionHearingLossCompensationLosseSpeechUncertainties(AuditoryLoss):
    """Controllable noise reduction and hearing loss compensation loss."""

    def __init__(
        self,
        am_kw=None,
        am_kw_hi=None,
        am_kw_nh=None,
        loss="mse",
        loss_kw=None,
        nh_denoising=True,
    ):
        super().__init__()
        self.am_hi = AuditoryModel(**(am_kw_hi or am_kw or {}))
        self.am_nh = AuditoryModel(**(am_kw_nh or am_kw or {}))
        self.loss = LossRegistry.get(loss)(**(loss_kw or {}))
        self.nh_denoising = nh_denoising
        self.log_uncertainty_denoising = nn.Parameter(torch.zeros(1))
        self.log_uncertainty_compensation = nn.Parameter(torch.zeros(1))
        self.log_uncertainty_modulation_speech = nn.Parameter(torch.zeros(1))
        self.denoising_loss = torch.tensor(0.0)
        self.compensation_loss = torch.tensor(0.0)
        self.modulation_speech_loss = torch.tensor(0.0)
    @override
    def compute(self, x, y, lengths, weight=None, audiogram=None):
        assert x.ndim == y.ndim == 3  # (batch_size, n_channels, time)
        compensated, denoised = x.unbind(1)
        noisy, clean = y.unbind(1)
        if self.nh_denoising:
            am_nh_denoised, am_nh_denoised_speech_mod, am_nh_denoised_env_mod = (
                self.am_nh(denoised, audiogram=None)
            )
            am_nh_clean, am_nh_clean_speech_mod, am_nh_clean_env_mod = (
                self.am_nh(clean, audiogram=None)
            )
            self.denoising_loss = self.loss(
                am_nh_denoised,
                am_nh_clean,
                lengths,
                weight=weight,
                audiogram=None,
            )
        else:
            am_hi_denoised, am_hi_denoised_speech_mod, am_hi_denoised_env_mod = (
                self.am_hi(denoised, audiogram=audiogram)
            )
            am_hi_clean, am_hi_clean_speech_mod, am_hi_clean_env_mod = (
                self.am_hi(clean, audiogram=audiogram)
            )
            self.denoising_loss = self.loss(
                am_hi_denoised,
                am_hi_clean,
                lengths,
                weight=weight,
                audiogram=None,
            )
        am_hi_compensated, am_hi_compensated_speech_mod, am_hi_compensated_env_mod = (
            self.am_hi(compensated, audiogram=audiogram)
        )
        am_nh_noisy, am_nh_noisy_speech_mod, am_nh_noisy_env_mod = (
            self.am_nh(noisy, audiogram=None)
        )
        self.compensation_loss = self.loss(
            am_hi_compensated,
            am_nh_noisy,
            lengths,
            weight=weight,
            audiogram=None,
        )
        self.modulation_speech_loss = self.loss(
            am_hi_compensated_speech_mod,
            am_nh_noisy_speech_mod,
            lengths,
            weight=weight,
            audiogram=None,
        )
        return (
            torch.exp(-self.log_uncertainty_denoising) * self.denoising_loss
            + torch.exp(-self.log_uncertainty_compensation) * self.compensation_loss
            + torch.exp(-self.log_uncertainty_modulation_speech) * self.modulation_speech_loss
            + self.log_uncertainty_denoising
            + self.log_uncertainty_compensation
            + self.log_uncertainty_modulation_speech
        )

@LossRegistry.register("cnrhlc-c3b-tg2")
class ControllableNoiseReductionHearingLossCompensationLosseEnvUncertainties(AuditoryLoss):
    """Controllable noise reduction and hearing loss compensation loss."""

    def __init__(
        self,
        am_kw=None,
        am_kw_hi=None,
        am_kw_nh=None,
        loss="mse",
        loss_kw=None,
        nh_denoising=True,
    ):
        super().__init__()
        self.am_hi = AuditoryModel(**(am_kw_hi or am_kw or {}))
        self.am_nh = AuditoryModel(**(am_kw_nh or am_kw or {}))
        self.loss = LossRegistry.get(loss)(**(loss_kw or {}))
        self.nh_denoising = nh_denoising
        self.log_uncertainty_denoising = nn.Parameter(torch.zeros(1))
        self.log_uncertainty_compensation = nn.Parameter(torch.zeros(1))
        self.log_uncertainty_modulation_env = nn.Parameter(torch.zeros(1))
        self.denoising_loss = torch.tensor(0.0)
        self.compensation_loss = torch.tensor(0.0)
        self.modulation_env_loss = torch.tensor(0.0)
    @override
    def compute(self, x, y, lengths, weight=None, audiogram=None):
        assert x.ndim == y.ndim == 3  # (batch_size, n_channels, time)
        compensated, denoised = x.unbind(1)
        noisy, clean = y.unbind(1)
        if self.nh_denoising:
            am_nh_denoised, am_nh_denoised_speech_mod, am_nh_denoised_env_mod = (
                self.am_nh(denoised, audiogram=None)
            )
            am_nh_clean, am_nh_clean_speech_mod, am_nh_clean_env_mod = (
                self.am_nh(clean, audiogram=None)
            )
            self.denoising_loss = self.loss(
                am_nh_denoised,
                am_nh_clean,
                lengths,
                weight=weight,
                audiogram=None,
            )
        else:
            am_hi_denoised, am_hi_denoised_speech_mod, am_hi_denoised_env_mod = (
                self.am_hi(denoised, audiogram=audiogram)
            )
            am_hi_clean, am_hi_clean_speech_mod, am_hi_clean_env_mod = (
                self.am_hi(clean, audiogram=audiogram)
            )
            self.denoising_loss = self.loss(
                am_hi_denoised,
                am_hi_clean,
                lengths,
                weight=weight,
                audiogram=None,
            )
        am_hi_compensated, am_hi_compensated_speech_mod, am_hi_compensated_env_mod = (
            self.am_hi(compensated, audiogram=audiogram)
        )
        am_nh_noisy, am_nh_noisy_speech_mod, am_nh_noisy_env_mod = (
            self.am_nh(noisy, audiogram=None)
        )
        self.compensation_loss = self.loss(
            am_hi_compensated,
            am_nh_noisy,
            lengths,
            weight=weight,
            audiogram=None,
        )
        self.modulation_env_loss = self.loss(
            am_hi_compensated_env_mod,
            am_nh_noisy_env_mod,
            lengths,
            weight=weight,
            audiogram=None,
        )
        return (
            torch.exp(-self.log_uncertainty_denoising) * self.denoising_loss
            + torch.exp(-self.log_uncertainty_compensation) * self.compensation_loss
            + torch.exp(-self.log_uncertainty_modulation_env) * self.modulation_env_loss
            + self.log_uncertainty_denoising
            + self.log_uncertainty_compensation
            + self.log_uncertainty_modulation_env
        )

@LossRegistry.register("cnrhlc-c4-tg2")
class ControllableNoiseReductionHearingLossCompensationLosseAllUncertainties(AuditoryLoss):
    """Controllable noise reduction and hearing loss compensation loss."""

    def __init__(
        self,
        am_kw=None,
        am_kw_hi=None,
        am_kw_nh=None,
        loss="mse",
        loss_kw=None,
        nh_denoising=True,
    ):
        super().__init__()
        self.am_hi = AuditoryModel(**(am_kw_hi or am_kw or {}))
        self.am_nh = AuditoryModel(**(am_kw_nh or am_kw or {}))
        self.loss = LossRegistry.get(loss)(**(loss_kw or {}))
        self.nh_denoising = nh_denoising
        self.log_uncertainty_denoising = nn.Parameter(torch.zeros(1))
        self.log_uncertainty_compensation = nn.Parameter(torch.zeros(1))
        self.log_uncertainty_modulation_speech = nn.Parameter(torch.zeros(1))
        self.log_uncertainty_modulation_env = nn.Parameter(torch.zeros(1))
        self.denoising_loss = torch.tensor(0.0)
        self.compensation_loss = torch.tensor(0.0)
        self.modulation_speech_loss = torch.tensor(0.0)
        self.modulation_env_loss = torch.tensor(0.0)
    @override
    def compute(self, x, y, lengths, weight=None, audiogram=None):
        assert x.ndim == y.ndim == 3  # (batch_size, n_channels, time)
        compensated, denoised = x.unbind(1)
        noisy, clean = y.unbind(1)
        if self.nh_denoising:
            am_nh_denoised, am_nh_denoised_speech_mod, am_nh_denoised_env_mod = (
                self.am_nh(denoised, audiogram=None)
            )
            am_nh_clean, am_nh_clean_speech_mod, am_nh_clean_env_mod = (
                self.am_nh(clean, audiogram=None)
            )
            self.denoising_loss = self.loss(
                am_nh_denoised,
                am_nh_clean,
                lengths,
                weight=weight,
                audiogram=None,
            )
        else:
            am_hi_denoised, am_hi_denoised_speech_mod, am_hi_denoised_env_mod = (
                self.am_hi(denoised, audiogram=audiogram)
            )
            am_hi_clean, am_hi_clean_speech_mod, am_hi_clean_env_mod = (
                self.am_hi(clean, audiogram=audiogram)
            )
            self.denoising_loss = self.loss(
                am_hi_denoised,
                am_hi_clean,
                lengths,
                weight=weight,
                audiogram=None,
            )
        am_hi_compensated, am_hi_compensated_speech_mod, am_hi_compensated_env_mod = (
            self.am_hi(compensated, audiogram=audiogram)
        )
        am_nh_noisy, am_nh_noisy_speech_mod, am_nh_noisy_env_mod = (
            self.am_nh(noisy, audiogram=None)
        )
        self.compensation_loss = self.loss(
            am_hi_compensated,
            am_nh_noisy,
            lengths,
            weight=weight,
            audiogram=None,
        )
        self.modulation_speech_loss = self.loss(
            am_hi_compensated_speech_mod,
            am_nh_noisy_speech_mod,
            lengths,
            weight=weight,
            audiogram=None,
        )
        self.modulation_env_loss = self.loss(
            am_hi_compensated_env_mod,
            am_nh_noisy_env_mod,
            lengths,
            weight=weight,
            audiogram=None,
        )
        return (
            torch.exp(-self.log_uncertainty_denoising) * self.denoising_loss
            + torch.exp(-self.log_uncertainty_compensation) * self.compensation_loss
            + torch.exp(-self.log_uncertainty_modulation_speech) * self.modulation_speech_loss
            + torch.exp(-self.log_uncertainty_modulation_env) * self.modulation_env_loss
            + self.log_uncertainty_denoising
            + self.log_uncertainty_compensation
            + self.log_uncertainty_modulation_speech
            + self.log_uncertainty_modulation_env
        )

@LossRegistry.register("cnrhlc_phi_tg2") # New name reflecting the change and TG2
class LossPhilippeStyleTG2(AuditoryLoss):
    """
    Implements the CNR-HLC loss based on Professor Philippe's suggestion:
    All losses are computed on modulation domain features (speech & environment)
    output directly by the auditory models.
    Uncertainty weighting is applied to these four modulation-based loss components.
    Targets for HLC modulation losses follow TG2 specification.
    """

    def __init__(
        self,
        am_kw=None,
        am_kw_hi=None,
        am_kw_nh=None,
        loss="l1", # Base L1/L2 loss function for comparing features
        loss_kw=None,
        nh_denoising=True, # Determines if NR targets use NH or HI model for clean speech
    ):
        super().__init__()
        # IMPORTANT ASSUMPTION: AuditoryModel's forward will now return (speech_mod, env_mod)
        # when configured with a modulation filter.
        self.am_hi = AuditoryModel(**(am_kw_hi or am_kw or {}))
        self.am_nh = AuditoryModel(**(am_kw_nh or am_kw or {}))
        self.loss_fn = LossRegistry.get(loss)(**(loss_kw or {}))
        self.nh_denoising = nh_denoising

        # Four uncertainty parameters for the four modulation-domain loss components
        self.log_uncertainty_nr_speech = nn.Parameter(torch.zeros(1))
        self.log_uncertainty_nr_env = nn.Parameter(torch.zeros(1))
        self.log_uncertainty_hlc_speech = nn.Parameter(torch.zeros(1)) # Formerly _modulation_speech
        self.log_uncertainty_hlc_env = nn.Parameter(torch.zeros(1))    # Formerly _modulation_env

        self.loss_nr_speech = torch.tensor(0.0)
        self.loss_nr_env = torch.tensor(0.0)
        self.loss_hlc_speech = torch.tensor(0.0)
        self.loss_hlc_env = torch.tensor(0.0)
        # # For logging metrics
        # self.nr_speech_loss_metric = torch.tensor(0.0)
        # self.nr_env_loss_metric = torch.tensor(0.0)
        # self.hlc_speech_loss_metric = torch.tensor(0.0)
        # self.hlc_env_loss_metric = torch.tensor(0.0)

    @override
    def compute(self, x, y, lengths, weight=None, audiogram=None):
        assert x.ndim == y.ndim == 3, "Inputs x and y must be 3D tensors (batch_size, n_channels, time)"
        # x contains model outputs: (compensated_signal, denoised_signal)
        # y contains targets: (noisy_signal, clean_signal)
        compensated_signal, denoised_signal = x.unbind(1)
        noisy_signal, clean_signal = y.unbind(1)

        # --- Get Auditory Model Outputs (now expected to be speech_mod, env_mod tuples) ---

        # For Denoising Path (NR)
        if self.nh_denoising:
            # Process denoised model output with NH model
            _, pred_nr_speech_mod, pred_nr_env_mod = self.am_nh(denoised_signal, audiogram=None)
            # Process clean target with NH model
            _, target_nr_speech_mod, target_nr_env_mod = self.am_nh(clean_signal, audiogram=None)
        else: # Use HI model for denoising path if nh_denoising is False
            _, pred_nr_speech_mod, pred_nr_env_mod = self.am_hi(denoised_signal, audiogram=audiogram)
            _, target_nr_speech_mod, target_nr_env_mod = self.am_hi(clean_signal, audiogram=audiogram) # Target also via HI

        # For Hearing Loss Compensation Path (HLC) - TG2 Targets
        # Process compensated model output with HI model
        _, pred_hlc_speech_mod, pred_hlc_env_mod = self.am_hi(compensated_signal, audiogram=audiogram)

        # # Target for HLC Speech (TG2): Clean speech processed by HI model
        # _, target_hlc_speech_mod, _ = self.am_hi(clean_signal, audiogram=audiogram) # We only need speech_mod

        # Target for HLC Environment (TG2 uses TG0's env target): Noisy speech processed by NH model
        _, target_hlc_speech_mod, target_hlc_env_mod = self.am_nh(noisy_signal, audiogram=None) # We only need env_mod


        # --- Calculate the Four Modulation-Domain Losses ---

        self.loss_nr_speech = self.loss_fn(
            pred_nr_speech_mod,
            target_nr_speech_mod,
            lengths,
            weight=weight, # Weight might apply per sample, if so, it's fine
            audiogram=None,
        )
        self.loss_nr_env = self.loss_fn(
            pred_nr_env_mod,
            target_nr_env_mod,
            lengths,
            weight=weight,
            audiogram=None,
        )
        self.loss_hlc_speech = self.loss_fn(
            pred_hlc_speech_mod,
            target_hlc_speech_mod, # TG2 target
            lengths,
            weight=weight,
            audiogram=None,
        )
        self.loss_hlc_env = self.loss_fn(
            pred_hlc_env_mod,
            target_hlc_env_mod,    # TG2's env target (from TG0)
            lengths,
            weight=weight,
            audiogram=None,
        )
        # Store metrics for logging (batch averaged)
        # Assuming self.loss_fn returns per-batch-item loss before final mean reduction
        # self.nr_speech_loss_metric = loss_nr_speech.detach().mean()
        # self.nr_env_loss_metric = loss_nr_env.detach().mean()
        # self.hlc_speech_loss_metric = loss_hlc_speech.detach().mean()
        # self.hlc_env_loss_metric = loss_hlc_env.detach().mean()
        
        # --- Combine Losses with Uncertainty Weighting ---
        total_loss_unreduced = (
            torch.exp(-self.log_uncertainty_nr_speech) * self.loss_nr_speech
            + torch.exp(-self.log_uncertainty_nr_env) * self.loss_nr_env
            + torch.exp(-self.log_uncertainty_hlc_speech) * self.loss_hlc_speech
            + torch.exp(-self.log_uncertainty_hlc_env) * self.loss_hlc_env
            + self.log_uncertainty_nr_speech
            + self.log_uncertainty_nr_env
            + self.log_uncertainty_hlc_speech
            + self.log_uncertainty_hlc_env
        )
        
        return total_loss_unreduced # Final scalar loss
    
@LossRegistry.register("cnrhlc-c6-tg2")
class ControllableNoiseReductionHearingLossCompensationLosseSixUncertainties(AuditoryLoss):
    """Controllable noise reduction and hearing loss compensation loss."""

    def __init__(
        self,
        am_kw=None,
        am_kw_hi=None,
        am_kw_nh=None,
        loss="mse",
        loss_kw=None,
        nh_denoising=True,
    ):
        super().__init__()
        self.am_hi = AuditoryModel(**(am_kw_hi or am_kw or {}))
        self.am_nh = AuditoryModel(**(am_kw_nh or am_kw or {}))
        self.loss = LossRegistry.get(loss)(**(loss_kw or {}))
        self.nh_denoising = nh_denoising
        self.log_uncertainty_denoising = nn.Parameter(torch.zeros(1))
        self.log_uncertainty_compensation = nn.Parameter(torch.zeros(1))
        self.log_uncertainty_nr_speech = nn.Parameter(torch.zeros(1))
        self.log_uncertainty_nr_env = nn.Parameter(torch.zeros(1))
        self.log_uncertainty_hlc_speech = nn.Parameter(torch.zeros(1)) # Formerly _modulation_speech
        self.log_uncertainty_hlc_env = nn.Parameter(torch.zeros(1))    # Formerly _modulation_env
        self.denoising_loss = torch.tensor(0.0)
        self.compensation_loss = torch.tensor(0.0)
        self.loss_nr_speech = torch.tensor(0.0)
        self.loss_nr_env = torch.tensor(0.0)
        self.loss_hlc_speech = torch.tensor(0.0)
        self.loss_hlc_env = torch.tensor(0.0)
    @override
    def compute(self, x, y, lengths, weight=None, audiogram=None):
        assert x.ndim == y.ndim == 3  # (batch_size, n_channels, time)
        compensated, denoised = x.unbind(1)
        noisy, clean = y.unbind(1)
        if self.nh_denoising:
            am_denoised, am_denoised_speech_mod, am_denoised_env_mod = (
                self.am_nh(denoised, audiogram=None)
            )
            am_clean, am_clean_speech_mod, am_clean_env_mod = (
                self.am_nh(clean, audiogram=None)
            )
            self.denoising_loss = self.loss(
                am_denoised,
                am_clean,
                lengths,
                weight=weight,
                audiogram=None,
            )
        else:
            am_denoised, am_denoised_speech_mod, am_denoised_env_mod = (
                self.am_hi(denoised, audiogram=audiogram)
            )
            am_clean, am_clean_speech_mod, am_clean_env_mod = (
                self.am_hi(clean, audiogram=audiogram)
            )
            self.denoising_loss = self.loss(
                am_denoised,
                am_clean,
                lengths,
                weight=weight,
                audiogram=None,
            )
        am_hi_compensated, am_hi_compensated_speech_mod, am_hi_compensated_env_mod = (
            self.am_hi(compensated, audiogram=audiogram)
        )
        am_nh_noisy, am_nh_noisy_speech_mod, am_nh_noisy_env_mod = (
            self.am_nh(noisy, audiogram=None)
        )
        self.compensation_loss = self.loss(
            am_hi_compensated,
            am_nh_noisy,
            lengths,
            weight=weight,
            audiogram=None,
        )
        self.modulation_speech_loss = self.loss(
            am_hi_compensated_speech_mod,
            am_nh_noisy_speech_mod,
            lengths,
            weight=weight,
            audiogram=None,
        )
        self.modulation_env_loss = self.loss(
            am_hi_compensated_env_mod,
            am_nh_noisy_env_mod,
            lengths,
            weight=weight,
            audiogram=None,
        )

        self.loss_nr_speech = self.loss(
            am_denoised_speech_mod,
            am_clean_speech_mod,
            lengths,
            weight=weight, # Weight might apply per sample, if so, it's fine
            audiogram=None,
        )
        self.loss_nr_env = self.loss(
            am_denoised_env_mod,
            am_clean_env_mod,
            lengths,
            weight=weight,
            audiogram=None,
        )
        self.loss_hlc_speech = self.loss(
            am_hi_compensated_speech_mod,
            am_nh_noisy_speech_mod, # TG2 target
            lengths,
            weight=weight,
            audiogram=None,
        )
        self.loss_hlc_env = self.loss(
            am_hi_compensated_env_mod,
            am_nh_noisy_env_mod,
            lengths,
            weight=weight,
            audiogram=None,
        )
        return (
            torch.exp(-self.log_uncertainty_denoising) * self.denoising_loss
            + torch.exp(-self.log_uncertainty_compensation) * self.compensation_loss
            + torch.exp(-self.log_uncertainty_nr_speech) * self.loss_nr_speech
            + torch.exp(-self.log_uncertainty_nr_env) * self.loss_nr_env
            + torch.exp(-self.log_uncertainty_hlc_speech) * self.loss_hlc_speech
            + torch.exp(-self.log_uncertainty_hlc_env) * self.loss_hlc_env
            + self.log_uncertainty_denoising
            + self.log_uncertainty_compensation
            + self.log_uncertainty_nr_speech
            + self.log_uncertainty_nr_env
            + self.log_uncertainty_hlc_speech
            + self.log_uncertainty_hlc_env
        )

@LossRegistry.register("cnrhlc-c4-tg2-old")
class ControllableNoiseReductionHearingLossCompensationLosseAllUncertaintiesOld(AuditoryLoss):
    """Controllable noise reduction and hearing loss compensation loss."""

    def __init__(
        self,
        am_kw=None,
        am_kw_hi=None,
        am_kw_nh=None,
        loss="mse",
        loss_kw=None,
        nh_denoising=True,
    ):
        super().__init__()
        self.am_hi = AuditoryModel(**(am_kw_hi or am_kw or {}))
        self.am_nh = AuditoryModel(**(am_kw_nh or am_kw or {}))
        self.loss = LossRegistry.get(loss)(**(loss_kw or {}))
        self.nh_denoising = nh_denoising
        self.denoising_loss = torch.tensor(0.0)
        self.compensation_loss = torch.tensor(0.0)
        self.modulation_speech_loss = torch.tensor(0.0)
        self.modulation_env_loss = torch.tensor(0.0)
        self.log_uncertainty_denoising = nn.Parameter(torch.zeros(1))
        self.log_uncertainty_compensation = nn.Parameter(torch.zeros(1))
        self.log_uncertainty_modulation_speech = nn.Parameter(torch.zeros(1))
        self.log_uncertainty_modulation_env = nn.Parameter(torch.zeros(1))
    @override
    def compute(self, x, y, lengths, weight=None, audiogram=None):
        assert x.ndim == y.ndim == 3  # (batch_size, n_channels, time)
        compensated, denoised = x.unbind(1)
        noisy, clean = y.unbind(1)
        if self.nh_denoising:
            am_nh_denoised, am_nh_denoised_speech_mod, am_nh_denoised_env_mod = (
                self.am_nh(denoised, audiogram=None)
            )
            am_nh_clean, am_nh_clean_speech_mod, am_nh_clean_env_mod = (
                self.am_nh(clean, audiogram=None)
            )
            self.denoising_loss = self.loss(
                am_nh_denoised,
                am_nh_clean,
                lengths,
                weight=weight,
                audiogram=None,
            )
        else:
            am_hi_denoised, am_hi_denoised_speech_mod, am_hi_denoised_env_mod = (
                self.am_hi(denoised, audiogram=audiogram)
            )
            am_hi_clean, am_hi_clean_speech_mod, am_hi_clean_env_mod = (
                self.am_hi(clean, audiogram=audiogram)
            )
            self.denoising_loss = self.loss(
                am_hi_denoised,
                am_hi_clean,
                lengths,
                weight=weight,
                audiogram=None,
            )
        am_hi_compensated, am_hi_compensated_speech_mod, am_hi_compensated_env_mod = (
            self.am_hi(compensated, audiogram=audiogram)
        )
        am_nh_noisy, am_nh_noisy_speech_mod, am_nh_noisy_env_mod = (
            self.am_nh(noisy, audiogram=None)
        )
        self.compensation_loss = self.loss(
            am_hi_compensated,
            am_nh_noisy,
            lengths,
            weight=weight,
            audiogram=None,
        )
        self.modulation_speech_loss = self.loss(
            am_hi_compensated_speech_mod,
            am_nh_clean_speech_mod, # 这里不太好，因为这里使用clean_speech数据他补偿了部分降噪，但是这里的目的是为了补偿，所以这里不太好
            lengths,
            weight=weight,
            audiogram=None,
        )
        self.modulation_env_loss = self.loss(
            am_hi_compensated_env_mod,
            am_nh_noisy_env_mod,
            lengths,
            weight=weight,
            audiogram=None,
        )
        return (
            torch.exp(-self.log_uncertainty_denoising) * self.denoising_loss
            + torch.exp(-self.log_uncertainty_compensation) * self.compensation_loss
            + torch.exp(-self.log_uncertainty_modulation_speech) * self.modulation_speech_loss
            + torch.exp(-self.log_uncertainty_modulation_env) * self.modulation_env_loss
            + self.log_uncertainty_denoising
            + self.log_uncertainty_compensation
            + self.log_uncertainty_modulation_speech
            + self.log_uncertainty_modulation_env
        )

@LossRegistry.register("cnrhlc-c7-tg2")
class ControllableNoiseReductionHearingLossCompensationLoss7(AuditoryLoss):
    """Controllable noise reduction and hearing loss compensation loss."""

    def __init__(
        self,
        am_kw=None,
        am_kw_hi=None,
        am_kw_nh=None,
        loss="mse",
        loss_kw=None,
        nh_denoising=True,
    ):
        super().__init__()
        self.am_hi = AuditoryModel(**(am_kw_hi or am_kw or {}))
        self.am_nh = AuditoryModel(**(am_kw_nh or am_kw or {}))
        self.loss = LossRegistry.get(loss)(**(loss_kw or {}))
        self.nh_denoising = nh_denoising
        self.denoising_loss = torch.tensor(0.0)
        self.compensation_loss = torch.tensor(0.0)
        self.modulation_speech_loss = torch.tensor(0.0)
        self.modulation_env_loss = torch.tensor(0.0)
        self.log_uncertainty_denoising = nn.Parameter(torch.zeros(1))
        self.log_uncertainty_compensation = nn.Parameter(torch.zeros(1))
        self.log_uncertainty_modulation_speech = nn.Parameter(torch.zeros(1))
        self.log_uncertainty_modulation_env = nn.Parameter(torch.zeros(1))
    @override
    def compute(self, x, y, lengths, weight=None, audiogram=None):
        assert x.ndim == y.ndim == 3  # (batch_size, n_channels, time)
        compensated, denoised = x.unbind(1)
        noisy, clean = y.unbind(1)
        if self.nh_denoising:
            am_nh_denoised, am_nh_denoised_speech_mod, am_nh_denoised_env_mod = (
                self.am_nh(denoised, audiogram=None)
            )
            am_nh_clean, am_nh_clean_speech_mod, am_nh_clean_env_mod = (
                self.am_nh(clean, audiogram=None)
            )
            self.denoising_loss = self.loss(
                am_nh_denoised,
                am_nh_clean,
                lengths,
                weight=weight,
                audiogram=None,
            )
        else:
            am_hi_denoised, am_hi_denoised_speech_mod, am_hi_denoised_env_mod = (
                self.am_hi(denoised, audiogram=audiogram)
            )
            am_hi_clean, am_hi_clean_speech_mod, am_hi_clean_env_mod = (
                self.am_hi(clean, audiogram=audiogram)
            )
            self.denoising_loss = self.loss(
                am_hi_denoised,
                am_hi_clean,
                lengths,
                weight=weight,
                audiogram=None,
            )
        am_hi_compensated, am_hi_compensated_speech_mod, am_hi_compensated_env_mod = (
            self.am_hi(compensated, audiogram=audiogram)
        )
        am_nh_noisy, am_nh_noisy_speech_mod, am_nh_noisy_env_mod = (
            self.am_nh(noisy, audiogram=None)
        )
        self.compensation_loss = self.loss(
            am_hi_compensated,
            am_nh_noisy,
            lengths,
            weight=weight,
            audiogram=None,
        )
        self.modulation_speech_loss = self.loss(
            am_hi_compensated_speech_mod,
            am_nh_clean_speech_mod,
            lengths,
            weight=weight,
            audiogram=None,
        )
        self.modulation_env_loss = self.loss(
            am_hi_compensated_env_mod,
            am_nh_clean_env_mod,
            lengths,
            weight=weight,
            audiogram=None,
        )
        return (
            torch.exp(-self.log_uncertainty_denoising) * self.denoising_loss
            + torch.exp(-self.log_uncertainty_compensation) * self.compensation_loss
            + torch.exp(-self.log_uncertainty_modulation_speech) * self.modulation_speech_loss
            + torch.exp(-self.log_uncertainty_modulation_env) * self.modulation_env_loss
            + self.log_uncertainty_denoising
            + self.log_uncertainty_compensation
            + self.log_uncertainty_modulation_speech
            + self.log_uncertainty_modulation_env
        )

@LossRegistry.register("cnrhlc-c8")
class CNRHLCLossC8(AuditoryLoss):
    """
    C8 strategy: Asymmetrical loss.
    NR_loss = holistic denoising loss
    HLC_loss = hlc_speech_loss + hlc_env_loss
    """

    def __init__(
        self,
        am_kw=None,
        am_kw_hi=None,
        am_kw_nh=None,
        loss="mse",
        loss_kw=None,
        nh_denoising=True,
    ):
        super().__init__()
        self.am_hi = AuditoryModel(**(am_kw_hi or am_kw or {}))
        self.am_nh = AuditoryModel(**(am_kw_nh or am_kw or {}))
        self.loss = LossRegistry.get(loss)(**(loss_kw or {}))
        self.nh_denoising = nh_denoising
        # C9: 只保留需要的三个不确定性参数
        self.log_uncertainty_denoising = nn.Parameter(torch.zeros(1))
        self.log_uncertainty_hlc_speech = nn.Parameter(torch.zeros(1))
        self.log_uncertainty_hlc_env = nn.Parameter(torch.zeros(1))
        # C9: 初始化对应的损失张量
        self.denoising_loss = torch.tensor(0.0)
        self.loss_hlc_speech = torch.tensor(0.0)
        self.loss_hlc_env = torch.tensor(0.0)

    @override
    def compute(self, x, y, lengths, weight=None, audiogram=None):
        assert x.ndim == y.ndim == 3  # (batch_size, n_channels, time)
        compensated, denoised = x.unbind(1)
        noisy, clean = y.unbind(1)

        # --- 数据准备部分 (与c6保持一致) ---
        if self.nh_denoising:
            am_denoised, _, _ = self.am_nh(denoised, audiogram=None)
            am_clean, _, _ = self.am_nh(clean, audiogram=None)
        else:
            am_denoised, _, _ = self.am_hi(denoised, audiogram=audiogram)
            am_clean, _, _ = self.am_hi(clean, audiogram=audiogram)

        _, am_hi_compensated_speech_mod, am_hi_compensated_env_mod = self.am_hi(
            compensated, audiogram=audiogram
        )
        _, am_nh_noisy_speech_mod, am_nh_noisy_env_mod = self.am_nh(noisy, audiogram=None)

        # --- C9 关键改动: 只计算需要的损失项 ---
        
        # 保留宏观NR损失
        self.denoising_loss = self.loss(
            am_denoised,
            am_clean,
            lengths,
            weight=weight,
            audiogram=None,
        )

        # 移除宏观HLC损失
        # self.compensation_loss = ... (移除)

        # 移除分解式NR损失
        # self.loss_nr_speech = ... (移除)
        # self.loss_nr_env = ... (移除)

        # 保留分解式HLC损失
        self.loss_hlc_speech = self.loss(
            am_hi_compensated_speech_mod,
            am_nh_noisy_speech_mod,
            lengths,
            weight=weight,
            audiogram=None,
        )
        self.loss_hlc_env = self.loss(
            am_hi_compensated_env_mod,
            am_nh_noisy_env_mod,
            lengths,
            weight=weight,
            audiogram=None,
        )

        # --- C9 关键改动: 更新最终的损失函数 ---
        return (
            torch.exp(-self.log_uncertainty_denoising) * self.denoising_loss
            + torch.exp(-self.log_uncertainty_hlc_speech) * self.loss_hlc_speech
            + torch.exp(-self.log_uncertainty_hlc_env) * self.loss_hlc_env
            + self.log_uncertainty_denoising
            + self.log_uncertainty_hlc_speech
            + self.log_uncertainty_hlc_env
        )
@LossRegistry.register("cnrhlc-c9")
class CNRHLCLossC9(AuditoryLoss):
    """
    C9 strategy: Inverse Asymmetrical loss.
    NR_loss = nr_speech_loss + nr_env_loss
    HLC_loss = holistic compensation loss
    """

    def __init__(
        self,
        am_kw=None,
        am_kw_hi=None,
        am_kw_nh=None,
        loss="mse",
        loss_kw=None,
        nh_denoising=True,
    ):
        super().__init__()
        self.am_hi = AuditoryModel(**(am_kw_hi or am_kw or {}))
        self.am_nh = AuditoryModel(**(am_kw_nh or am_kw or {}))
        self.loss = LossRegistry.get(loss)(**(loss_kw or {}))
        self.nh_denoising = nh_denoising
        # C9: 只保留需要的三个不确定性参数
        self.log_uncertainty_compensation = nn.Parameter(torch.zeros(1))
        self.log_uncertainty_nr_speech = nn.Parameter(torch.zeros(1))
        self.log_uncertainty_nr_env = nn.Parameter(torch.zeros(1))
        # C9: 初始化对应的损失张量
        self.compensation_loss = torch.tensor(0.0)
        self.loss_nr_speech = torch.tensor(0.0)
        self.loss_nr_env = torch.tensor(0.0)

    @override
    def compute(self, x, y, lengths, weight=None, audiogram=None):
        assert x.ndim == y.ndim == 3  # (batch_size, n_channels, time)
        compensated, denoised = x.unbind(1)
        noisy, clean = y.unbind(1)

        # --- 数据准备部分 (与c6保持一致) ---
        if self.nh_denoising:
            _, am_denoised_speech_mod, am_denoised_env_mod = self.am_nh(denoised, audiogram=None)
            _, am_clean_speech_mod, am_clean_env_mod = self.am_nh(clean, audiogram=None)
        else:
            _, am_denoised_speech_mod, am_denoised_env_mod = self.am_hi(denoised, audiogram=audiogram)
            _, am_clean_speech_mod, am_clean_env_mod = self.am_hi(clean, audiogram=audiogram)

        am_hi_compensated, _, _ = self.am_hi(compensated, audiogram=audiogram)
        am_nh_noisy, _, _ = self.am_nh(noisy, audiogram=None)

        # --- C10 关键改动: 只计算需要的损失项 ---

        # 移除宏观NR损失
        # self.denoising_loss = ... (移除)

        # 保留宏观HLC损失
        self.compensation_loss = self.loss(
            am_hi_compensated,
            am_nh_noisy,
            lengths,
            weight=weight,
            audiogram=None,
        )

        # 保留分解式NR损失
        self.loss_nr_speech = self.loss(
            am_denoised_speech_mod,
            am_clean_speech_mod,
            lengths,
            weight=weight,
            audiogram=None,
        )
        self.loss_nr_env = self.loss(
            am_denoised_env_mod,
            am_clean_env_mod,
            lengths,
            weight=weight,
            audiogram=None,
        )

        # 移除分解式HLC损失
        # self.loss_hlc_speech = ... (移除)
        # self.loss_hlc_env = ... (移除)

        # --- C10 关键改动: 更新最终的损失函数 ---
        return (
            torch.exp(-self.log_uncertainty_compensation) * self.compensation_loss
            + torch.exp(-self.log_uncertainty_nr_speech) * self.loss_nr_speech
            + torch.exp(-self.log_uncertainty_nr_env) * self.loss_nr_env
            + self.log_uncertainty_compensation
            + self.log_uncertainty_nr_speech
            + self.log_uncertainty_nr_env
        )

@LossRegistry.register("cnrhlc-c10")
class CNRHLCLossC10(AuditoryLoss):
    """
    C10 strategy: Redefining the ultimate goal for HLC.
    NR_loss = holistic denoising loss (target: clean)
    HLC_loss = holistic compensation loss (target: clean)
    This intentionally couples the two tasks to optimize for a single, ideal output.
    """

    def __init__(
        self,
        am_kw=None,
        am_kw_hi=None,
        am_kw_nh=None,
        loss="mse",
        loss_kw=None,
        nh_denoising=True,
    ):
        super().__init__()
        self.am_hi = AuditoryModel(**(am_kw_hi or am_kw or {}))
        self.am_nh = AuditoryModel(**(am_kw_nh or am_kw or {}))
        self.loss = LossRegistry.get(loss)(**(loss_kw or {}))
        self.nh_denoising = nh_denoising
        # C11: 只保留两个宏观整体损失的不确定性参数
        self.log_uncertainty_denoising = nn.Parameter(torch.zeros(1))
        self.log_uncertainty_compensation = nn.Parameter(torch.zeros(1))
        # C11: 初始化对应的损失张量
        self.denoising_loss = torch.tensor(0.0)
        self.compensation_loss = torch.tensor(0.0)

    @override
    def compute(self, x, y, lengths, weight=None, audiogram=None):
        assert x.ndim == y.ndim == 3  # (batch_size, n_channels, time)
        compensated, denoised = x.unbind(1)
        _, clean = y.unbind(1) # We don't need the noisy signal for this loss

        # --- 数据准备部分 (与c6保持一致，但只获取需要的特征) ---
        if self.nh_denoising:
            am_denoised, _, _ = self.am_nh(denoised, audiogram=None)
            am_clean, _, _ = self.am_nh(clean, audiogram=None)
        else:
            am_denoised, _, _ = self.am_hi(denoised, audiogram=audiogram)
            am_clean, _, _ = self.am_hi(clean, audiogram=audiogram)

        am_hi_compensated, _, _ = self.am_hi(compensated, audiogram=audiogram)

        # --- C10 关键改动: 只计算两个宏观损失 ---

        # 1. 计算标准的宏观NR损失
        self.denoising_loss = self.loss(
            am_denoised,
            am_clean,
            lengths,
            weight=weight,
            audiogram=None,
        )

        # 2. 计算以“干净语音”为目标的宏观HLC损失
        self.compensation_loss = self.loss(
            am_hi_compensated,
            am_clean, # C10 核心改动: HLC的目标不再是am_nh_noisy，而是am_clean
            lengths,
            weight=weight,
            audiogram=None,
        )

        # --- C10 关键改动: 更新最终的损失函数 ---
        return (
            torch.exp(-self.log_uncertainty_denoising) * self.denoising_loss
            + torch.exp(-self.log_uncertainty_compensation) * self.compensation_loss
            + self.log_uncertainty_denoising
            + self.log_uncertainty_compensation
        )
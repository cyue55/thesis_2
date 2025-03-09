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

    @override
    def compute(self, x, y, lengths, weight=None, audiogram=None):
        assert x.ndim == y.ndim == 3  # (batch_size, n_channels, time)
        compensated, denoised = x.unbind(1)
        noisy, clean = y.unbind(1)
        if self.nh_denoising:
            denoising_loss = self.loss(
                self.am_nh(denoised, audiogram=None),
                self.am_nh(clean, audiogram=None),
                lengths,
                weight=weight,
                audiogram=None,
            )
        else:
            denoising_loss = self.loss(
                self.am_hi(denoised, audiogram=audiogram),
                self.am_hi(clean, audiogram=audiogram),
                lengths,
                weight=weight,
                audiogram=None,
            )
        compensation_loss = self.loss(
            self.am_hi(compensated, audiogram=audiogram),
            self.am_nh(noisy, audiogram=None),
            lengths,
            weight=weight,
            audiogram=None,
        )
        return (
            torch.exp(-self.log_uncertainty_denoising) * denoising_loss
            + torch.exp(-self.log_uncertainty_compensation) * compensation_loss
            + self.log_uncertainty_denoising
            + self.log_uncertainty_compensation
        )

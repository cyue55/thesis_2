# Copyright (c) 2019 Yi Luo
# CC BY-NC-SA 3.0 US License
# https://github.com/naplab/Conv-TasNet

# Copyright (c) 2022 The PyClarity Team
# MIT License
# https://github.com/claritychallenge/

# Copyright (c) 2024 Philippe Gonzalez
# Apache License Version 2.0

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..layers import CausalLayerNorm, FusionLayerRegistry
from .registry import NetRegistry


@NetRegistry.register("convtasnet")
class ConvTasNet(nn.Module):
    """Conv-TasNet.

    Proposed in [1]. Multi-channel version proposed in [2].

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
        causal=True,
        fusion_layer=None,
        shared_fusion=True,
        emb_dim=None,
    ):
        super().__init__()
        assert 0 < len(reference_channels) <= input_channels
        assert all(i < input_channels for i in reference_channels)
        self.spectral_encoder = _Encoder(1, filters, filter_length)
        if input_channels > 1:
            self.spatial_encoder = _Encoder(input_channels, filters, filter_length)
        else:
            self.spatial_encoder = None
        self.decoder = _Decoder(filters, filter_length)
        self.tcn = _TCN(
            input_channels=2 * filters if input_channels > 1 else filters,
            output_channels=filters,
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
        self.reference_channels = reference_channels

    def forward(self, x, emb=None):
        """Forward pass."""
        b, c_in, n = x.shape
        c_out = len(self.reference_channels)
        x_ref = x[:, self.reference_channels, :]  # (b, c_out, n)
        x_ref = x_ref.reshape(-1, 1, n)  # (b*c_out, 1, n)
        x_ref = self.spectral_encoder(x_ref)  # (b*c_out, filters, n)
        if self.spatial_encoder is None:
            tcn_input = x_ref
        else:
            x_spatial = self.spatial_encoder(x)  # (b, filters, T)
            x_spatial = x_spatial.tile(c_out, 1, 1)  # (b*c_out, filters, n)
            tcn_input = torch.cat([x_ref, x_spatial], dim=1)  # (b*c_out, 2*filters, n)
        masks = self.tcn(tcn_input, emb=emb)  # (b*c_out, 1, filters, n)
        x_out = self.decoder(x_ref, masks)  # (b*c_out, 1, n)
        x_out = x_out.reshape(b, c_out, -1)
        return x_out[:, :, :n]


class _Encoder(nn.Module):
    def __init__(self, in_channels, filters, filter_length, aggregate=False):
        super().__init__()
        self.filter_length = filter_length
        self.stride = filter_length // 2
        self.conv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=filters,
            kernel_size=filter_length,
            stride=self.stride,
            bias=False,
        )
        self.aggregate = aggregate

    def forward(self, x):
        # pad right to get integer number of frames
        padding = (self.filter_length - x.shape[-1]) % self.stride
        x = F.pad(x, (0, padding))
        x = self.conv(x)
        if self.aggregate:
            x = x.mean(dim=-1)
        return x


class _Decoder(nn.Module):
    def __init__(self, filters, filter_length):
        super().__init__()
        self.filter_length = filter_length
        self.stride = filter_length // 2
        self.trans_conv = nn.ConvTranspose1d(
            in_channels=filters,
            out_channels=1,
            kernel_size=filter_length,
            stride=self.stride,
            bias=False,
        )

    def forward(self, x, masks):
        batch_size, sources, channels, length = masks.shape
        x = x.unsqueeze(1)
        x = x * masks
        x = x.view(batch_size * sources, channels, length)
        x = self.trans_conv(x)
        x = x.view(batch_size, sources, -1)
        return x


class _TCN(nn.Module):
    def __init__(
        self,
        input_channels,
        output_channels,
        bottleneck_channels,
        hidden_channels,
        skip_channels,
        kernel_size,
        layers,
        repeats,
        causal,
        fusion_layer,
        shared_fusion,
        emb_dim,
    ):
        super().__init__()
        if fusion_layer is not None and shared_fusion:
            fusion_layer_cls = FusionLayerRegistry.get(fusion_layer)
            fusion_layer = fusion_layer_cls(bottleneck_channels, emb_dim)
        self.layer_norm = _init_norm(causal, input_channels)
        self.bottleneck_conv = nn.Conv1d(
            in_channels=input_channels,
            out_channels=bottleneck_channels,
            kernel_size=1,
        )
        self.conv_blocks = nn.ModuleList()
        for b in range(repeats):
            for i in range(layers):
                dilation = 2**i
                last = b == repeats - 1 and i == layers - 1
                self.conv_blocks.append(
                    _Conv1DBlock(
                        input_channels=bottleneck_channels,
                        hidden_channels=hidden_channels,
                        skip_channels=skip_channels,
                        dilation=dilation,
                        kernel_size=kernel_size,
                        causal=causal,
                        last=last,
                        fusion_layer=fusion_layer,
                        emb_dim=emb_dim,
                    )
                )
        self.prelu = nn.PReLU()
        self.output_conv = nn.Conv1d(
            in_channels=skip_channels,
            out_channels=output_channels,
            kernel_size=1,
        )

    def forward(self, x, emb=None):
        batch_size, channels, length = x.shape
        x = self.layer_norm(x)
        x = self.bottleneck_conv(x)
        skip_sum = 0
        for conv_block in self.conv_blocks:
            x, skip = conv_block(x, emb)
            skip_sum += skip
        x = self.prelu(skip_sum)
        x = self.output_conv(x)
        x = torch.sigmoid(x)
        return x.unsqueeze(1)


class _Conv1DBlock(nn.Module):
    def __init__(
        self,
        input_channels,
        hidden_channels,
        skip_channels,
        kernel_size,
        dilation,
        causal,
        last,
        fusion_layer,
        emb_dim,
    ):
        super().__init__()
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.causal = causal
        self.conv = nn.Conv1d(
            in_channels=input_channels,
            out_channels=hidden_channels,
            kernel_size=1,
        )
        self.d_conv = nn.Conv1d(
            in_channels=hidden_channels,
            out_channels=hidden_channels,
            kernel_size=kernel_size,
            dilation=dilation,
            groups=hidden_channels,
        )
        # the output of the residual convolution in the last block is not used
        if last:
            self.res_conv = None
        else:
            self.res_conv = nn.Conv1d(
                in_channels=hidden_channels,
                out_channels=input_channels,
                kernel_size=1,
            )
        self.skip_conv = nn.Conv1d(
            in_channels=hidden_channels,
            out_channels=skip_channels,
            kernel_size=1,
        )
        self.norm_1 = _init_norm(causal, hidden_channels)
        self.norm_2 = _init_norm(causal, hidden_channels)
        self.prelu_1 = nn.PReLU()
        self.prelu_2 = nn.PReLU()
        if isinstance(fusion_layer, str):
            fusion_layer_cls = FusionLayerRegistry.get(fusion_layer)
            self.fusion = fusion_layer_cls(input_channels, emb_dim)
        else:
            self.fusion = fusion_layer

    def forward(self, input_, emb=None):
        assert not (emb is None and self.fusion is not None)
        assert not (emb is not None and self.fusion is None)
        if emb is not None and emb.ndim == 3:
            emb = emb.squeeze(1)
        x = input_ if emb is None else self.fusion(input_, emb, axis=-2)
        x = self.conv(x)
        x = self.prelu_1(x)
        x = self.norm_1(x)
        padding = (self.kernel_size - 1) * self.dilation
        if self.causal:
            padding_left = padding
            padding_right = 0
        else:
            padding_left = padding // 2
            padding_right = padding - padding_left
        x = F.pad(x, (padding_left, padding_right))
        x = self.d_conv(x)
        x = self.prelu_2(x)
        x = self.norm_2(x)
        if self.res_conv is None:
            output = None
        else:
            output = input_ + self.res_conv(x)
        skip = self.skip_conv(x)
        return output, skip


def _init_norm(causal, dim):
    if causal:
        module = CausalLayerNorm(num_channels=dim, time_dim=-1)
    else:
        module = nn.GroupNorm(num_channels=dim, num_groups=1)
    return module

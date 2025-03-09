# Copyright (c) 2017 Johns Hopkins University (Shinji Watanabe)
# Apache License 2.0
# https://github.com/espnet/espnet

import torch
import torch.nn as nn

from .registry import NetRegistry


@NetRegistry.register("tcndenseunet")
class TCNDenseUNet(nn.Module):
    """TCNDenseUNet.

    Proposed in [1] and [2].

    .. [1] Z.-Q. Wang, G. Wichern and J. Le Roux, "Leveraging Low-Distortion Target
       Estimates for Improved Speech Enhancement", arXiv preprint  arXiv:2110.00570,
       2021.
    .. [2] Y.-J. Lu, S. Cornell, X. Chang, W. Zhang, C. Li, Z. Ni, Z.-Q. Wang and S.
       Watanabe, "Towards Low-Distortion Multi-Channel Speech Enhancement: The ESPNET-SE
       Submission to the L3DAS22 Challenge", in Proc. ICASSP, 2022.
    """

    def __init__(
        self,
        n_spk=1,
        in_freqs=257,
        mic_channels=1,
        hid_chans=32,
        hid_chans_dense=32,
        ksz_dense=(3, 3),
        ksz_tcn=3,
        tcn_repeats=4,
        tcn_blocks=7,
        tcn_channels=384,
        activation=nn.ELU,
        no_decoder=False,
        emb_dim=None,
    ):
        super().__init__()
        self.n_spk = n_spk
        self.in_channels = in_freqs
        self.mic_channels = mic_channels
        self.no_decoder = no_decoder

        num_freqs = in_freqs - 2
        freq_axis_dims = self._get_depth(num_freqs)

        self.encoder = _Encoder(
            num_freqs,
            freq_axis_dims,
            mic_channels,
            hid_chans,
            hid_chans_dense,
            ksz_dense,
            tcn_channels,
            activation,
        )

        self.tcn = _TCN(
            ksz_tcn,
            tcn_repeats,
            tcn_blocks,
            tcn_channels,
            activation,
        )

        if not no_decoder:
            self.decoder = _Decoder(
                n_spk,
                num_freqs,
                freq_axis_dims,
                hid_chans,
                hid_chans_dense,
                ksz_dense,
                tcn_channels,
                activation,
            )

    def _get_depth(self, num_freq):
        n_layers = 0
        freqs = []
        while num_freq > 15:
            num_freq = int(num_freq / 2)
            freqs.append(num_freq)
            n_layers += 1
        return freqs

    def forward(self, x, emb=None):
        """Forward pass."""
        if emb is not None:
            raise NotImplementedError(
                f"Embedding not implemented for {self.__class__.__name__}"
            )

        # B, C, F, T
        bsz, mics, _, frames = x.shape
        assert mics == self.mic_channels

        x = x.permute(0, 1, 3, 2)  # B, C, T, F
        x = torch.cat((x.real, x.imag), 1)

        x, enc_out = self.encoder(x)
        x = self.tcn(x)

        if self.no_decoder:
            # squeeze freq axis and average over time to extract speaker embedding
            return x.squeeze(-1).mean(-1)

        x = self.decoder(x, enc_out)

        x = x.reshape(bsz, 2, self.n_spk, -1, self.in_channels)
        x = torch.complex(x[:, 0], x[:, 1])  # B, C, T, F
        return x.permute(0, 1, 3, 2)  # B, C, F, T


class _Encoder(nn.Module):
    def __init__(
        self,
        num_freqs,
        freq_axis_dims,
        mic_channels,
        hid_chans,
        hid_chans_dense,
        ksz_dense,
        tcn_channels,
        activation,
    ):
        super().__init__()

        first = nn.Sequential(
            nn.Conv2d(
                mic_channels * 2,
                hid_chans,
                (3, 3),
                (1, 1),
                (1, 0),
                padding_mode="reflect",
            ),
            _DenseBlock(
                hid_chans,
                hid_chans,
                num_freqs,
                ksz=ksz_dense,
                activation=activation,
                hid_chans=hid_chans_dense,
            ),
        )

        self.encoder = nn.ModuleList([])
        self.encoder.append(first)

        for layer_indx in range(len(freq_axis_dims)):
            downsample = _Conv2DActNorm(
                hid_chans, hid_chans, (3, 3), (1, 2), (1, 0), activation=activation
            )
            denseblocks = _DenseBlock(
                hid_chans,
                hid_chans,
                freq_axis_dims[layer_indx],
                ksz=ksz_dense,
                activation=activation,
                hid_chans=hid_chans_dense,
            )
            c_layer = nn.Sequential(downsample, denseblocks)
            self.encoder.append(c_layer)

        self.encoder.append(
            _Conv2DActNorm(
                hid_chans, hid_chans * 2, (3, 3), (1, 2), (1, 0), activation=activation
            )
        )
        self.encoder.append(
            _Conv2DActNorm(
                hid_chans * 2,
                hid_chans * 4,
                (3, 3),
                (1, 2),
                (1, 0),
                activation=activation,
            )
        )
        self.encoder.append(
            _Conv2DActNorm(
                hid_chans * 4,
                tcn_channels,
                (3, 3),
                (1, 1),
                (1, 0),
                activation=activation,
            )
        )

    def forward(self, x):
        enc_out = []
        for enc_layer in self.encoder:
            x = enc_layer(x)
            enc_out.append(x)
        return x, enc_out


class _Decoder(nn.Module):
    def __init__(
        self,
        n_spk,
        num_freqs,
        freq_axis_dims,
        hid_chans,
        hid_chans_dense,
        ksz_dense,
        tcn_channels,
        activation,
    ):
        super().__init__()

        self.decoder = nn.ModuleList([])
        self.decoder.append(
            _Conv2DActNorm(
                tcn_channels * 2,
                hid_chans * 4,
                (3, 3),
                (1, 1),
                (1, 0),
                activation=activation,
                upsample=True,
            )
        )
        self.decoder.append(
            _Conv2DActNorm(
                hid_chans * 8,
                hid_chans * 2,
                (3, 3),
                (1, 2),
                (1, 0),
                activation=activation,
                upsample=True,
            )
        )
        self.decoder.append(
            _Conv2DActNorm(
                hid_chans * 4,
                hid_chans,
                (3, 3),
                (1, 2),
                (1, 0),
                activation=activation,
                upsample=True,
            )
        )

        for dec_indx in range(len(freq_axis_dims)):
            c_num_freqs = freq_axis_dims[len(freq_axis_dims) - dec_indx - 1]
            denseblocks = _DenseBlock(
                hid_chans * 2,
                hid_chans * 2,
                c_num_freqs,
                ksz=ksz_dense,
                activation=activation,
                hid_chans=hid_chans_dense,
            )
            upsample = _Conv2DActNorm(
                hid_chans * 2,
                hid_chans,
                (3, 3),
                (1, 2),
                (1, 0),
                activation=activation,
                upsample=True,
            )
            c_layer = nn.Sequential(denseblocks, upsample)
            self.decoder.append(c_layer)

        last = nn.Sequential(
            _DenseBlock(
                hid_chans * 2,
                hid_chans * 2,
                num_freqs,
                ksz=ksz_dense,
                activation=activation,
                hid_chans=hid_chans_dense,
            ),
            nn.ConvTranspose2d(hid_chans * 2, 2 * n_spk, (3, 3), (1, 1), (1, 0)),
        )
        self.decoder.append(last)

    def forward(self, x, enc_out):
        for indx, dec_layer in enumerate(self.decoder):
            c_input = torch.cat((x, enc_out[-(indx + 1)]), 1)
            x = dec_layer(c_input)
        return x


class _TCN(nn.Module):
    def __init__(
        self,
        ksz_tcn,
        tcn_repeats,
        tcn_blocks,
        tcn_channels,
        activation,
    ):
        super().__init__()
        self.tcn = []
        for r in range(tcn_repeats):
            for x in range(tcn_blocks):
                self.tcn.append(
                    _TCNResBlock(
                        tcn_channels,
                        tcn_channels,
                        ksz_tcn,
                        dilation=2**x,
                        activation=activation,
                    )
                )
        self.tcn = nn.Sequential(*self.tcn)

    def forward(self, x):
        assert x.shape[-1] == 1
        return self.tcn(x.squeeze(-1)).unsqueeze(-1)


class _DenseBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        num_freqs,
        pre_blocks=2,
        freq_proc_blocks=1,
        post_blocks=2,
        ksz=(3, 3),
        activation=nn.ELU,
        hid_chans=32,
    ):
        super().__init__()

        assert post_blocks >= 1
        assert pre_blocks >= 1

        self.pre_blocks = nn.ModuleList([])
        tot_layers = 0
        for indx in range(pre_blocks):
            c_layer = _Conv2DActNorm(
                in_channels + hid_chans * tot_layers,
                hid_chans,
                ksz,
                (1, 1),
                (1, 1),
                activation=activation,
            )
            self.pre_blocks.append(c_layer)
            tot_layers += 1

        self.freq_proc_blocks = nn.ModuleList([])
        for indx in range(freq_proc_blocks):
            c_layer = _FreqWiseBlock(
                in_channels + hid_chans * tot_layers,
                num_freqs,
                hid_chans,
                activation=activation,
            )
            self.freq_proc_blocks.append(c_layer)
            tot_layers += 1

        self.post_blocks = nn.ModuleList([])
        for indx in range(post_blocks - 1):
            c_layer = _Conv2DActNorm(
                in_channels + hid_chans * tot_layers,
                hid_chans,
                ksz,
                (1, 1),
                (1, 1),
                activation=activation,
            )
            self.post_blocks.append(c_layer)
            tot_layers += 1

        last = _Conv2DActNorm(
            in_channels + hid_chans * tot_layers,
            out_channels,
            ksz,
            (1, 1),
            (1, 1),
            activation=activation,
        )
        self.post_blocks.append(last)

    def forward(self, x):
        # batch, channels, frames, freq
        out = [x]
        for pre_block in self.pre_blocks:
            c_out = pre_block(torch.cat(out, 1))
            out.append(c_out)
        for freq_block in self.freq_proc_blocks:
            c_out = freq_block(torch.cat(out, 1))
            out.append(c_out)
        for post_block in self.post_blocks:
            c_out = post_block(torch.cat(out, 1))
            out.append(c_out)
        return c_out


class _TCNResBlock(nn.Module):
    def __init__(
        self, in_chan, out_chan, ksz=3, stride=1, dilation=1, activation=nn.ELU
    ):
        super().__init__()
        padding = dilation
        dconv = nn.Conv1d(
            in_chan,
            in_chan,
            ksz,
            stride,
            padding=padding,
            dilation=dilation,
            padding_mode="reflect",
            groups=in_chan,
        )
        point_conv = nn.Conv1d(in_chan, out_chan, 1)

        self.layer = nn.Sequential(
            nn.GroupNorm(in_chan, in_chan, eps=1e-7),
            activation(),
            dconv,
            point_conv,
        )

    def forward(self, x):
        # batch, channels, frames
        return self.layer(x) + x


class _Conv2DActNorm(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        ksz=(3, 3),
        stride=(1, 2),
        padding=(1, 0),
        upsample=False,
        activation=nn.ELU,
    ):
        super().__init__()
        if upsample:
            conv = nn.ConvTranspose2d(in_channels, out_channels, ksz, stride, padding)
        else:
            conv = nn.Conv2d(
                in_channels, out_channels, ksz, stride, padding, padding_mode="reflect"
            )
        act = activation()
        norm = nn.GroupNorm(out_channels, out_channels, eps=1e-7)
        self.layer = nn.Sequential(conv, act, norm)

    def forward(self, x):
        return self.layer(x)


class _FreqWiseBlock(nn.Module):
    def __init__(self, in_channels, num_freqs, out_channels, activation=nn.ELU):
        super().__init__()
        self.bottleneck = _Conv2DActNorm(
            in_channels, out_channels, (1, 1), (1, 1), (0, 0), activation=activation
        )
        self.freq_proc = _Conv2DActNorm(
            num_freqs, num_freqs, (1, 1), (1, 1), (0, 0), activation=activation
        )

    def forward(self, x):
        out = self.freq_proc(self.bottleneck(x).permute(0, 3, 2, 1)).permute(0, 3, 2, 1)
        return out

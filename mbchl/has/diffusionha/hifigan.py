# Copyright (c) 2020 Jungil Kong
# MIT License
# https://github.com/jik876/hifi-gan

import json

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Conv1d, ConvTranspose1d
from torch.nn.utils import remove_weight_norm, weight_norm

from ...utils import AttrDict

LRELU_SLOPE = 0.1


def _init_weights(m, mean=0.0, std=0.01):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        m.weight.data.normal_(mean, std)


def _get_padding(kernel_size, dilation=1):
    return int((kernel_size * dilation - dilation) / 2)


class _ResBlock1(torch.nn.Module):
    def __init__(self, h, channels, kernel_size=3, dilation=(1, 3, 5)):
        super().__init__()
        self.h = h
        self.convs1 = nn.ModuleList(
            [
                weight_norm(
                    Conv1d(
                        channels,
                        channels,
                        kernel_size,
                        1,
                        dilation=dilation[0],
                        padding=_get_padding(kernel_size, dilation[0]),
                    )
                ),
                weight_norm(
                    Conv1d(
                        channels,
                        channels,
                        kernel_size,
                        1,
                        dilation=dilation[1],
                        padding=_get_padding(kernel_size, dilation[1]),
                    )
                ),
                weight_norm(
                    Conv1d(
                        channels,
                        channels,
                        kernel_size,
                        1,
                        dilation=dilation[2],
                        padding=_get_padding(kernel_size, dilation[2]),
                    )
                ),
            ]
        )
        self.convs1.apply(_init_weights)

        self.convs2 = nn.ModuleList(
            [
                weight_norm(
                    Conv1d(
                        channels,
                        channels,
                        kernel_size,
                        1,
                        dilation=1,
                        padding=_get_padding(kernel_size, 1),
                    )
                ),
                weight_norm(
                    Conv1d(
                        channels,
                        channels,
                        kernel_size,
                        1,
                        dilation=1,
                        padding=_get_padding(kernel_size, 1),
                    )
                ),
                weight_norm(
                    Conv1d(
                        channels,
                        channels,
                        kernel_size,
                        1,
                        dilation=1,
                        padding=_get_padding(kernel_size, 1),
                    )
                ),
            ]
        )
        self.convs2.apply(_init_weights)

    def forward(self, x):
        for c1, c2 in zip(self.convs1, self.convs2):
            xt = F.leaky_relu(x, LRELU_SLOPE)
            xt = c1(xt)
            xt = F.leaky_relu(xt, LRELU_SLOPE)
            xt = c2(xt)
            x = xt + x
        return x

    def remove_weight_norm(self):
        for l_ in self.convs1:
            remove_weight_norm(l_)
        for l_ in self.convs2:
            remove_weight_norm(l_)


class _ResBlock2(torch.nn.Module):
    def __init__(self, h, channels, kernel_size=3, dilation=(1, 3)):
        super().__init__()
        self.h = h
        self.convs = nn.ModuleList(
            [
                weight_norm(
                    Conv1d(
                        channels,
                        channels,
                        kernel_size,
                        1,
                        dilation=dilation[0],
                        padding=_get_padding(kernel_size, dilation[0]),
                    )
                ),
                weight_norm(
                    Conv1d(
                        channels,
                        channels,
                        kernel_size,
                        1,
                        dilation=dilation[1],
                        padding=_get_padding(kernel_size, dilation[1]),
                    )
                ),
            ]
        )
        self.convs.apply(_init_weights)

    def forward(self, x):
        for c in self.convs:
            xt = F.leaky_relu(x, LRELU_SLOPE)
            xt = c(xt)
            x = xt + x
        return x

    def remove_weight_norm(self):
        for l_ in self.convs:
            remove_weight_norm(l_)


class _Generator_HiFiRes(torch.nn.Module):
    def __init__(self, h):
        super().__init__()
        self.h = h
        self.num_kernels = len(h.resblock_kernel_sizes)
        self.num_upsamples = len(h.upsample_rates)
        self.conv_pre = weight_norm(
            Conv1d(256, h.upsample_initial_channel, 7, 1, padding=3)
        )
        resblock = _ResBlock1 if h.resblock == "1" else _ResBlock2

        self.ups = nn.ModuleList()
        for i, (u, k) in enumerate(zip(h.upsample_rates, h.upsample_kernel_sizes)):
            self.ups.append(
                weight_norm(
                    ConvTranspose1d(
                        h.upsample_initial_channel // (2**i),
                        h.upsample_initial_channel // (2 ** (i + 1)),
                        u * 2,
                        u,
                        padding=u // 2 + u % 2,
                        output_padding=u % 2,
                    )
                )
            )

        self.resblocks = nn.ModuleList()
        for i in range(len(self.ups)):
            ch = h.upsample_initial_channel // (2 ** (i + 1))
            for j, (k, d) in enumerate(
                zip(h.resblock_kernel_sizes, h.resblock_dilation_sizes)
            ):
                self.resblocks.append(resblock(h, ch, k, d))

        self.conv_post = weight_norm(Conv1d(ch, 1, 7, 1, padding=3))
        self.ups.apply(_init_weights)
        self.conv_post.apply(_init_weights)

    def forward(self, x):
        x = self.conv_pre(x)
        for i in range(self.num_upsamples):
            x = F.leaky_relu(x, LRELU_SLOPE)
            x = self.ups[i](x)
            xs = None
            for j in range(self.num_kernels):
                if xs is None:
                    xs = self.resblocks[i * self.num_kernels + j](x)
                else:
                    xs += self.resblocks[i * self.num_kernels + j](x)
            x = xs / self.num_kernels
        x = F.leaky_relu(x)
        x = self.conv_post(x)
        x = torch.tanh(x)

        return x

    def remove_weight_norm(self):
        print("Removing weight norm...")
        for l_ in self.ups:
            remove_weight_norm(l_)
        for l_ in self.resblocks:
            l_.remove_weight_norm()
        remove_weight_norm(self.conv_pre)
        remove_weight_norm(self.conv_post)


def init_hifigan(ckpt_path, json_path):
    """Initialize HiFi-GAN model from checkpoint and JSON config."""
    with open(json_path) as f:
        h = AttrDict(json.load(f))
    net = _Generator_HiFiRes(h)
    net.load_state_dict(
        torch.load(ckpt_path, map_location="cpu", weights_only=True)["generator"]
    )
    net.requires_grad_(False)
    return net


if __name__ == "__main__":
    import argparse

    import torchaudio

    from mbchl.signal.mel import MelSpectrogram

    parser = argparse.ArgumentParser()
    parser.add_argument("input")
    parser.add_argument("--ckpt", default="data/hifigan_48k_256bins.ckpt")
    parser.add_argument("--json", default="data/hifigan_48k_256bins.json")
    args = parser.parse_args()

    net = init_hifigan(args.ckpt, args.json)

    torch.set_grad_enabled(False)
    net.eval()

    melspec = MelSpectrogram(
        frame_length=2048,
        hop_length=480,
        n_fft=None,
        window="hann",
        center=True,
        pad_mode="constant",
        normalized=False,
        n_filters=256,
        f_min=0.0,
        f_max=None,
        fs=48000,
        norm="slaney",
        scale="slaney",
        power=1,
        log=True,
        log_eps=1e-7,
        mean=False,
        std=False,
        vad_dyn_range=None,
    )
    x, fs = torchaudio.load(args.input)
    mels = melspec(x)
    y = net(mels).squeeze(0)
    torchaudio.save("hifigan_in.wav", x, fs)
    torchaudio.save("hifigan_out.wav", y, fs)

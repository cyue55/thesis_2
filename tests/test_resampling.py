import random

import pytest
import soxr
import torch
import torch.nn.functional as F

from mbchl.layers import Downsample, Resample, Upsample
from mbchl.utils import soxr_output_lenght


class TestResample:
    kernel = (1, 1)
    x_even = torch.randn(1, 1, 16, 16)
    x_odd = torch.randn(1, 1, 17, 17)
    x_odd_output_shape = {
        "up": (1, 1, 34, 34),
        "down": (1, 1, 9, 9),
    }

    def test_resample(self):
        resample = Resample(self.kernel)

        resample_func = lambda x: resample(x, "up")  # noqa: E731
        self._test_resample(resample_func, "up")
        self._test_shape(resample_func, "up")

        resample_func = lambda x: resample(x, "down")  # noqa: E731
        self._test_resample(resample_func, "down")
        self._test_shape(resample_func, "down")

        # test the buffer_padding option for seamless down-up sampling
        resample = Resample(self.kernel, buffer_padding=True)
        skips = []
        for i in range(10):
            H, W = random.randint(2, 100), random.randint(2, 100)
            x = torch.randn(1, 1, H, W)
            y = resample(x, "down")
            skips.append((x, y))
        for i in range(10):
            x, y = skips.pop()
            z = resample(y, "up")
            assert z.shape == x.shape

    def test_upsample(self):
        upsample = Upsample(self.kernel)
        self._test_resample(upsample, "up")
        self._test_shape(upsample, "up")

    def test_downsample(self):
        downsample = Downsample(self.kernel)
        self._test_resample(downsample, "down")
        self._test_shape(downsample, "down")

    def _test_resample(self, resample_func, which):
        # check that the implementation of FIR up/down-sampling from
        # huggingface/diffusers is the same as ours for even input shapes and even
        # kernel lengths
        y = resample_func(self.x_even)
        z = self.resample_2d(self.x_even, which, self.kernel)
        assert torch.isclose(y, z).all()

    def _test_shape(self, resample_func, which):
        # check output shape when using odd shaped input
        y = resample_func(self.x_odd)
        assert y.shape == self.x_odd_output_shape[which]

    def resample_2d(self, x, which, kernel=(1, 3, 3, 1), factor=2, gain=1):
        # set up arguments for call to upfirdn2d
        kernel = torch.tensor(kernel, dtype=torch.float32)
        if kernel.ndim == 1:
            kernel = torch.outer(kernel, kernel)
        kernel /= torch.sum(kernel)
        pad = kernel.shape[0] - factor

        if which == "up":
            kernel *= gain * (factor**2)
            pad = ((pad + 1) // 2 + factor - 1, pad // 2)
            up, down = factor, 1
        elif which == "down":
            kernel *= gain
            pad = ((pad + 1) // 2, pad // 2)
            up, down = 1, factor
        else:
            raise ValueError(f"which must be up or down, got {which}")

        return self.upfirdn2d(
            x,
            kernel.to(device=x.device),
            up=up,
            down=down,
            pad=pad,
        )

    def upfirdn2d(self, tensor, kernel, up=1, down=1, pad=(0, 0)):
        # Copyright (c) 2023 The HuggingFace Team
        # Apache License Version 2.0
        # https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/resnet.py

        up_x = up_y = up
        down_x = down_y = down
        pad_x0 = pad_y0 = pad[0]
        pad_x1 = pad_y1 = pad[1]

        _, channel, in_h, in_w = tensor.shape
        tensor = tensor.reshape(-1, in_h, in_w, 1)

        _, in_h, in_w, minor = tensor.shape
        kernel_h, kernel_w = kernel.shape

        out = tensor.view(-1, in_h, 1, in_w, 1, minor)
        out = F.pad(out, [0, 0, 0, up_x - 1, 0, 0, 0, up_y - 1])
        out = out.view(-1, in_h * up_y, in_w * up_x, minor)

        out = F.pad(
            out,
            [
                0,
                0,
                max(pad_x0, 0),
                max(pad_x1, 0),
                max(pad_y0, 0),
                max(pad_y1, 0),
            ],
        )
        out = out[
            :,
            max(-pad_y0, 0) : out.shape[1] - max(-pad_y1, 0),
            max(-pad_x0, 0) : out.shape[2] - max(-pad_x1, 0),
            :,
        ]

        out = out.permute(0, 3, 1, 2)
        out = out.reshape(
            [
                -1,
                1,
                in_h * up_y + pad_y0 + pad_y1,
                in_w * up_x + pad_x0 + pad_x1,
            ]
        )
        w = torch.flip(kernel, [0, 1]).view(1, 1, kernel_h, kernel_w)
        out = F.conv2d(out, w)
        out = out.reshape(
            -1,
            minor,
            in_h * up_y + pad_y0 + pad_y1 - kernel_h + 1,
            in_w * up_x + pad_x0 + pad_x1 - kernel_w + 1,
        )
        out = out.permute(0, 2, 3, 1)
        out = out[:, ::down_y, ::down_x, :]

        out_h = (in_h * up_y + pad_y0 + pad_y1 - kernel_h) // down_y + 1
        out_w = (in_w * up_x + pad_x0 + pad_x1 - kernel_w) // down_x + 1

        return out.view(-1, channel, out_h, out_w)


@pytest.mark.parametrize("n", [10000, 10001, 10007])
@pytest.mark.parametrize("fs_in", [48000, 44100, 32000, 22050, 16000, 8000])
@pytest.mark.parametrize("fs_out", [48000, 44100, 32000, 22050, 16000, 8000])
def test_soxr_output_length(np_rng, n, fs_in, fs_out):
    x = np_rng.standard_normal(n)
    y = soxr.resample(x, fs_in, fs_out)
    assert len(y) == soxr_output_lenght(n, fs_in, fs_out)

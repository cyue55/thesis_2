import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class Resample(nn.Module):
    """Up- or down- sampling with FIR filtering.

    The code for this operation in huggingface/diffusers, sp-uhh/sgmse and
    yang-song/score_sde is extremely convoluted. The code below is inspired from the
    implementation by Tero Karras in NVlabs/edm and is much simpler. Results are the
    same during unit testing.
    """

    def __init__(self, fir_kernel, buffer_padding=False, causal=False):
        super().__init__()
        kernel = torch.as_tensor(fir_kernel, dtype=torch.float32)
        kernel = kernel.outer(kernel).unsqueeze(0).unsqueeze(1)
        kernel /= kernel.sum()
        self.causal = causal
        self.register_buffer("kernel", kernel)
        self._paddings = [] if buffer_padding else None

    def forward(self, x, up_or_down):
        """Forward pass."""
        kernel = self.kernel.tile([x.shape[1], 1, 1, 1])
        func_kwargs = dict(groups=x.shape[1], stride=2)
        if up_or_down == "down":
            padding = tuple(
                # the 2 lines below could be simplified by removing the if/else
                # statement but this prevents an error with torch.compile
                # see https://github.com/pytorch/pytorch/issues/101014
                math.ceil(self.kernel.shape[-1] / 2) - 1
                if dim % 2 == 0
                else math.ceil((self.kernel.shape[-1] + 1) / 2) - 1
                for dim in x.shape[-2:]
            )
            if self._paddings is not None:
                output_padding = tuple(
                    # same here, looks silly but fixes
                    # https://github.com/pytorch/pytorch/issues/101014
                    0 if (dim + 2 * pad - self.kernel.shape[-1]) % 2 == 0 else 1
                    for dim, pad in zip(x.shape[-2:], padding)
                )
                self._paddings.append((padding, output_padding))
            func = F.conv2d
            if self.causal:
                # manually pad left
                x = F.pad(x, (2 * padding[1], 0, padding[0], padding[0]))
                padding = (0, 0)
        elif up_or_down == "up":
            kernel *= 4
            if self._paddings is not None:
                padding, output_padding = self._paddings.pop()
            else:
                padding, output_padding = (self.kernel.shape[-1] - 1) // 2, 0
            if self.causal:
                # manually compute the transposed convolution
                # 1. insert interleaved zeros
                x = torch.stack([x, torch.zeros_like(x)], dim=-1)
                x = torch.flatten(x, start_dim=-2, end_dim=-1)
                x = torch.stack([x, torch.zeros_like(x)], dim=-2)
                x = torch.flatten(x, start_dim=-3, end_dim=-2)
                x = x[..., :-1, :-1]
                # 2. manually pad left
                h_pad = self.kernel.shape[-1] - 1 - padding[0]
                w_pad = self.kernel.shape[-1] - 1 - padding[1]
                h_pad_out, w_pad_out = output_padding
                x = F.pad(x, (2 * w_pad, w_pad_out, h_pad, h_pad + h_pad_out))
                padding = (0, 0)
                # 3. convolve with stride 1
                func = F.conv2d
                func_kwargs["stride"] = 1
            else:
                func = F.conv_transpose2d
                func_kwargs["output_padding"] = output_padding
        else:
            raise ValueError(f"up_or_down must be up or down, got " f"{up_or_down}")
        out = func(x, kernel, padding=padding, **func_kwargs)
        return out


class Upsample(Resample):
    """Upsampling with FIR filtering."""

    def __init__(self, fir_kernel):
        super().__init__(fir_kernel=fir_kernel, buffer_padding=False)

    def forward(self, x):
        """Forward pass."""
        return super().forward(x, "up")


class Downsample(Resample):
    """Downsampling with FIR filtering."""

    def __init__(self, fir_kernel):
        super().__init__(fir_kernel=fir_kernel, buffer_padding=False)

    def forward(self, x):
        """Forward pass."""
        return super().forward(x, "down")

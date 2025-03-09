import torch.nn as nn
import torch.nn.functional as F


class CausalConv1d(nn.Conv1d):
    """Causal 1D convolution layer."""

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        dilation=1,
        groups=1,
        bias=True,
        device=None,
        dtype=None,
    ):
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=0,
            dilation=dilation,
            groups=groups,
            bias=bias,
            device=device,
            dtype=dtype,
        )

    def forward(self, x):
        """Forward pass."""
        padding = (self.kernel_size[0] - 1) * self.dilation[0]
        x = F.pad(x, (padding, 0))
        x = super().forward(x)
        return x


class CausalConv2d(nn.Conv2d):
    """Causal 2D convolution layer."""

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        h_padding=0,
        dilation=1,
        groups=1,
        bias=True,
        device=None,
        dtype=None,
    ):
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=0,
            dilation=dilation,
            groups=groups,
            bias=bias,
            device=device,
            dtype=dtype,
        )
        self.h_padding = h_padding

    def forward(self, x):
        """Forward pass."""
        w_padding = (self.kernel_size[0] - 1) * self.dilation[0]
        x = F.pad(x, (w_padding, 0, self.h_padding, self.h_padding))
        x = super().forward(x)
        return x

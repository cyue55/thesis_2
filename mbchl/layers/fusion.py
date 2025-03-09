"""Fusion layer module.

A nice overview of the different fusion layers can be found in Table 2 in [1].

.. [1] K. Zmolikova, M. Delcroix, T. Ochiai, K. Kinoshita, J. Černocký and D. Yu,
   "Neural Target Speech Extraction: An overview", in IEEE Signal Processing Magazine,
   2023.
"""

import torch
import torch.nn as nn

from ..utils import Registry

FusionLayerRegistry = Registry("fusion_layer")


class _BaseFusionLayer(nn.Module):
    def forward(self, x, emb, axis=-1):
        # x: (batch_size, ..., input_channels)
        # emb: (batch_size, emb_channels)
        assert emb.ndim == 2, emb.shape
        x = x.swapaxes(-1, axis)
        while emb.ndim < x.ndim:
            emb = emb.unsqueeze(1)
        x = self._fuse(x, emb)
        return x.swapaxes(-1, axis)

    def _fuse(self, x, emb):
        raise NotImplementedError


@FusionLayerRegistry.register("cat")
class Concatenation(_BaseFusionLayer):
    """Concatenation fusion layer."""

    def __init__(self, *args, **kwargs):
        super().__init__()

    def _fuse(self, x, emb):
        return torch.cat([x, emb], dim=-1)


@FusionLayerRegistry.register("add")
class Addition(_BaseFusionLayer):
    """Addition fusion layer."""

    def __init__(self, input_channels, emb_channels, bias=False):
        super().__init__()
        self.input_channels = input_channels
        self.linear = nn.Linear(emb_channels, input_channels, bias=bias)

    def _fuse(self, x, emb):
        return x + self.linear(emb)


@FusionLayerRegistry.register("mult")
class Multiplication(_BaseFusionLayer):
    """Multiplication fusion layer."""

    def __init__(self, input_channels, emb_channels, bias=False):
        super().__init__()
        self.input_channels = input_channels
        self.linear = nn.Linear(emb_channels, input_channels, bias=bias)

    def _fuse(self, x, emb):
        return x * self.linear(emb)


@FusionLayerRegistry.register("film")
class FiLM(_BaseFusionLayer):
    """FiLM fusion layer."""

    def __init__(self, input_channels, emb_channels, bias=False):
        super().__init__()
        self.input_channels = input_channels
        self.linear = nn.Linear(emb_channels, input_channels * 2, bias=bias)

    def _fuse(self, x, emb):
        gamma, beta = self.linear(emb).chunk(2, dim=-1)
        return x * gamma + beta

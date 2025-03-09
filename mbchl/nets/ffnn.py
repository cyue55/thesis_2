import torch.nn as nn

from ..layers import FusionLayerRegistry
from .registry import NetRegistry


@NetRegistry.register("ffnn")
class FFNN(nn.Module):
    """Feed-forward neural network."""

    def __init__(
        self,
        input_size,
        output_size,
        hidden_sizes=[1024, 1024],
        dropout=0.2,
        aggregate=False,
        fusion_layer=None,
        emb_dim=None,
    ):
        super().__init__()
        self.aggregate = aggregate

        if fusion_layer is not None:
            assert isinstance(fusion_layer, str)
            assert len(hidden_sizes) > 0
            assert all(hidden_sizes[0] == h for h in hidden_sizes)
            fusion_layer_cls = FusionLayerRegistry.get(fusion_layer)
            fusion_layer = fusion_layer_cls(hidden_sizes[0], emb_dim)
        self.fusion_layer = fusion_layer

        self.input_size = input_size
        self.output_size = output_size
        self.module_list = nn.ModuleList()
        start_size = input_size
        for i in range(len(hidden_sizes)):
            end_size = hidden_sizes[i]
            self.module_list.append(nn.Linear(start_size, end_size))
            self.module_list.append(nn.ReLU())
            self.module_list.append(nn.Dropout(dropout))
            start_size = end_size
        self.module_list.append(nn.Linear(start_size, output_size))
        self.module_list.append(nn.Sigmoid())

    def forward(self, x, emb=None):
        """Forward pass."""
        # (batch, features, frames)
        assert not (emb is None and self.fusion_layer is not None)
        assert not (emb is not None and self.fusion_layer is None)
        x = x.transpose(1, 2)
        for module in self.module_list:
            if (
                emb is not None
                and isinstance(module, nn.Linear)
                and module.in_features == self.fusion_layer.input_channels
            ):
                x = self.fusion_layer(x, emb)
            x = module(x)
        x = x.transpose(1, 2)
        if self.aggregate:
            x = x.mean(dim=-1)
        return x

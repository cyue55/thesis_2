import torch
import torch.nn as nn

from ..utils import Registry

NormRegistry = Registry("norm")


@NormRegistry.register("cgn")
class CausalGroupNorm(nn.Module):
    r"""Causal Group Normalization.

    Input tensors must have shape ``(N, C, ...)`` where ``N`` is the batch dimension,
    ``C`` is the channel dimension, and ``...`` are the spatial dimensions. The
    statistics are calculated over the grouped channels and spatial dimensions as
    illustrated in [1], Figure 2.

    .. [1] Y\. Wu and K. He, "Group Normalization", in Proc. ECCV, 2018.

    Parameters
    ----------
    num_groups : int
        Number of groups to separate the channels into.
    num_channels : int
        Number of channels in the input tensor.
    time_dim : int
        Time dimension of the input tensor. Cannot be 0 or 1.
    eps : float
        Epsilon value for numerical stability.

    """

    def __init__(self, num_groups, num_channels, time_dim=-1, eps=1e-7):
        super().__init__()

        if num_channels % num_groups != 0:
            raise ValueError("num_channels must be divisible by num_groups")
        self._check_time_dim(time_dim)

        self.num_groups = num_groups
        self.time_dim = time_dim
        self.eps = eps

        self.gain = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))

    def forward(self, x):
        """Forward pass."""
        time_dim = list(range(x.ndim))[self.time_dim]
        self._check_time_dim(time_dim)

        orig_shape = x.shape
        new_shape = [
            x.shape[0],
            self.num_groups,
            x.shape[1] // self.num_groups,
            *[x.shape[i] for i in range(2, x.ndim)],
        ]
        x = x.reshape(new_shape)
        time_dim += 1

        sum_dims = [i for i in range(x.ndim) if i not in [0, 1, time_dim]]
        count = torch.ones(x.shape, device=x.device)
        count = count.sum(sum_dims, keepdims=True).cumsum(time_dim)
        mean = x.sum(sum_dims, keepdims=True).cumsum(time_dim)
        mean = mean / count
        var = x.pow(2).sum(sum_dims, keepdims=True).cumsum(time_dim)
        var = var / count - mean.pow(2)
        x = (x - mean) / (var + self.eps).sqrt()

        x = x.reshape(orig_shape)

        param_shape = [1 if i != 1 else x.shape[1] for i in range(x.ndim)]
        return x * self.gain.view(*param_shape) + self.bias.view(*param_shape)

    def _check_time_dim(self, time_dim):
        if time_dim == 0:
            raise ValueError("time_dim cannot be the batch dimension (0)")
        elif time_dim == 1:
            raise ValueError("time_dim cannot be the channel dimension (1)")


@NormRegistry.register("cln")
class CausalLayerNorm(CausalGroupNorm):
    r"""Causal Layer Normalization.

    Input tensors must have shape ``(N, C, ...)`` where ``N`` is the batch dimension,
    ``C`` is the channel dimension, and ``...`` are the spatial dimensions. The
    statistics are calculated over the channel and spatial dimensions as illustrated in
    [1], Figure 2.

    Note this does not match the layer normalization commonly used in NLP which
    calculates the statistics over the channel dimension only as illustrated in [2],
    Figure 1. See `here <https://stackoverflow.com/questions/70065235/understanding-torch-nn-layernorm-in-nlp>`__
    for a detailed discussion.

    .. [1] Y\. Wu and K. He, "Group Normalization", in Proc. ECCV, 2018.
    .. [2] S. Shen, Z. Yao, A. Gholami, M. W. Mahoney and K. Keutzer, "PowerNorm:
       Rethinking Batch Normalization in Transformers", in Proc. ICML, 2020.

    Parameters
    ----------
    num_channels : int
        Number of channels in the input tensor.
    time_dim : int
        Time dimension of the input tensor. Cannot be 0 or 1.
    eps : float
        Epsilon value for numerical stability.

    """

    def __init__(self, num_channels, time_dim=-1, eps=1e-7):
        super().__init__(
            num_groups=1,
            num_channels=num_channels,
            time_dim=time_dim,
            eps=eps,
        )


@NormRegistry.register("cin")
class CausalInstanceNorm(CausalGroupNorm):
    r"""Causal Instance Normalization.

    Input tensors must have shape ``(N, C, ...)`` where ``N`` is the batch dimension,
    ``C`` is the channel dimension, and ``...`` are the spatial dimensions. The
    statistics are calculated over the spatial dimensions as illustrated in [1], Figure
    2.

    .. [1] Y\. Wu and K. He, "Group Normalization", in Proc. ECCV, 2018.

    Parameters
    ----------
    num_channels : int
        Number of channels in the input tensor.
    time_dim : int
        Time dimension of the input tensor. Cannot be 0 or 1.
    eps : float
        Epsilon value for numerical stability.

    """

    def __init__(self, num_channels, time_dim=-1, eps=1e-7):
        super().__init__(
            num_groups=num_channels,
            num_channels=num_channels,
            time_dim=time_dim,
            eps=eps,
        )

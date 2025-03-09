import torch

from ..utils import Registry


class _HARegistry(Registry):
    def init(self, key, seed=None, **kwargs):
        if seed is not None:
            current_seed = torch.seed()
            torch.manual_seed(seed)
        ha_cls = self.get(key)
        ha = ha_cls(**kwargs)
        if seed is not None:
            torch.manual_seed(current_seed)
        return ha


HARegistry = _HARegistry("ha")

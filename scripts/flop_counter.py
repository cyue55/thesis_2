import argparse
import os

import torch
from thop import profile

from mbchl.has import HARegistry
from mbchl.utils import random_audiogram, read_yaml, seed_everything


def main():
    # load config
    cfg_path = os.path.join(args.input, "config.yaml")
    cfg = read_yaml(cfg_path)

    # seed for reproducibility
    seed_everything(cfg["global_seed"])

    # initialize model
    model = HARegistry.init(cfg["ha"], **cfg["ha_kw"])

    # disable gradients and set model to eval
    torch.set_grad_enabled(False)
    model.eval()

    # move model to device
    device = "cuda" if args.cuda else "cpu"
    model.to(device)

    # initialize random audiogram
    if model._audiogram:
        audiogram = random_audiogram()
        audiogram = torch.tensor(audiogram, dtype=torch.float32, device=device)
        extra_inputs = (audiogram,)
    else:
        extra_inputs = None

    # count flops
    class Net(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.model = model

        def forward(self, x, extra_inputs):
            return self.model.enhance(x, extra_inputs=extra_inputs)

    net = Net()

    durations = [1, 4]
    macs = []
    for duration in durations:
        x = torch.randn(1, duration * cfg["dataset"]["train_kw"]["fs"], device=device)
        mac, _ = profile(net, inputs=(x, extra_inputs))
        mac = int(mac / duration)
        macs.append(mac)

    for duration, mac in zip(durations, macs):
        print(f"{duration}-s-long segment: {mac:_} ({mac * 1e-9:.3f} G) MAC/s")

    params = model.count_params()
    print(f"{params:_} ({params * 1e-6:.1f} M) params.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input")
    parser.add_argument("--cuda", action="store_true")
    args = parser.parse_args()
    main()

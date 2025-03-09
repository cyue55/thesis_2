import argparse

import torch
from tqdm import tqdm

from mbchl.metrics import MetricRegistry


def main():
    for metric in args.metrics:
        metric_cls = MetricRegistry.get(metric)
        metric_obj = metric_cls()
        generator = torch.Generator().manual_seed(42)
        x = torch.randn(1, 4 * 16000, generator=generator)
        y = torch.randn(1, 4 * 16000, generator=generator)
        for i in tqdm(range(args.n), desc=metric):
            metric_obj(x, y)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", type=int, default=10)
    parser.add_argument("--metrics", nargs="+", default=MetricRegistry.keys())
    args = parser.parse_args()
    main()

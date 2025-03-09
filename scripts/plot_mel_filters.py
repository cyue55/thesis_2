import argparse

import matplotlib.pyplot as plt
import torch

from mbchl.signal.mel import MelFilterbank


def main():
    mel_fb = MelFilterbank(
        n_filters=args.n_filters,
        n_fft=args.n_fft,
        f_min=args.f_min,
        f_max=args.f_max,
        fs=args.fs,
        norm=args.norm,
        scale=args.scale,
    )

    fig, axes = plt.subplots(2, 1)
    axes[0].plot(mel_fb.filters_cpu.T)
    axes[0].set_title("filters")
    axes[1].plot(mel_fb.inverse_filters_cpu.T)
    axes[1].set_title("inverse filters")
    fig.tight_layout()

    fig, ax = plt.subplots(1, 1)
    im = ax.imshow(mel_fb.inverse_filters_cpu @ mel_fb.filters_cpu)
    plt.colorbar(im, ax=ax)
    ax.set_title("analysis-synthesis function")
    fig.tight_layout()

    def plot(ax, data, title, vmin, vmax):
        im = ax.imshow(data, aspect="auto", vmin=vmin, vmax=vmax)
        plt.colorbar(im, ax=ax)
        ax.set_title(title)

    def example(x):
        fig, axes = plt.subplots(3, 1)
        y = mel_fb(x)
        z = mel_fb.inverse(y)
        vmin = min(x.min(), y.min(), z.min())
        vmax = max(x.max(), y.max(), z.max())
        plot(axes[0], x, "input", vmin, vmax)
        plot(axes[1], y, "analysis", vmin, vmax)
        plot(axes[2], z, "synthesis", vmin, vmax)
        fig.tight_layout()

    example(torch.rand(257, 500))
    example(torch.randn(257, 500))
    example(torch.ones(257, 500))

    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_filters", type=int, default=64)
    parser.add_argument("--n_fft", type=int, default=512)
    parser.add_argument("--f_min", type=float, default=0.0)
    parser.add_argument("--f_max", type=float)
    parser.add_argument("--fs", type=int, default=16000)
    parser.add_argument("--norm", default="valid")
    parser.add_argument("--scale", default="slaney")
    args = parser.parse_args()

    main()

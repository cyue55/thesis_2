import argparse
import math
import warnings

import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
import soxr
import torch

from mbchl.plot import plot_waveform
from mbchl.signal.auditory import AuditoryModel
from mbchl.utils import impulse


def main():
    if args.input is None:
        x = impulse(args.n)
    else:
        x, fs = sf.read(
            args.input,
            always_2d=True,
        )
        if x.shape[1] > 1:
            warnings.warn("input has more than one channel, using only the first one")
        x = x[:, 0]
        x = x / np.abs(x).max()
        end = round(args.end * fs) if args.end is not None else len(x)
        x = x[round(args.start * fs) : end]
        if fs != args.fs:
            x = soxr.resample(x, fs, args.fs)
        x = torch.tensor(x, dtype=torch.float32)

    plot_waveform(x, fs=args.fs)

    am = AuditoryModel(
        fs=args.fs,
        filterbank=args.filterbank,
        ihc=args.ihc,
        adaptation=args.adaptation,
        modulation=args.modulation,
    )

    y = am(x)
    if args.modulation == "none":
        y = y.unsqueeze(1)

    nrow = math.ceil(y.shape[1] / args.ncol)
    fig, axes = plt.subplots(
        nrow,
        args.ncol,
        figsize=args.figsize,
        sharex=True,
        sharey=True,
    )
    if not isinstance(axes, np.ndarray):
        axes = np.array([axes])
    axes = axes.flatten()

    xmin, xmax = 0, y.shape[-1] / args.fs
    ymin, ymax = am.filterbank.fc[0], am.filterbank.fc[-1]

    for i in range(y.shape[1]):
        axes[i].imshow(
            y[:, i, :],
            aspect="auto",
            origin="lower",
            extent=[xmin, xmax, ymin, ymax],
        )
        if args.modulation != "none":
            axes[i].set_title(f"{round(am.modulation.fc[i])} Hz")
        if i >= len(axes) - args.ncol:
            axes[i].set_xlabel("Time (s)")
        if i % args.ncol == 0:
            axes[i].set_ylabel("Frequency (Hz)")

    for i in range(y.shape[1], len(axes)):
        axes[i].axis("off")

    # add colorbar
    if y.shape[1] == 1:
        fig.colorbar(axes[0].get_images()[0], ax=axes, orientation="vertical")
    else:
        plt.tight_layout()

    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input")
    parser.add_argument("--start", type=float, default=0.0)
    parser.add_argument("--end", type=float)
    parser.add_argument("--n", type=int, default=20000)
    parser.add_argument("--fs", type=int, default=20000)
    parser.add_argument("--filterbank", default="gammatone")
    parser.add_argument("--ihc", default="hwrlp")
    parser.add_argument("--adaptation", default="log")
    parser.add_argument("--modulation", default="none")
    parser.add_argument("--ncol", type=int, default=1)
    parser.add_argument("--figsize", type=int, nargs=2, default=(16, 8))
    args = parser.parse_args()
    main()

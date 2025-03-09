import argparse

import matplotlib.pyplot as plt
import torch

from mbchl.plot import plot_spectrum
from mbchl.signal.filters import GammatoneFilterbank
from mbchl.utils import impulse


def main():
    dtype = torch.float32 if args.precision == "single" else torch.float64
    x = impulse(args.n_fft, dtype=dtype)
    fig, ax = plt.subplots(figsize=(10, 5))
    for filter_type in args.filter_types:
        fb = GammatoneFilterbank(
            fs=args.fs,
            fc=args.fc,
            order=args.order,
            bw_mult=args.bw_mult,
            filter_type=filter_type,
            fir_ntaps=args.fir_ntaps,
            iir_output=args.iir_output,
            precision=args.precision,
        )
        fig, ax = plot_spectrum(
            fb(x).real,
            n_fft=args.n_fft,
            fs=args.fs,
            dbscale=True,
            semilogx=True,
            xmin=50,
            xmax=args.fs / 2,
            ymin=args.ymin,
            ymax=args.ymax,
            ax=ax,
            label=filter_type,
        )
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_fft", type=int, default=4096)
    parser.add_argument("--fs", type=int, default=20000)
    parser.add_argument("--fc", type=int, default=1000)
    parser.add_argument("--order", type=int, default=4)
    parser.add_argument("--bw_mult", type=float)
    parser.add_argument(
        "--filter_types",
        nargs="+",
        default=["fir", "gtf", "apgf", "ozgf", "amt_classic", "amt_allpole"],
    )
    parser.add_argument("--fir_ntaps", type=int, default=512)
    parser.add_argument("--iir_output", default="ba")
    parser.add_argument("--precision", default="double")
    parser.add_argument("--ymin", type=float, default=-100.0)
    parser.add_argument("--ymax", type=float, default=10.0)
    args = parser.parse_args()
    main()

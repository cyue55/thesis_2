import argparse

import matplotlib.pyplot as plt
import torch

from mbchl.plot import plot_spectrum, plot_waveform
from mbchl.signal.filters import DRNLFilterbank
from mbchl.utils import impulse


def main():
    dtype = torch.float32 if args.precision == "single" else torch.float64
    fb = DRNLFilterbank(
        fs=args.fs,
        f_min=args.f_min,
        f_max=args.f_max,
        n_filters=args.n_filters,
        erb_space=args.erb_space,
        precision=args.precision,
        filter_type=args.filter_type,
        fir_ntaps=args.fir_ntaps,
    )
    x = impulse(args.n_fft, dtype=dtype)
    y = fb(x)
    plot_spectrum(
        y,
        n_fft=args.n_fft,
        fs=args.fs,
        dbscale=True,
        semilogx=True,
        xmin=50,
        xmax=args.fs / 2,
        ymin=args.ymin,
        ymax=args.ymax,
    )
    plot_waveform(y)
    print("Center frequencies:", fb.fc)
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_fft", type=int, default=4096)
    parser.add_argument("--fs", type=int, default=20000)
    parser.add_argument("--f_min", type=int, default=80)
    parser.add_argument("--f_max", type=int, default=8000)
    parser.add_argument("--n_filters", type=int)
    parser.add_argument("--erb_space", type=int, default=1)
    parser.add_argument("--precision", default="double")
    parser.add_argument("--ymin", type=float, default=-100.0)
    parser.add_argument("--ymax", type=float, default=10.0)
    parser.add_argument("--filter_type", default="amt_classic")
    parser.add_argument("--fir_ntaps", type=int, default=512)
    args = parser.parse_args()
    args.erb_space = None if args.erb_space < 0 else args.erb_space
    main()

import argparse

import matplotlib.pyplot as plt
import torch

from mbchl.plot import plot_spectrum, plot_waveform
from mbchl.signal.filters import GammatoneFilterbank
from mbchl.utils import impulse


def main():
    dtype = torch.float32 if args.precision == "single" else torch.float64
    x = impulse(args.n_fft, dtype=dtype)
    fb = GammatoneFilterbank(
        fs=args.fs,
        f_min=args.f_min,
        f_max=args.f_max,
        n_filters=args.n_filters,
        order=args.order,
        bw_mult=args.bw_mult,
        filter_type=args.filter_type,
        fir_ntaps=args.fir_ntaps,
        iir_output=args.iir_output,
        gain=args.gain,
        precision=args.precision,
        compensate_delay=args.compensate_delay,
    )
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
    inv, _delayed = fb.inverse(y, _return_delayed=True)
    if args.synthesis:
        plot_spectrum(
            inv,
            n_fft=args.n_fft,
            fs=args.fs,
            dbscale=True,
            semilogx=True,
            xmin=50,
            xmax=args.fs / 2,
            ymin=args.ymin,
            ymax=args.ymax,
        )
        plot_waveform(_delayed)
        plot_waveform(_delayed.sum(-2))
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_fft", type=int, default=4096)
    parser.add_argument("--fs", type=int, default=20000)
    parser.add_argument("--f_min", type=int, default=80)
    parser.add_argument("--f_max", type=int, default=8000)
    parser.add_argument("--n_filters", type=int, default=30)
    parser.add_argument("--order", type=int, default=4)
    parser.add_argument("--bw_mult", type=float)
    parser.add_argument("--filter_type", default="apgf")
    parser.add_argument("--fir_ntaps", type=int, default=512)
    parser.add_argument("--iir_output", default="ba")
    parser.add_argument("--gain", nargs="+", type=float)
    parser.add_argument("--precision", default="double")
    parser.add_argument("--ymin", type=float, default=-100.0)
    parser.add_argument("--ymax", type=float, default=10.0)
    parser.add_argument("--synthesis", action="store_true")
    parser.add_argument("--compensate_delay", action="store_true")
    args = parser.parse_args()
    main()

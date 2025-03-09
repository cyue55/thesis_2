import argparse
import warnings

import matplotlib.pyplot as plt
import soxr
import torch
import torchaudio

from mbchl.signal.mel import MelSpectrogram


def main():
    x, fs = torchaudio.load(args.input)
    if x.shape[0] > 1:
        warnings.warn(
            "Input file is multi-channel. Only the first channel will be plotted."
        )
    x = x[0]
    if args.fs is not None and fs != args.fs:
        warnings.warn(f"Resampling from {fs} to {args.fs}")
        x = torch.tensor(soxr.resample(x, fs, args.fs), dtype=x.dtype)
        fs = args.fs
    melspec_obj = MelSpectrogram(
        frame_length=args.frame_length,
        hop_length=args.hop_length,
        n_fft=args.n_fft,
        window=args.window,
        center=args.center,
        pad_mode=args.pad_mode,
        normalized=args.normalized,
        n_filters=args.n_filters,
        f_min=args.f_min,
        f_max=args.f_max,
        fs=fs,
        norm=args.norm,
        scale=args.scale,
        power=args.power,
        log=args.log,
        log_eps=args.log_eps,
    )
    x_mel = melspec_obj(x)

    fig, ax = plt.subplots()
    ax.imshow(x_mel, aspect="auto", origin="lower")
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input", help="input audio file")
    parser.add_argument("--frame_length", type=int, default=512)
    parser.add_argument("--hop_length", type=int, default=256)
    parser.add_argument("--n_fft", type=int)
    parser.add_argument("--window", default="hann")
    parser.add_argument("--no_center", action="store_false", dest="center")
    parser.add_argument("--pad_mode", default="constant")
    parser.add_argument("--normalized", action="store_true")
    parser.add_argument("--n_filters", type=int, default=64)
    parser.add_argument("--f_min", type=float, default=0.0)
    parser.add_argument("--f_max", type=float)
    parser.add_argument("--norm", default="valid")
    parser.add_argument("--scale", default="slaney")
    parser.add_argument("--power", type=int, default=2)
    parser.add_argument("--no_log", action="store_false", dest="log")
    parser.add_argument("--log_eps", type=float, default=1e-7)
    parser.add_argument("--fs", type=int)
    args = parser.parse_args()

    main()

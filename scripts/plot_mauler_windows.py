import argparse

import matplotlib.pyplot as plt

from mbchl.signal.stft import mauler_windows

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--analysis_length", type=int, default=1024)
    parser.add_argument("--synthesis_length", type=int, default=256)
    parser.add_argument("--n_fft", type=int, default=1024)
    args = parser.parse_args()

    analysis_window, synthesis_window = mauler_windows(
        analysis_length=args.analysis_length,
        synthesis_length=args.synthesis_length,
        n_fft=args.n_fft,
    )

    plt.plot(analysis_window, label="Analysis window")
    plt.plot(synthesis_window, label="Synthesis window")
    plt.legend()
    plt.show()

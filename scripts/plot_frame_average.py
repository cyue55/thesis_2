import os
import torch
import numpy as np
import matplotlib.pyplot as plt

def debug_plot_frame_avg(x, y, orig_shape, save_dir="./frame_debug_plots", max_plot=1):
    """
    Plot and save comparisons between original signal and smoothed RMS output.

    Parameters
    ----------
    x : torch.Tensor
        Input tensor (possibly reshaped). Shape: (batch * channel, 1, time)
    y : torch.Tensor
        Smoothed output. Shape: (batch, channel, num_frames)
    orig_shape : tuple
        The original shape before reshaping
    save_dir : str
        Directory to save debug plots
    max_plot : int
        Max number of batch and channel plots to generate
    """
    os.makedirs(save_dir, exist_ok=True)

    B = orig_shape[0]
    T = orig_shape[-1]
    C = y.shape[1]

    # Reshape x back to (B, C, T)
    x_recover = x.view(B, -1, T).detach().cpu().numpy()
    y = y.detach().cpu().numpy()

    for b in range(min(B, max_plot)):
        for c in range(min(C, max_plot)):
            orig = x_recover[b, c]
            smoothed = y[b, c]

            t_orig = np.linspace(0, 1, len(orig))
            t_smooth = np.linspace(0, 1, len(smoothed))
            smoothed_interp = np.interp(t_orig, t_smooth, smoothed)

            plt.figure(figsize=(10, 4))
            plt.plot(t_orig, orig, label="Original", alpha=0.6)
            plt.plot(t_orig, smoothed_interp, label="Smoothed RMS", linewidth=2)
            plt.title(f"Batch {b}, Channel {c} - FrameBasedAveraging")
            plt.xlabel("Time (normalized)")
            plt.ylabel("Amplitude / RMS")
            plt.legend()
            plt.grid(True)

            fname = f"frameavg_b{b}_c{c}.png"
            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, fname))
            plt.close()
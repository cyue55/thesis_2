import argparse

import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm

from mbchl.has.diffusionha.sdes import SDERegistry


def gaussian(x, mu, sigma):
    return torch.exp(-0.5 * ((x - mu) / sigma) ** 2)


def forward():
    t = torch.linspace(0, 1, N_steps)
    dt = 1 / N_steps
    x = torch.as_tensor([x_0])
    output = [x]
    for i in range(N_steps - 1):
        x = x + sde.f(x, y, t[i]) * dt + sde.g(t[i]) * torch.randn(1) * dt**0.5
        output.append(x)
    return t, torch.cat(output)


def backward(x_T):
    t = torch.linspace(1, 0, N_steps)
    dt = -1 / N_steps
    x = torch.as_tensor([x_T])
    output = [x]
    for i in range(N_steps - 1):
        x = (
            x
            + (sde.f(x, y, t[i]) - sde.g(t[i]) ** 2 * score(x, t[i])) * dt
            + sde.g(t[i]) * torch.randn(1) * (-dt) ** 0.5
        )
        output.append(x)
    return t, torch.cat(output)


def score(x, t):
    mu = sde.s(t) * (x_0 - y) + y
    return -(x - mu) / (sde.s(t) * sde.sigma(t)) ** 2


def plot_prob(ax, prob_, t, mu_, sigma_, _title=None, no_xticks=False, no_yticks=False):
    ax.imshow(
        prob_,
        origin="lower",
        aspect="auto",
        cmap="Oranges",
        interpolation="bilinear",
        extent=[0, 1, xmin, xmax],
    )
    ax.hlines(y, 0, 1, color="black", linestyle="--")
    ax.annotate(
        r"$y$",
        xy=(1, y) if no_yticks else (0, y),
        xytext=(-4, 8) if no_yticks else (12, 8),
        textcoords="offset points",
        ha="right",
        va="center",
    )
    ax.plot(t, mu_, color="dimgrey")
    ax.plot(t, mu_ + 0.5 * sigma_, color="grey", alpha=0.3)
    ax.plot(t, mu_ - 0.5 * sigma_, color="grey", alpha=0.3)
    ax.plot(t, mu_ + sigma_, color="grey", alpha=0.3)
    ax.plot(t, mu_ - sigma_, color="grey", alpha=0.3)
    format_axes(ax, _title, no_xticks, no_yticks)


def format_axes(ax, _title, no_xticks=False, no_yticks=False):
    ax.set_xlim(0, 1)
    ax.set_ylim(xmin, xmax)
    if _title is not None:
        ax.set_title(_title)
    if no_xticks:
        ax.set_xticklabels([])
    else:
        xticklabels = ax.get_xticklabels()
        xticklabels[0].set_horizontalalignment("left")
        xticklabels[-1].set_horizontalalignment("right")
        ax.set_xlabel(r"$t$")
    ax.set_yticks([0, 0.5, 1, 1.5])
    if no_yticks:
        ax.set_yticklabels([])
    else:
        yticklabels = ax.get_yticklabels()
        yticklabels[0].set_verticalalignment("bottom")
        yticklabels[-1].set_verticalalignment("top")
        ax.set_ylabel(r"$x_t$")


def plot(sde, title, i_sde):
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    axes = axes.flatten()
    t = torch.linspace(0, 1, res)
    x = torch.linspace(xmin, xmax, res)
    T, X = torch.meshgrid(t, x, indexing="ij")
    sigma = sde.s(T) * sde.sigma(T)
    mu = sde.s(T) * (x_0 - y) + y
    prob = gaussian(X, mu, sigma)
    for ax_, prob_, mu_, sigma, title_, no_yticks in zip(
        axes,
        [prob.T, prob.flip(0).T],
        [mu[:, 0], mu[:, 0].flip(0)],
        [sigma[:, 0], sigma[:, 0].flip(0)],
        ["forward SDE", "reverse SDE"],
        [False, True],
    ):
        plot_prob(ax_, prob_, t, mu_, sigma, title_, no_yticks=no_yticks)
    fig.suptitle(title)
    fig.tight_layout()

    hist2d_x_0, hist2d_y_0 = [], []
    hist2d_x_1, hist2d_y_1 = [], []
    for i in tqdm(range(args.nsim), desc=title):
        t_sim, x_sim = forward()
        axes[0].plot(t_sim, x_sim, c="k", alpha=0.2)
        hist2d_x_0 += t_sim.tolist()
        hist2d_y_0 += x_sim.tolist()
        _, x_sim = backward(sde.prior(torch.tensor(y)))
        axes[1].plot(t_sim, x_sim, c="k", alpha=0.2)
        hist2d_x_1 += t_sim.tolist()
        hist2d_y_1 += x_sim.tolist()

    if args.plot_2dhist:
        fig, axes = plt.subplots(1, 2, figsize=(11, 5))
        axes = axes.flatten()
        bins = [
            np.linspace(0, 1, res + 1),
            np.linspace(xmin, xmax, res + 1),
        ]
        for i, (hist_x, hist_y, title_) in enumerate(
            [
                (hist2d_x_0, hist2d_y_0, "forward SDE"),
                (hist2d_x_1, hist2d_y_1, "reverse SDE"),
            ]
        ):
            H, _, _ = np.histogram2d(hist_x, hist_y, bins=bins)
            axes[i].imshow(
                H.T,
                origin="lower",
                aspect="auto",
                cmap="Oranges",
                extent=[0, 1, xmin, xmax],
            )
            format_axes(axes[i], title_, no_yticks=i == 1)
        fig.suptitle(title)
        fig.tight_layout()

    if args.plot_dists:
        t_hist = torch.linspace(t_eps, 1 - t_eps, 1000)
        snr_hist_1 = -sde.sigma(t_hist).pow(2).log()
        snr_hist_2 = -(sde.s(t_hist) * sde.sigma(t_hist)).pow(2).log()
        sigma_hist = sde.sigma(t_hist)
        color = colors[i_sde]
        for hist, ax_h, ax_t in zip(
            [snr_hist_1, snr_hist_2, sigma_hist],
            [ax_hist_snr_1, ax_hist_snr_2, ax_hist_sigma],
            [ax_time_snr_1, ax_time_snr_2, ax_time_sigma],
        ):
            mean = hist.mean()
            label = f"{title}, mean={hist.mean():.2f}"
            ax_h.hist(hist, alpha=0.7, label=label, bins=50, density=True, color=color)
            ax_h.axvline(mean, color=color, linestyle="--", zorder=-1)
            ax_t.plot(t_hist, hist, color=color, label=title)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("sdes", nargs="+")
    parser.add_argument("--plot_2dhist", action="store_true")
    parser.add_argument("--plot_dists", action="store_true")
    parser.add_argument("--nsim", type=int, default=3)
    args = parser.parse_args()

    x_0 = 1
    y = 0.2
    N_steps = 100
    t_eps = 1e-2
    xmin = 0
    xmax = 1.5
    res = 100
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    if args.plot_dists:
        fig_dist_snr_1, (ax_hist_snr_1, ax_time_snr_1) = plt.subplots(1, 2)
        fig_dist_snr_2, (ax_hist_snr_2, ax_time_snr_2) = plt.subplots(1, 2)
        fig_dist_sigma, (ax_hist_sigma, ax_time_sigma) = plt.subplots(1, 2)

    for i_sde, sde_name in enumerate(args.sdes):
        sde_cls = SDERegistry.get(sde_name)
        sde = sde_cls()
        plot(sde, title=sde_name, i_sde=i_sde)

    if args.plot_dists:
        ax_hist_snr_1.legend()
        ax_hist_snr_1.set_title(r"Unscaled $p(\lambda)$")
        ax_hist_snr_2.legend()
        ax_hist_snr_2.set_title(r"Scaled $p(\lambda)$")
        ax_hist_sigma.legend()
        ax_hist_sigma.set_title(r"Unscaled $p(\sigma)$")
        ax_time_snr_1.legend()
        ax_time_snr_1.set_title(r"Unscaled $\lambda(t)$")
        ax_time_snr_2.legend()
        ax_time_snr_2.set_title(r"Scaled $\lambda(t)$")
        ax_time_sigma.legend()
        ax_time_sigma.set_title(r"Unscaled $\sigma(t)$")
        fig_dist_snr_1.tight_layout()
        fig_dist_snr_2.tight_layout()
        fig_dist_sigma.tight_layout()

    plt.show()

import matplotlib.pyplot as plt
import numpy as np

from mbchl.utils import ltas

SEABORN_DEFAULT_PALETTE = [
    "#4C72B0",
    "#DD8452",
    "#55A868",
    "#C44E52",
    "#8172B3",
    "#937860",
    "#DA8BC3",
    "#8C8C8C",
    "#CCB974",
    "#64B5CD",
]


def set_seaborn_palette():
    """Set the default :mod:`matplotlib` color palette to Seaborn's."""
    plt.rcParams["axes.prop_cycle"] = plt.cycler(color=SEABORN_DEFAULT_PALETTE)


def plot_spectrum(
    x,
    n_fft=2048,
    power=2.0,
    fs=1,
    xmin=None,
    xmax=None,
    ymin=None,
    ymax=None,
    dbscale=False,
    semilogx=False,
    grid=True,
    axis=-1,
    eps=1e-10,
    ax=None,
    label=None,
):
    """Plot the long-term average spectrum of a signal.

    Parameters
    ----------
    x : torch.Tensor or numpy.ndarray
        Input signal. One- or two-dimensional.
    n_fft : int, optional
        Frame length and number of FFT points.
    power : float, optional
        Exponent to apply to the STFT magnitude before averaging.
    fs : float, optional
        Sampling frequency.
    xmin : float, optional
        X-axis range minimum.
    xmax : float, optional
        X-axis range maximum.
    ymin : float, optional
        Y-axis range minimum.
    ymax : float, optional
        Y-axis range maximum.
    dbscale : bool, optional
        Whether to show the Y-axis in dB scale.
    semilogx : bool, optional
        Whether to show the X-axis in semi-log scale.
    grid : bool, optional
        Whether to show the grid.
    axis : int, optional
        Time axis of the input signal.
    eps : float, optional
        Small value to avoid log of 0 when plotting in dB scale.
    ax : matplotlib.axes.Axes, optional
        Axes to plot on.
    label : str, optional
        Label for the plot.

    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure instance.
    ax : matplotlib.axes.Axes
        Axes instance.

    """
    if x.ndim > 2:
        raise ValueError("input must be one- or two-dimensional")
    x = np.asarray(x)
    spec = ltas(x, n_fft, power=power, axis=axis)
    if dbscale:
        spec = 10 * np.log10(spec.clip(min=eps))
    f = np.fft.rfftfreq(n_fft, 1 / fs)
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 5))
    else:
        fig = None
    if x.ndim == 2 and axis % x.ndim == 1:
        spec = spec.T
    if semilogx:
        ax.semilogx(f, spec, label=label)
    else:
        ax.plot(f, spec, label=label)
    if xmin is not None:
        ax.set_xlim(left=xmin)
    if xmax is not None:
        ax.set_xlim(right=xmax)
    if ymin is not None:
        ax.set_ylim(bottom=ymin)
    if ymax is not None:
        ax.set_ylim(top=ymax)
    ax.set_xlabel("Frequency (Hz)")
    if dbscale:
        ax.set_ylabel("Magnitude (dB)")
    else:
        ax.set_ylabel("Magnitude")
    if grid:  # grid is enabled when line properties are supplied
        ax.grid(True, which="both", linestyle=":")
    if label is not None:
        ax.legend()
    if fig is not None:
        fig.tight_layout()
    return fig, ax


def plot_waveform(x, fs=1.0, axis=-1, ax=None):
    """Plot a waveform.

    Parameters
    ----------
    x : torch.Tensor or numpy.ndarray
        Input signal. One- or two-dimensional.
    fs : float, optional
        Sampling frequency.
    axis : int, optional
        Time axis of the input signal.
    ax : matplotlib.axes.Axes, optional
        Axes to plot on.

    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure instance.
    ax : matplotlib.axes.Axes
        Axes instance.

    """
    if x.ndim > 2:
        raise ValueError("input must be one- or two-dimensional")
    x = np.asarray(x)
    axis = axis % x.ndim
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 5))
    else:
        fig = None
    t = np.arange(x.shape[axis]) / fs
    if x.ndim == 2 and axis == 1:
        x = x.T
    ax.plot(t, x)
    if fig is not None:
        fig.tight_layout()
    return fig, ax


def boxplot_from_stats(
    meds,
    q1s,
    q3s,
    mins,
    maxs,
    colors=None,
    ax=None,
    grid=True,
    labels=None,
    iqr=False,
):
    """Create a box plot from summary statistics."""
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = None
    if iqr:
        iqrs = [q3 - q1 for q3, q1 in zip(q3s, q1s)]
        whislos = [max(min_, q1 - 1.5 * iqr) for min_, q1, iqr in zip(mins, q1s, iqrs)]
        whishis = [min(max_, q3 + 1.5 * iqr) for max_, q3, iqr in zip(maxs, q3s, iqrs)]
    else:
        whislos = mins
        whishis = maxs
    stats = [
        {
            "med": med,
            "q1": q1,
            "q3": q3,
            "whislo": whislo,
            "whishi": whishi,
            "fliers": [],
        }
        for med, q1, q3, whislo, whishi in zip(meds, q1s, q3s, whislos, whishis)
    ]
    bplots = ax.bxp(stats, patch_artist=True, medianprops={"color": "k"})
    if colors is None:
        colors = SEABORN_DEFAULT_PALETTE
    for i, patch in enumerate(bplots["boxes"]):
        patch.set_facecolor(colors[i % len(colors)])
    if grid:
        ax.grid(True, axis="y", linestyle=":")
    if labels is not None:
        ax.set_xticklabels(labels)
    if fig is not None:
        fig.tight_layout()
    return fig, ax

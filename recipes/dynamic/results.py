import argparse

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

standard_audiograms = {
    "NH": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    "N1": [10, 10, 10, 10, 10, 10, 15, 20, 30, 40],
    "N2": [20, 20, 20, 22.5, 25, 30, 35, 40, 45, 50],
    "N3": [35, 35, 35, 35, 40, 45, 50, 55, 60, 65],
    "N4": [55, 55, 55, 55, 55, 60, 65, 70, 75, 80],
    "N5": [65, 67.5, 70, 72.5, 75, 80, 80, 80, 80, 80],
    "N6": [75, 77.5, 80, 82.5, 85, 90, 90, 95, 100, 100],
    "N7": [90, 92.5, 95, 100, 105, 105, 105, 105, 105, 105],
    "S1": [10, 10, 10, 10, 10, 10, 15, 30, 55, 70],
    "S2": [20, 20, 20, 22.5, 25, 35, 55, 75, 95, 95],
    "S3": [30, 30, 35, 47.5, 60, 70, 75, 80, 80, 85],
}

models = [
    {
        "model": "noisy",
        "loss": None,
        "alpha": None,
        "label": "Noisy",
        "audiogram": "NH",
    },
    {
        "model": "bsrnn-snr",
        "loss": None,
        "alpha": None,
        "label": "BSRNN-SNR",
        "audiogram": "NH",
    },
    {
        "model": "bsrnn-nr",
        "loss": "mse",
        "alpha": None,
        "label": "BSRNN-NR",
        "audiogram": "NH",
    },
    {
        "model": "bsrnn-nr",
        "loss": "mae",
        "alpha": None,
        "label": "BSRNN-NR",
        "audiogram": "NH",
    },
    {
        "model": "bsrnn-hlc",
        "loss": "mse",
        "alpha": None,
        "label": "BSRNN-HLC",
        "audiogram": "NH",
    },
    {
        "model": "bsrnn-hlc",
        "loss": "mae",
        "alpha": None,
        "label": "BSRNN-HLC",
        "audiogram": "NH",
    },
    {
        "model": "bsrnn-nrhlc",
        "loss": "mse",
        "alpha": None,
        "label": "BSRNN-NR-HLC",
        "audiogram": "NH",
    },
    {
        "model": "bsrnn-nrhlc",
        "loss": "mae",
        "alpha": None,
        "label": "BSRNN-NR-HLC",
        "audiogram": "NH",
    },
    {
        "model": "bsrnn-cnrhlc",
        "loss": "mse",
        "alpha": 0.0,
        "label": "BSRNN-C-NR-HLC",
        "audiogram": "NH",
    },
    {
        "model": "bsrnn-cnrhlc",
        "loss": "mae",
        "alpha": 0.0,
        "label": "BSRNN-C-NR-HLC",
        "audiogram": "NH",
    },
    {
        "model": "bsrnn-cnrhlc",
        "loss": "mse",
        "alpha": 0.6,
        "label": "BSRNN-C-NR-HLC",
        "audiogram": "NH",
    },
    {
        "model": "bsrnn-cnrhlc",
        "loss": "mae",
        "alpha": 0.6,
        "label": "BSRNN-C-NR-HLC",
        "audiogram": "NH",
    },
    {
        "model": "bsrnn-cnrhlc",
        "loss": "mse",
        "alpha": 1.0,
        "label": "BSRNN-C-NR-HLC",
        "audiogram": "NH",
    },
    {
        "model": "bsrnn-cnrhlc",
        "loss": "mae",
        "alpha": 1.0,
        "label": "BSRNN-C-NR-HLC",
        "audiogram": "NH",
    },
    {
        "model": "noisy",
        "loss": None,
        "alpha": None,
        "label": "Noisy",
        "audiogram": "HI",
    },
    {
        "model": "noisy-nalr",
        "loss": None,
        "alpha": None,
        "label": "Noisy+NAL-R",
        "audiogram": "HI",
    },
    {
        "model": "bsrnn-snr",
        "loss": None,
        "alpha": None,
        "label": "BSRNN-SNR",
        "audiogram": "HI",
    },
    {
        "model": "bsrnn-snr-nalr",
        "loss": None,
        "alpha": None,
        "label": "BSRNN-SNR+NAL-R",
        "audiogram": "HI",
    },
    {
        "model": "bsrnn-nr",
        "loss": "mse",
        "alpha": None,
        "label": "BSRNN-NR",
        "audiogram": "HI",
    },
    {
        "model": "bsrnn-nr",
        "loss": "mae",
        "alpha": None,
        "label": "BSRNN-NR",
        "audiogram": "HI",
    },
    {
        "model": "bsrnn-nr-nalr",
        "loss": "mse",
        "alpha": None,
        "label": "BSRNN-NR+NAL-R",
        "audiogram": "HI",
    },
    {
        "model": "bsrnn-nr-nalr",
        "loss": "mae",
        "alpha": None,
        "label": "BSRNN-NR+NAL-R",
        "audiogram": "HI",
    },
    {
        "model": "bsrnn-hlc",
        "loss": "mse",
        "alpha": None,
        "label": "BSRNN-HLC",
        "audiogram": "HI",
    },
    {
        "model": "bsrnn-hlc",
        "loss": "mae",
        "alpha": None,
        "label": "BSRNN-HLC",
        "audiogram": "HI",
    },
    {
        "model": "bsrnn-nrhlc",
        "loss": "mse",
        "alpha": None,
        "label": "BSRNN-NR-HLC",
        "audiogram": "HI",
    },
    {
        "model": "bsrnn-nrhlc",
        "loss": "mae",
        "alpha": None,
        "label": "BSRNN-NR-HLC",
        "audiogram": "HI",
    },
    {
        "model": "bsrnn-cnrhlc",
        "loss": "mse",
        "alpha": 0.0,
        "label": "BSRNN-C-NR-HLC",
        "audiogram": "HI",
    },
    {
        "model": "bsrnn-cnrhlc",
        "loss": "mae",
        "alpha": 0.0,
        "label": "BSRNN-C-NR-HLC",
        "audiogram": "HI",
    },
    {
        "model": "bsrnn-cnrhlc",
        "loss": "mse",
        "alpha": 0.6,
        "label": "BSRNN-C-NR-HLC",
        "audiogram": "HI",
    },
    {
        "model": "bsrnn-cnrhlc",
        "loss": "mae",
        "alpha": 0.6,
        "label": "BSRNN-C-NR-HLC",
        "audiogram": "HI",
    },
    {
        "model": "bsrnn-cnrhlc",
        "loss": "mse",
        "alpha": 1.0,
        "label": "BSRNN-C-NR-HLC",
        "audiogram": "HI",
    },
    {
        "model": "bsrnn-cnrhlc",
        "loss": "mae",
        "alpha": 1.0,
        "label": "BSRNN-C-NR-HLC",
        "audiogram": "HI",
    },
]

metrics = [
    "snr",
    "pesq",
    "estoi",
    "haspi",
    "hasqi",
]

snr_ranges = [
    (None, 0),
    (0, 5),
    (5, 10),
    (10, 15),
    (15, None),
]

alphas = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

n_scenes = 1000


def set_style(paper=False):
    if paper:
        plt.rcParams["axes.grid"] = True
        plt.rcParams["axes.labelpad"] = 2.0
        plt.rcParams["axes.linewidth"] = 0.25
        plt.rcParams["figure.dpi"] = 140
        plt.rcParams["figure.figsize"] = [3.2, 1.6]
        plt.rcParams["font.family"] = "Times New Roman"
        plt.rcParams["font.size"] = 7
        plt.rcParams["grid.linestyle"] = [1, 3]
        plt.rcParams["grid.linewidth"] = 0.25
        plt.rcParams["lines.linewidth"] = 0.5
        plt.rcParams["lines.markersize"] = 1
        plt.rcParams["patch.linewidth"] = 0.25  # controls legend frame edge width
        plt.rcParams["mathtext.fontset"] = "cm"
        plt.rcParams["mathtext.rm"] = "serif"
        plt.rcParams["savefig.bbox"] = "tight"
        plt.rcParams["savefig.pad_inches"] = 0.00
        plt.rcParams["xtick.direction"] = "in"
        plt.rcParams["xtick.labelsize"] = "small"
        plt.rcParams["xtick.major.size"] = 1
        plt.rcParams["xtick.major.width"] = 0.25
        plt.rcParams["ytick.direction"] = "in"
        plt.rcParams["ytick.labelsize"] = "small"
        plt.rcParams["ytick.major.size"] = 1
        plt.rcParams["ytick.major.width"] = 0.25
    else:
        sns.set()


def load_file(file):
    try:
        data = np.load(file)
    except FileNotFoundError:
        print(f"File not found: {file}")
        return None
    for audiogram in standard_audiograms.keys():
        for metric in metrics:
            key = f"{metric}.{audiogram}"
            assert key in data, (key, data.keys())
            assert data[key].shape[0] == n_scenes, (data[key].shape[0], n_scenes)
            if metric != "hasqi":
                assert (data[key] != 0).all()
    return data


def load_files():
    files = {}
    for model in set(m["model"] for m in models):
        if model in ["noisy", "noisy-nalr", "bsrnn-snr", "bsrnn-snr-nalr"]:
            files[model] = load_file(f"models/{model}/scores.npz")
        elif model == "bsrnn-nr-nalr":
            files["bsrnn-nr-nalr"] = load_file("models/bsrnn-nr-nalr/scores.npz")
            files["bsrnn-nr-l1-nalr"] = load_file("models/bsrnn-nr-l1-nalr/scores.npz")
        else:
            for loss in ["mse", "mae"]:
                file = f"models/{model + '-l1' if loss == 'mae' else model}/scores.npz"
                files[f"{model}-{loss}"] = load_file(file)
    return files


def get_scores(
    model,
    metric,
    alpha=None,
    loss=None,
    audiogram=None,
    mean=True,
    snr_min=None,
    snr_max=None,
    return_idx=False,
):
    assert loss in ["mse", "mae", None]
    channel = 0
    if model in ["noisy", "noisy-nalr", "bsrnn-snr", "bsrnn-snr-nalr"]:
        data = files[model]
    elif model == "bsrnn-nr-nalr":
        data = files["bsrnn-nr-nalr" if loss == "mae" else "bsrnn-nr-l1-nalr"]
    else:
        data = files[f"{model}-{loss}"]
        if "cnrhlc" in model:
            channel = alphas.index(alpha)
    if data is None:
        return
    if audiogram is None:
        audiograms = standard_audiograms.keys()
    else:
        if audiogram == "HI":
            audiograms = [a for a in standard_audiograms.keys() if a != "NH"]
        else:
            audiograms = [audiogram]
    data = np.stack([data[f"{metric}.{a}"] for a in audiograms], axis=-1)
    idx = np.ones(n_scenes, dtype=bool)
    if snr_min is not None or snr_max is not None:
        snrs = get_scores("noisy", "snr", mean=False)
        if snr_min is not None:
            idx = idx & (snrs >= snr_min)
        if snr_max is not None:
            idx = idx & (snrs < snr_max)
    if mean:
        output = data[idx, channel, :].flatten()
        if metric == "hasqi":
            output = output[output != 0]
        else:
            assert (output != 0).all()
        output = output.mean()
    else:
        output = data[idx, channel, :].mean(-1)
    if return_idx:
        return output, idx
    else:
        return output


def violinplots():
    for audiogram in ["NH", "HI"]:
        for metric in metrics:
            data = {}
            for model in models:
                if model["audiogram"] != audiogram:
                    continue
                key = (
                    f"{model['model']}"
                    f"{'-l1' if model['loss'] == 'mae' else ''}"
                    f"{f'-{model['alpha']}' if model['alpha'] is not None else ''}"
                )
                data[key] = get_scores(
                    model["model"],
                    metric,
                    alpha=model["alpha"],
                    loss=model["loss"],
                    audiogram=model["audiogram"],
                    mean=False,
                )
            data = {k: v for k, v in data.items() if v is not None}
            fig, ax = plt.subplots()
            sns.violinplot(data=data, ax=ax, cut=0)
            ax.set_ylabel(metric)
            plt.setp(
                ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor"
            )
            plt.title(f"{audiogram}")
            fig.tight_layout()


def alphaplots(
    metrics=["haspi", "hasqi"],
    losses=["mse", "mae"],
    snr_ranges=None,
    audiogram=None,
    save_to=None,
    title=True,
):
    snr_ranges = snr_ranges or [(None, None)]
    fig, axes = plt.subplots(1, len(metrics))
    for ax, metric in zip(axes, metrics):
        for snr_min, snr_max in snr_ranges:
            for loss in losses:
                if len(losses) > 1:
                    if len(snr_ranges) > 1:
                        label = rf"{loss.upper()}, SNR $\in$ [{snr_min}, {snr_max}]"
                    else:
                        label = loss.upper()
                else:
                    if len(snr_ranges) > 1:
                        label = rf"SNR $\in$ [{snr_min}, {snr_max}]"
                    else:
                        label = "bsrnn-c-nr-hlc".upper()
                ax.plot(
                    alphas,
                    [
                        get_scores(
                            "bsrnn-cnrhlc",
                            metric,
                            alpha=a,
                            loss=loss,
                            snr_min=snr_min,
                            snr_max=snr_max,
                            audiogram=audiogram,
                        )
                        for a in alphas
                    ],
                    label=label,
                    color=None if len(losses) > 1 or len(snr_ranges) > 1 else "black",
                )
        linestyles = ["--", "-."]
        for loss, linestyle in zip(losses, linestyles):
            label = f"bsrnn-nr-hlc, {loss}" if len(losses) > 1 else "bsrnn-nr-hlc"
            ax.axhline(
                get_scores("bsrnn-nrhlc", metric, loss=loss, audiogram=audiogram),
                color="black",
                linestyle=linestyle,
                label=label.upper(),
            )
        ax.axhline(
            get_scores("noisy", metric, audiogram=audiogram),
            color="black",
            linestyle=":",
            label="Noisy",
        )
        ax.set_xlabel(r"$\alpha$")
        ax.set_ylabel(metric.upper())
        ax.set_xlim(0.0, 1.0)
        if title:
            ax.set_title(f"audiogram={audiogram}")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, ncol=3, loc="upper center", columnspacing=1)
    fig.tight_layout(rect=[0, 0, 1, 0.85])
    if save_to is not None:
        plt.savefig(save_to)


def heatmap(model, metric, loss=None, alpha=None):
    data = np.zeros((len(standard_audiograms), len(snr_ranges)))
    count = np.zeros((len(standard_audiograms), len(snr_ranges)))
    for i, audiogram in enumerate(standard_audiograms.keys()):
        for j, (snr_min, snr_max) in enumerate(snr_ranges):
            scores = get_scores(
                model,
                metric,
                audiogram=audiogram,
                snr_min=snr_min,
                snr_max=snr_max,
                loss=loss,
                alpha=alpha,
                mean=False,
            )
            count[i, j] = scores.size
            data[i, j] = scores.mean()
    fig, ax = plt.subplots()
    sns.heatmap(
        data,
        ax=ax,
        xticklabels=[
            f"[{a[0]}, {a[1]}]"
            if a[0] is not None and a[1] is not None
            else f">{a[0]}"
            if a[0] is not None
            else f"<{a[1]}"
            if a[1] is not None
            else "all"
            for a in snr_ranges
        ],
        yticklabels=standard_audiograms,
        # annot=count,
        # fmt=".0f",
        annot=True,
        fmt=".2f",
    )
    plt.xlabel("SNR")
    plt.ylabel("Audiogram")
    plt.title(f"{model} {metric} {loss} {alpha}")
    fig.tight_layout()


def metric_vs_metric(
    x_metric,
    y_metric,
    model,
    loss=None,
    alpha=None,
    audiogram=None,
    snr_min=None,
    snr_max=None,
    delta=True,
    x_min=None,
    x_max=None,
    y_min=None,
    y_max=None,
    dx_min=None,
    dx_max=None,
    dy_min=None,
    dy_max=None,
):
    x, idxs = get_scores(
        model,
        x_metric,
        mean=False,
        loss=loss,
        alpha=alpha,
        snr_min=snr_min,
        snr_max=snr_max,
        audiogram=audiogram,
        return_idx=True,
    )
    y = get_scores(
        model,
        y_metric,
        mean=False,
        loss=loss,
        alpha=alpha,
        snr_min=snr_min,
        snr_max=snr_max,
        audiogram=audiogram,
    )
    dx = x - get_scores(
        "noisy",
        x_metric,
        mean=False,
        snr_min=snr_min,
        snr_max=snr_max,
        audiogram=audiogram,
    )
    dy = y - get_scores(
        "noisy",
        y_metric,
        mean=False,
        snr_min=snr_min,
        snr_max=snr_max,
        audiogram=audiogram,
    )
    f = np.ones_like(x, dtype=bool)
    if x_min is not None:
        f &= x >= x_min
    if x_max is not None:
        f &= x < x_max
    if y_min is not None:
        f &= y >= y_min
    if y_max is not None:
        f &= y < y_max
    if dx_min is not None:
        f &= dx >= dx_min
    if dx_max is not None:
        f &= dx < dx_max
    if dy_min is not None:
        f &= dy >= dy_min
    if dy_max is not None:
        f &= dy < dy_max
    x, y, dx, dy = x[f], y[f], dx[f], dy[f]
    if delta:
        x, y = dx, dy
    plt.figure()
    plt.plot(x, y, linestyle="", marker="o")
    # annotate indexes
    idxs = np.where(idxs)[0][f]
    for i, idx in enumerate(idxs):
        plt.annotate(idx, (x[i], y[i]))
    plt.xlabel((r"$\Delta$" if delta else "") + x_metric.upper())
    plt.ylabel((r"$\Delta$" if delta else "") + y_metric.upper())


def table():
    audiograms = ["NH", "HI"]
    all_scores = [
        [
            [
                get_scores(
                    model["model"],
                    met,
                    alpha=model["alpha"],
                    loss=model["loss"],
                    audiogram=model["audiogram"],
                )
                or float("nan")
                for met in metrics
            ]
            for model in models
            if model["audiogram"] == audiogram
        ]
        for audiogram in audiograms
    ]
    all_scores = [np.array(all_scores[0]), np.array(all_scores[1])]
    i_max = [
        np.argmax(np.nan_to_num(all_scores[0], nan=-np.inf), axis=0),
        np.argmax(np.nan_to_num(all_scores[1], nan=-np.inf), axis=0),
    ]
    print(r"\begin{table}")
    print(r"  \scriptsize")
    print(r"  \setlength{\tabcolsep}{1.68pt}")
    print(
        r"  \caption{Objective metrics for the different speech processor "
        r"configurations.}"
    )
    print(r"  \label{tab:results}")
    print(r"  \centering")
    print(r"  \begin{tabular}{@{}lcccSSSSS@{}}")
    print(r"    \toprule")
    print(
        r" & ".join(
            ["   ", r"$\ell$", r"$\alpha$", r"$a$"]
            + [f"{{{metric.upper()}}}" for metric in metrics]
        )
        + r" \\"
    )
    print(r"    \midrule")
    for i_aud, audiogram in enumerate(audiograms):
        i_model = 0
        for model in models:
            if model["audiogram"] != audiogram:
                continue
            print(
                r" & ".join(
                    [
                        f"    {model['label']}",
                        f"{'-' if model['loss'] is None else model['loss'].upper()}",
                        f"{'-' if model['alpha'] is None else model['alpha']}",
                        f"{audiogram}",
                    ]
                    + [
                        rf"\bfseries {all_scores[i_aud][i_model, i_metric]:.2f}"
                        if i_model == i_max[i_aud][i_metric]
                        else f"{all_scores[i_aud][i_model, i_metric]:.2f}"
                        for i_metric in range(len(metrics))
                    ]
                )
                + r" \\"
            )
            i_model += 1
        if i_aud != len(audiograms) - 1:
            print(r"    \midrule")
    print(r"    \bottomrule")
    print(r"  \end{tabular}")
    print(r"\end{table}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--paper", action="store_true")
    args = parser.parse_args()

    set_style(paper=args.paper)

    files = load_files()

    table()

    if args.paper:
        alphaplots(audiogram="HI", losses=["mae"], save_to="alphaplot.pdf", title=False)
        plt.show()
        exit()

    violinplots()

    alphaplots()
    alphaplots(audiogram="HI")
    alphaplots(losses=["mae"], snr_ranges=snr_ranges)

    heatmap("bsrnn-nrhlc", "haspi", "mae")
    heatmap("noisy", "haspi")

    metric_vs_metric(
        "snr",
        "haspi",
        "bsrnn-cnrhlc",
        loss="mae",
        alpha=1.0,
        audiogram="NH",
        snr_max=0.0,
        delta=True,
    )
    metric_vs_metric(
        "snr",
        "estoi",
        "bsrnn-cnrhlc",
        loss="mae",
        alpha=1.0,
        audiogram="NH",
        snr_max=0.0,
        delta=True,
    )
    metric_vs_metric(
        "estoi",
        "haspi",
        "bsrnn-cnrhlc",
        loss="mae",
        alpha=1.0,
        audiogram="NH",
        snr_max=0.0,
        delta=True,
    )

    plt.show()

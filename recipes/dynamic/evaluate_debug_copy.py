import argparse
import logging
import os

import numpy as np
import torch
import torchaudio
from tqdm import tqdm

from mbchl.has import HARegistry
from mbchl.metrics import MetricRegistry
from mbchl.training.ema import EMARegistry
from mbchl.training.losses import ControllableNoiseReductionHearingLossCompensationLoss
from mbchl.utils import read_yaml

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
audiogram_freqs = [250, 375, 500, 750, 1000, 1500, 2000, 3000, 4000, 6000]


def peaknorm(x):
    return x / x.abs().max()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    custom_args = [
        "/Users/yue/Documents/Thesis/Codes/thesis/recipes/cnrhlc/models/bsrnn-cnrhlc-l1/checkpoints/last.ckpt",
        "/Users/yue/Documents/Thesis/Codes/thesis/recipes/cnrhlc/models/testset/",
        "--cfg",
        "/Users/yue/Documents/Thesis/Codes/thesis/recipes/cnrhlc/models/bsrnn-cnrhlc-l1/config.yaml"
                   ]
    parser = argparse.ArgumentParser()
    parser.add_argument("ckpt")
    parser.add_argument("testset")
    parser.add_argument("--cuda", action="store_true")
    parser.add_argument("--use_amp", action="store_true")
    parser.add_argument("--metrics", nargs="+", default=["haspi", "hasqi"])
    parser.add_argument("--cfg")
    args = parser.parse_args(custom_args)

    assert "haspi" in args.metrics
    assert "hasqi" in args.metrics
    # assert os.path.exists("debug")

    if "haspi" in args.metrics:
        from clarity.evaluator.haspi import haspi_v2
        from clarity.utils.audiogram import Audiogram
    if "hasqi" in args.metrics:
        from clarity.evaluator.hasqi import hasqi_v2
        from clarity.utils.audiogram import Audiogram

    # load config
    dirname = os.path.dirname(args.ckpt)
    if args.cfg is None:
        cfg_path = os.path.join(dirname, "..", "config.yaml")
    else:
        cfg_path = args.cfg
    cfg = read_yaml(cfg_path)

    # initialize model
    model = HARegistry.init(cfg["ha"], **cfg["ha_kw"])

    # move model to device
    device = "cuda" if args.cuda else "cpu"
    model.to(device)

    # load checkpoint
    logging.info(f"Loading checkpoint {args.ckpt}")
    state = torch.load(args.ckpt, map_location=device, weights_only=True)
    model.load_state_dict(state["model"])

    # load EMA
    if "ema" in cfg["trainer"] and cfg["trainer"]["ema"] is not None:
        ema_kw = cfg["trainer"]["ema_kw"]
        ema_cls = EMARegistry.get(cfg["trainer"]["ema"])
        ema_obj = ema_cls(model, **(ema_kw or {}))
        ema_obj.load_state_dict(state["ema"])
        ema_obj.apply()

    # disable gradients and set model to eval
    torch.set_grad_enabled(False)
    model.eval()

    # list files
    in_files = [f for f in os.listdir(args.testset) if f.endswith("_mix.wav")]
    in_files = [f.split("_")[0] for f in in_files]
    in_files = [(f"{f}_mix.wav", f"{f}_target.wav") for f in sorted(in_files)]
    in_files = in_files[:1]

    # main loop
    outfile = os.path.normpath(os.path.join(dirname, "..", "scores.npz"))
    logging.info(f"Writing scores in {outfile}")

    def process_file(f):
        x, fs = torchaudio.load(os.path.join(args.testset, f[0]))
        assert fs == cfg["dataset"]["val_kw"]["fs"] == cfg["dataset"]["train_kw"]["fs"]
        y, fs = torchaudio.load(os.path.join(args.testset, f[1]))
        assert fs == cfg["dataset"]["val_kw"]["fs"] == cfg["dataset"]["train_kw"]["fs"]
        assert y.shape[0] == 1
        x, y = x.to(device), y.to(device)
        score = {}
        for profile, thresholds in standard_audiograms.items():
            print(f"Processing {f[0]} with profile {profile}")
            audiogram = torch.tensor(
                list(zip(audiogram_freqs, thresholds)), device=device
            )
            if model._audiogram:
                extra_inputs = [audiogram]
            else:
                extra_inputs = None
            output = model.enhance(x, extra_inputs=extra_inputs, use_amp=args.use_amp)
            if isinstance(
                model._loss, ControllableNoiseReductionHearingLossCompensationLoss
            ):
                assert output.shape[0] == 2  # compensated, denoised
                y = y.expand(2, -1)
            else:
                assert output.shape[0] == 1
            score = {}
            for j, metric in enumerate(args.metrics):
                if metric in ["haspi", "hasqi"]:
                    if metric == "haspi":
                        metric_func = haspi_v2
                    if metric == "hasqi":
                        metric_func = hasqi_v2
                    audiogram_obj = Audiogram(
                        frequencies=audiogram[:, 0], levels=audiogram[:, 1]
                    )
                    score[f"{metric}.{profile}"] = [
                        metric_func(
                            reference=y[i].cpu().numpy(),
                            reference_sample_rate=fs,
                            processed=output[i].cpu().numpy(),
                            processed_sample_rate=fs,
                            audiogram=audiogram_obj,
                        )[0]
                        for i in range(output.shape[0])
                    ]
                else:
                    metric_kw = {}
                    if metric == "pesq":
                        metric_kw["multiprocessing"] = False
                    if metric in ["pesq", "estoi"]:
                        metric_kw["fs"] = fs
                    metric_func = MetricRegistry.get(metric)(**metric_kw)
                    score[f"{metric}.{profile}"] = metric_func(output, y)

            if isinstance(
                model._loss, ControllableNoiseReductionHearingLossCompensationLoss
            ):
                import matplotlib.pyplot as plt

                noisy_nh = model._loss.am_nh(x[0])
                noisy_hi = model._loss.am_hi(x[0], audiogram=audiogram)
                comp_hi = model._loss.am_hi(output[0], audiogram=audiogram)

                basename = f[0].split("_")[0]

                torchaudio.save(
                    f"debug/output_{basename}_{profile}.wav",
                    peaknorm(output[0:1]).cpu(),
                    fs,
                )
                torchaudio.save(
                    f"debug/input_{basename}.wav", peaknorm(x[0:1].cpu()), fs
                )
                torchaudio.save(
                    f"debug/target_{basename}.wav", peaknorm(y[0:1].cpu()), fs
                )

                input_haspi = haspi_v2(
                    reference=y[0].cpu().numpy(),
                    reference_sample_rate=fs,
                    processed=x[0].cpu().numpy(),
                    processed_sample_rate=fs,
                    audiogram=audiogram_obj,
                )[0]
                input_hasqi = hasqi_v2(
                    reference=y[0].cpu().numpy(),
                    reference_sample_rate=fs,
                    processed=x[0].cpu().numpy(),
                    processed_sample_rate=fs,
                    audiogram=audiogram_obj,
                )[0]
                output_haspi = score[f"haspi.{profile}"][0]
                output_hasqi = score[f"hasqi.{profile}"][0]

                input_score = f"haspi={input_haspi:.3f}, hasqi={input_hasqi:.3f}"
                output_score = f"haspi={output_haspi:.3f}, hasqi={output_hasqi:.3f}"

                from mbchl.plot import plot_spectrum

                figspec, axspec = plt.subplots(1, 1, figsize=(12, 8))
                plot_spectrum(
                    x[0].cpu().numpy(),
                    fs=fs,
                    dbscale=True,
                    semilogx=True,
                    ax=axspec,
                    label=f"input, {input_score}",
                )
                plot_spectrum(
                    output[0].cpu().numpy(),
                    fs=fs,
                    dbscale=True,
                    semilogx=True,
                    ax=axspec,
                    label=f"compensated, {output_score}",
                )
                figspec.savefig(f"debug/spectrum_{basename}_{profile}.png")

                fig0, ax = plt.subplots(2, 1, figsize=(14, 8))
                ax[0].plot(x[0].cpu().numpy(), color="tab:blue")
                ax[1].plot(output[0].cpu().numpy(), color="tab:orange")
                ax[0].set_title(f"input, {input_score}")
                ax[1].set_title(f"compensated, {output_score}")
                fig0.tight_layout()
                fig0.savefig(f"debug/waveforms_{basename}_{profile}.png")

                fig1, ax = plt.subplots()
                plt.plot(-audiogram[:, 1])
                plt.xticks(range(len(audiogram)), audiogram[:, 0].int().tolist())
                plt.title(f"audiogram {profile}")
                plt.grid(linestyle=":")
                fig1.savefig(f"debug/audiogram_{basename}_{profile}.png")

                seg_len = 1600

                for i in range(0, len(x[0]), seg_len):
                    fig2, ax = plt.subplots(3, 1, figsize=(12, 8))
                    ax[0].imshow(
                        noisy_nh[:, i : i + seg_len].cpu().numpy(),
                        aspect="auto",
                        origin="lower",
                    )
                    ax[1].imshow(
                        noisy_hi[:, i : i + seg_len].cpu().numpy(),
                        aspect="auto",
                        origin="lower",
                    )
                    ax[2].imshow(
                        comp_hi[:, i : i + seg_len].cpu().numpy(),
                        aspect="auto",
                        origin="lower",
                    )
                    ax[0].set_title("input NH internal representation")
                    ax[1].set_title(f"input {profile} internal representation")
                    ax[2].set_title(f"compensated {profile} internal representation")
                    fig2.tight_layout()
                    fig2.savefig(f"debug/am_output_{basename}_{profile}_{i}.png")

                    # plt.show()
                    # breakpoint()
                    plt.close("all")
                    break

        return score

    scores = [process_file(f) for f in tqdm(in_files)]
    scores = {k: np.array([s[k] for s in scores]) for k in scores[0].keys()}
    # np.savez(outfile, **scores)

    for k in scores.keys():
        for j in range(scores[k].shape[1]):
            logging.info(f"{k} ch {j}: {scores[k][:, j].mean():.3f}")

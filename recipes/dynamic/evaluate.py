import argparse
import logging
import multiprocessing
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

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument("ckpt")
    parser.add_argument("testset")
    parser.add_argument("--cuda", action="store_true")
    parser.add_argument("--use_amp", action="store_true")
    parser.add_argument("--metrics", nargs="+", default=["pesq", "estoi", "snr"])
    parser.add_argument("--cfg")
    parser.add_argument("--write_wav", action="store_true")
    parser.add_argument("--normalize", action="store_true")
    parser.add_argument("--nalr", action="store_true")
    parser.add_argument("--n_cpu", type=int, default=0)
    parser.add_argument("--noisy", action="store_true")
    parser.add_argument("--noisy_output_dir")
    parser.add_argument("--alphas", nargs="+", type=float)
    parser.add_argument("--filenum")
    parser.add_argument("-n", type=int)
    args = parser.parse_args()

    if "haspi" in args.metrics:
        from clarity.evaluator.haspi import haspi_v2
        from clarity.utils.audiogram import Audiogram
    if "hasqi" in args.metrics:
        from clarity.evaluator.hasqi import hasqi_v2
        from clarity.utils.audiogram import Audiogram
    if args.nalr:
        from clarity.enhancer.nalr import NALR
        from clarity.utils.audiogram import Audiogram

    if args.noisy:
        if args.noisy_output_dir is None:
            raise ValueError("--noisy_output_dir must be provided when --noisy is used")
        if not os.path.exists(args.noisy_output_dir):
            os.makedirs(args.noisy_output_dir)

    if args.alphas is None:
        args.alphas = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

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
    if args.filenum is None:
        in_files = [f for f in os.listdir(args.testset) if f.endswith("_mix.wav")]
        in_files = [f.split("_")[0] for f in in_files]
        in_files = [(f"{f}_mix.wav", f"{f}_target.wav") for f in sorted(in_files)]
        if args.n is not None:
            in_files = in_files[: args.n]
    else:
        in_files = [(f"{args.filenum}_mix.wav", f"{args.filenum}_target.wav")]

    # main loop
    if args.noisy:
        outdirname = args.noisy_output_dir
    else:
        outdirname = os.path.join(dirname, "..")
    outfile = os.path.normpath(os.path.join(outdirname, "scores.npz"))
    logging.info(f"Writing scores in {outfile}")
    if args.write_wav:
        audio_dir = os.path.normpath(os.path.join(outdirname, "audio"))
        os.makedirs(audio_dir, exist_ok=True)
        logging.info(f"Writing wav files in {audio_dir}")

    def process_file(f):
        x, fs = torchaudio.load(os.path.join(args.testset, f[0]))
        assert fs == cfg["dataset"]["val_kw"]["fs"] == cfg["dataset"]["train_kw"]["fs"]
        y, fs = torchaudio.load(os.path.join(args.testset, f[1]))
        assert fs == cfg["dataset"]["val_kw"]["fs"] == cfg["dataset"]["train_kw"]["fs"]
        assert y.shape[0] == 1
        x, y = x.to(device), y.to(device)
        score = {}
        for profile, thresholds in standard_audiograms.items():
            audiogram = torch.tensor(
                list(zip(audiogram_freqs, thresholds)), device=device
            )
            if model._audiogram:
                extra_inputs = [audiogram]
            else:
                extra_inputs = None
            if args.noisy:
                output = x
            else:
                output = model.enhance(
                    x, extra_inputs=extra_inputs, use_amp=args.use_amp
                )
            if (
                isinstance(
                    model._loss, ControllableNoiseReductionHearingLossCompensationLoss
                )
                and not args.noisy
            ):
                assert output.shape[0] == 2  # compensated, denoised
                output = torch.stack(
                    [
                        alpha * output[1] + (1 - alpha) * output[0]
                        for alpha in args.alphas
                    ]
                )
                y = y.expand(len(args.alphas), -1)
            else:
                assert output.shape[0] == 1
            if "haspi" in args.metrics or "hasqi" in args.metrics or args.nalr:
                audiogram_obj = Audiogram(
                    frequencies=audiogram[:, 0], levels=audiogram[:, 1]
                )
            if args.nalr:
                nalr = NALR(80, fs)
                nalr_fir, _ = nalr.build(audiogram_obj)
                assert output.shape[0] == 1
                output = nalr.apply(nalr_fir, output[0])[None, : output.shape[-1]]
                output = torch.from_numpy(output)
            for j, metric in enumerate(args.metrics):
                if metric in ["haspi", "hasqi"]:
                    if metric == "haspi":
                        metric_func = haspi_v2
                    if metric == "hasqi":
                        metric_func = hasqi_v2
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
            if args.write_wav:
                basename = f[0].split("_")[0]
                for i in range(output.shape[0]):
                    if args.normalize:
                        to_write = output[i : i + 1] / output[i : i + 1].abs().max()
                    else:
                        to_write = output[i : i + 1]
                    torchaudio.save(
                        os.path.join(audio_dir, f"{basename}_out_{profile}_{i}.wav"),
                        to_write.cpu(),
                        fs,
                    )
        return score

    if args.n_cpu == 0:
        scores = [process_file(f) for f in tqdm(in_files)]
    else:
        n_cpu = min(args.n_cpu, len(in_files))
        logging.info(f"Using {n_cpu} CPUs")
        torch.set_num_threads(1)  # prevents hanging
        torch.multiprocessing.set_sharing_strategy("file_system")
        with multiprocessing.Pool(n_cpu) as p:
            scores = list(tqdm(p.imap(process_file, in_files), total=len(in_files)))
    scores = {k: np.array([s[k] for s in scores]) for k in scores[0].keys()}

    np.savez(outfile, **scores)

    for k in scores.keys():
        for j in range(scores[k].shape[1]):
            logging.info(f"{k} ch {j}: {scores[k][:, j].mean():.3f}")

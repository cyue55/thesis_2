import argparse
import logging
import os

import numpy as np
import torch
import torchaudio
from tqdm import tqdm

from mbchl.data.datasets import AudioDataset
from mbchl.has import HARegistry
from mbchl.metrics import MetricRegistry
from mbchl.training.ema import EMARegistry
from mbchl.utils import read_yaml


def main():
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

    # initialize dataset
    logging.info("Initializing dataset")
    dataset = AudioDataset(
        dirs=[
            "../../data/vbdemand/noisy_testset_wav",
            "../../data/vbdemand/clean_testset_wav",
        ],
        segment_length=None,
        fs=cfg["dataset"]["train_kw"]["fs"],
        seed=cfg["dataset"]["train_kw"]["seed"],
        n_files=args.n_files,
    )

    # main loop
    if not args.no_wav:
        wav_dir = os.path.join(args.output, "wav")
        os.makedirs(wav_dir, exist_ok=True)
    logging.info(f"Writing output files in {wav_dir}")
    scores = np.empty((len(dataset), len(args.metrics), 2))
    for i in tqdm(range(len(dataset))):
        signals, metadata = dataset.get_item(i, _return_metadata=True)
        signals = [s.to(device) for s in signals]
        input_, target = signals
        output = model.enhance(input_, use_amp=args.use_amp)
        if not args.no_wav:
            basename = os.path.splitext(metadata[0]["filename"])[0]
            to_save = [(output, f"{basename}_out.wav")]
            if args.save_input:
                to_save.append((input_, f"{basename}_in.wav"))
            if args.save_target:
                to_save.append((target, f"{basename}_target.wav"))
            for signal, filename in to_save:
                if args.normalize:
                    signal = signal / signal.abs().max()
                torchaudio.save(
                    os.path.join(wav_dir, filename),
                    signal.cpu().float(),
                    cfg["dataset"]["train_kw"]["fs"],
                )
        if args.evaluate:
            for j, metric in enumerate(args.metrics):
                metric_func = MetricRegistry.get(metric)()
                in_score = metric_func(input_, target)
                out_score = metric_func(output, target)
                scores[i, j, 0] = in_score.item()
                scores[i, j, 1] = out_score.item()

    if args.evaluate:
        score_file = os.path.join(args.output, "scores.npy")
        logging.info(f"Writing scores in {score_file}")
        np.save(score_file, scores)
        for j, metric in enumerate(args.metrics):
            in_score = scores[:, j, 0]
            out_score = scores[:, j, 1]
            delta = out_score - in_score
            logging.info(f"{metric}:")
            logging.info(f"  Input:  {in_score.mean():.3f} ± {in_score.std():.3f}")
            logging.info(f"  Output: {out_score.mean():.3f} ± {out_score.std():.3f}")
            logging.info(f"  Delta:  {delta.mean():.3f} ± {delta.std():.3f}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument("ckpt")
    parser.add_argument("output")
    parser.add_argument("--cuda", action="store_true")
    parser.add_argument("--n_files", default="all")
    parser.add_argument("--use_amp", action="store_true")
    parser.add_argument("--save_input", action="store_true")
    parser.add_argument("--save_target", action="store_true")
    parser.add_argument("--normalize", action="store_true")
    parser.add_argument("--evaluate", action="store_true")
    parser.add_argument("--metrics", nargs="+", default=["pesq", "estoi", "snr"])
    parser.add_argument("--no_wav", action="store_true")
    parser.add_argument("--cfg")
    args = parser.parse_args()

    if args.n_files != "all":
        args.n_files = int(args.n_files)

    main()

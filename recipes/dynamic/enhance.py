import argparse
import logging
import math
import os

import matplotlib.pyplot as plt
import torch
import torchaudio
from tqdm import tqdm

from mbchl.data.datasets import DynamicAudioDataset
from mbchl.has import HARegistry
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
    del cfg["dataset"]["val_kw"]["length"]
    dataset = DynamicAudioDataset(length=args.length, **cfg["dataset"]["val_kw"])

    # main loop
    logging.info(f"Writing output files in {args.output}")
    os.makedirs(args.output, exist_ok=True)
    digits = int(math.log10(args.length)) + 1
    for i, signals in enumerate(tqdm(dataset)):
        signals = [s.to(device) for s in signals if isinstance(s, torch.Tensor)]
        if model._audiogram:
            x, y, audiogram = signals
        elif model.spk_adapt_net is not None:
            x, y, spk_adapt = signals
            audiogram = None
        else:
            x, y = signals
            audiogram = None
        output = model.enhance(x, extra_inputs=signals[2:], use_amp=args.use_amp)
        basename = f"{i:0{digits}d}"
        for signal, filename in [
            (x, f"{basename}_in.wav"),
            (y, f"{basename}_target.wav"),
            (output, f"{basename}_out.wav"),
        ]:
            if args.normalize:
                signal = signal / signal.abs().max()
            torchaudio.save(
                os.path.join(args.output, filename),
                signal.cpu().float(),
                cfg["dataset"]["val_kw"]["fs"],
            )
        if audiogram is not None:
            fig, ax = plt.subplots()
            audiogram = audiogram.cpu().numpy()
            ax.plot(-audiogram[:, 1])
            ax.set_xticklabels(audiogram[:, 0])
            figpath = os.path.join(args.output, f"{basename}_audiogram.png")
            fig.savefig(figpath)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument("ckpt")
    parser.add_argument("output")
    parser.add_argument("--cuda", action="store_true")
    parser.add_argument("--length", type=int, default=10)
    parser.add_argument("--use_amp", action="store_true")
    parser.add_argument("--normalize", action="store_true")
    parser.add_argument("--cfg")
    args = parser.parse_args()

    main()

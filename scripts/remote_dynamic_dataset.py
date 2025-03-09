import argparse
import logging
import os

import torch
import torchaudio
from dotenv import load_dotenv
from tqdm import tqdm

from mbchl.data.dataloader import AudioDataLoader
from mbchl.data.datasets import DNS5RIRDataset, DynamicAudioDataset

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("speech_url")
    parser.add_argument("n_speech_archives", type=int)
    parser.add_argument("noise_url")
    parser.add_argument("n_noise_archives", type=int)
    parser.add_argument("--segment_length", type=float)
    parser.add_argument("--length", type=int, default=10)
    parser.add_argument("--fs", type=int, default=16000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--audiogram", action="store_true")
    parser.add_argument("--audiogram_jitter", type=float, default=10.0)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--dns5rir_path")
    parser.add_argument("--snr_range", nargs=2, type=float, default=[-10.0, 20.0])
    parser.add_argument("-o", "--output_dir")
    args = parser.parse_args()

    if args.debug:
        logging.getLogger("root").setLevel(logging.DEBUG)
        logging.getLogger("asyncssh").setLevel(logging.WARNING)
        logging.getLogger("fsspec").setLevel(logging.WARNING)
        logging.getLogger("urllib3").setLevel(logging.WARNING)
        logging.getLogger("botocore").setLevel(logging.WARNING)

    if args.output_dir is not None and not os.path.exists(args.output_dir):
        raise OSError(f"{args.output_dir} does not exist")

    load_dotenv()

    if args.dns5rir_path is None:
        _rir_dset = None
    else:
        _rir_dset = DNS5RIRDataset(args.dns5rir_path, args.fs)

    dataset = DynamicAudioDataset(
        length=args.length,
        fs=args.fs,
        speech_dataset="remote",
        speech_dataset_kw={
            "url": args.speech_url,
            "n_archives": args.n_speech_archives,
            "loop": True,
            "tensor": False,
        },
        noise_dataset="remote",
        noise_dataset_kw={
            "url": args.noise_url,
            "n_archives": args.n_noise_archives,
            "loop": True,
            "tensor": False,
        },
        segment_length=args.segment_length,
        snr_range=args.snr_range,
        seed=args.seed,
        audiogram=args.audiogram,
        audiogram_jitter=args.audiogram_jitter,
        _rir_dset=_rir_dset,
    )

    dataloader = AudioDataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.workers,
        persistent_workers=args.workers > 0,
    )

    k = 0
    for i in range(args.epochs):
        print(f"Epoch {i}")
        for batch, lengths in dataloader if args.debug else tqdm(dataloader):
            if args.output_dir is not None:
                for j, (signals, length) in enumerate(zip(zip(*batch), lengths)):
                    x, y = signals[:2]
                    torchaudio.save(
                        os.path.join(args.output_dir, f"{k:06d}_mix.wav"),
                        x[..., : length[0]],
                        args.fs,
                    )
                    torchaudio.save(
                        os.path.join(args.output_dir, f"{k:06d}_target.wav"),
                        y[..., : length[1]],
                        args.fs,
                    )
                    if args.audiogram:
                        audiogram = signals[2]
                        torch.save(
                            audiogram,
                            os.path.join(args.output_dir, f"{k:06d}_audiogram.pt"),
                        )
                    k += 1

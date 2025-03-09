import argparse
import logging

from dotenv import load_dotenv
from tqdm import tqdm

from mbchl.data.dataloader import AudioDataLoader
from mbchl.data.datasets import RemoteAudioDataset

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("url", nargs="+")
    parser.add_argument("--n_archives", type=int, required=True)
    parser.add_argument("--n_files", type=int)
    parser.add_argument("--loop", action="store_true")
    parser.add_argument("--buffer_size", type=int)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--dataloader_buffer_size", type=int)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    if args.debug:
        logging.getLogger("root").setLevel(logging.DEBUG)
        logging.getLogger("asyncssh").setLevel(logging.WARNING)
        logging.getLogger("fsspec").setLevel(logging.WARNING)
        logging.getLogger("urllib3").setLevel(logging.WARNING)
        logging.getLogger("botocore").setLevel(logging.WARNING)

    load_dotenv()

    dataset = RemoteAudioDataset(
        args.url,
        args.n_archives,
        n_files=args.n_files,
        buffer_size=args.buffer_size,
        loop=args.loop,
        seed=args.seed,
    )

    dataloader = AudioDataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.workers,
        persistent_workers=args.workers > 0,
        buffer_size=args.dataloader_buffer_size,
    )

    for i in range(args.epochs):
        print(f"Epoch {i}")
        for x, length in dataloader if args.debug else tqdm(dataloader):
            pass

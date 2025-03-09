import argparse
import os
import sys
from multiprocessing import Pool, cpu_count

import soundfile as sf
from tqdm import tqdm

from mbchl.utils import find_files, split_list

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("indir")
    parser.add_argument("-o", "--outdir")
    parser.add_argument("--inext", default=".wav")
    parser.add_argument("--outext", default=".flac")
    args = parser.parse_args()

    if args.outdir is None:
        outdir = os.path.normpath(args.indir) + "-" + args.outext.lstrip(".")
    else:
        outdir = args.outdir

    print(f"Scanning {args.indir}")
    all_files = list(find_files(args.indir, ext=args.inext))

    n_cpu = cpu_count()
    split_files = list(enumerate(split_list(all_files, n_cpu)))

    assert len(split_files) == n_cpu
    assert sum(len(files) for _, files in split_files) == len(all_files)

    print(f"Converting {len(all_files)} files with {n_cpu} CPUs")

    if all_files and not os.path.exists(outdir):
        os.makedirs(outdir)

    def process_files(files):
        i_proc, files = files
        bad_files = []
        for file in tqdm(files, position=i_proc, file=sys.stdout):
            out_path = os.path.splitext(file)[0] + "." + args.outext.lstrip(".")
            out_path = os.path.join(outdir, os.path.relpath(out_path, args.indir))
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            try:
                x, fs = sf.read(file)
            except sf.LibsndfileError:
                bad_files.append(file)
            sf.write(out_path, x, fs)
        return bad_files

    with Pool(cpu_count()) as p:
        results = p.map(process_files, split_files)

    bad_files = [file for files in results for file in files]
    print(f"Failed to convert {len(bad_files)} files:")
    for file in bad_files:
        print(file)

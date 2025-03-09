import argparse
import os
import re
import tarfile
from multiprocessing import Pool, cpu_count

import soundfile as sf
from tqdm import tqdm

from mbchl.utils import (
    MathDict,
    dump_json,
    estimate_bandwidth,
    find_files,
    nextpow2,
    read_json,
    split_list,
)


def _get_file_stats(fileobj, filename, durations, bws, rmss, fss, exts, channels):
    try:
        info = sf.info(fileobj)
    except sf.LibsndfileError:
        return
    durations.append(info.duration)
    if info.samplerate not in fss:
        fss[info.samplerate] = 0
    fss[info.samplerate] += 1
    ext = os.path.splitext(filename)[1]
    if ext not in exts:
        exts[ext] = 0
    exts[ext] += 1
    if info.channels not in channels:
        channels[info.channels] = 0
    channels[info.channels] += 1
    fileobj.seek(0)
    try:
        x, fs = sf.read(fileobj, always_2d=True)
    except sf.LibsndfileError:
        return
    n_fft = nextpow2(512 * fs / 16000)
    bw = estimate_bandwidth(x.T, n_fft=n_fft, fs=fs)
    bws.append(bw)
    rms = (x**2).mean() ** 0.5
    rmss.append(rms)


def _get_archive_stats(archive):
    durations = []
    bws = []
    rmss = []
    fss = {}
    exts = {}
    channels = {}
    with tarfile.open(archive) as tar:
        for tarinfo in tqdm(tar, total=len(tar.getmembers()), desc=archive):
            fileobj = tar.extractfile(tarinfo)
            _get_file_stats(
                fileobj, tarinfo.name, durations, bws, rmss, fss, exts, channels
            )
    return {
        "durations": durations,
        "bws": bws,
        "rmss": rmss,
        "fss": MathDict(fss),
        "exts": MathDict(exts),
        "channels": MathDict(channels),
    }


def _get_file_list_stats(files):
    durations = []
    bws = []
    rmss = []
    fss = {}
    exts = {}
    channels = {}
    for file in tqdm(files):
        with open(file, "rb") as fileobj:
            _get_file_stats(fileobj, file, durations, bws, rmss, fss, exts, channels)
    return {
        "durations": durations,
        "bws": bws,
        "rmss": rmss,
        "fss": MathDict(fss),
        "exts": MathDict(exts),
        "channels": MathDict(channels),
    }


def _mean_and_quantiles(sorted_numbers):
    n = len(sorted_numbers)
    return {
        "mean": sum(sorted_numbers) / n,
        "min": min(sorted_numbers),
        "max": max(sorted_numbers),
        "median": sorted_numbers[n // 2],
        "q1": sorted_numbers[n // 4],
        "q3": sorted_numbers[3 * n // 4],
    }


def _fmt_dict(d):
    out = {}
    for k, v in d.items():
        if isinstance(v, float):
            out[k] = round(v, 2)
        elif isinstance(v, dict):
            out[k] = _fmt_dict(v)
        else:
            out[k] = v
    return out


def main(input_, pool=None):
    if ("{}" in input_ or re.match(r".*\{:0\d+d\}.*", input_)) and input_.endswith(
        ".tar"
    ):
        files = [input_.format(i) for i in range(args.n_splits)]
        func = _get_archive_stats

    elif os.path.isdir(input_):
        files = split_list(list(find_files(input_)), args.n_splits)
        func = _get_file_list_stats
    else:
        raise ValueError("Input must have '{}' and end with .tar or be a directory")

    if pool is None:
        results = [func(files) for files in files]
    else:
        results = pool.map(func, files)

    durations = sum([r["durations"] for r in results], [])
    durations = sorted(durations)

    bws = sum([r["bws"] for r in results], [])
    bws = sorted(bws)

    rmss = sum([r["rmss"] for r in results], [])
    rmss = sorted(rmss)

    total = {
        "files": len(durations),
        "duration": _mean_and_quantiles(durations),
        "bw": _mean_and_quantiles(bws),
        "rms": _mean_and_quantiles(rmss),
        "fs": sum(r["fss"] for r in results),
        "ext": sum(r["exts"] for r in results),
        "channels": sum(r["channels"] for r in results),
    }

    total = _fmt_dict(total)

    return total


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input", nargs="+")
    parser.add_argument("-n", "--n_splits", type=int, required=True)
    parser.add_argument("-o", "--output", required=True)
    parser.add_argument("-f", "--force", action="store_true")
    parser.add_argument("--no_multiproc", action="store_true")
    args = parser.parse_args()

    if args.no_multiproc:
        pool = None
    else:
        n_cpu = min(args.n_splits, cpu_count())
        print(f"Using {n_cpu} CPUs")
        pool = Pool(n_cpu)

    try:
        if args.output is not None and os.path.exists(args.output):
            output = read_json(args.output)
        else:
            output = {}

        for input_ in args.input:
            if input_ in output and not args.force:
                print(f"{input_} already in output, skipping")
                continue
            stats = main(input_, pool=pool)
            if input_ in output:
                output[input_].update(stats)
            else:
                output[input_] = stats
            if args.output is not None:
                dump_json(output, args.output)

    finally:
        if pool is not None:
            pool.close()
            pool.join()

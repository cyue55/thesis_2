import argparse
import math
import os
import random
import re
import subprocess
import sys
import tarfile
import tempfile
from multiprocessing import Pool, cpu_count

import soundfile as sf
import soxr
from tqdm import tqdm

from mbchl.utils import find_files, split_list

SUPPORTED_OPUS_SAMPLING_RATES = {8000, 12000, 16000, 24000, 48000}


def str_to_bytes(s):
    d = {"K": 1024, "M": 1024**2, "G": 1024**3}
    assert s[:-1].isdigit(), "Size be of the form <number><unit>"
    assert s[-1] in d, f"Size must end with one of {list(d.keys())}"
    return int(s[:-1]) * d[s[-1]]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("indir", help="Input directory.")
    parser.add_argument("n_splits", type=int, help="Number of splits.")
    parser.add_argument("-o", "--output", help="Output tarfile formatting string.")
    parser.add_argument(
        "-e",
        "--ext",
        default=".wav",
        help="Extension of files to consider in input directory. Default is '.wav'.",
    )
    parser.add_argument(
        "-S",
        "--shuffle",
        action="store_true",
        help="Whether to shuffle input files. If toggled, files are renamed.",
    )
    parser.add_argument(
        "-c",
        "--check",
        action="store_true",
        help="Whether to check if input files are corrupted.",
    )
    parser.add_argument(
        "-r", "--regex", help="Regular expression to filter files in input directory."
    )
    parser.add_argument("-C", "--convert", help="Output audio format.")
    parser.add_argument(
        "-f",
        "--ffmpeg",
        nargs="?",
        const="ffmpeg",
        help="FFmpeg command. If not provided, libsndfile is used (recommended).",
    )
    parser.add_argument(
        "-y", "--skip", action="store_true", help="Skip corrupted files."
    )
    parser.add_argument(
        "-R",
        "--resample",
        type=int,
        help="Output sampling frequency. If not provided, files are not resampled.",
    )
    parser.add_argument(
        "-s",
        "--segment",
        type=float,
        help="Segment duration. If not provided, files are not segmented.",
    )
    parser.add_argument(
        "-M",
        "--mono",
        action="store_true",
        help="Whether to convert multi-channel files to mono.",
    )
    parser.add_argument(
        "-d",
        "--min_duration",
        type=float,
        help=(
            "Minimum output file duration in seconds. Useful with --segment to skip "
            "short trailing segments."
        ),
    )
    parser.add_argument("-l", "--min_rms", type=float, help="Minimum output file RMS.")
    parser.add_argument(
        "--opus_resample",
        action="store_true",
        help=(
            "Whether to resample to the next supported Opus sampling rate. "
            "Ignored unless '--convert opus' is used."
        ),
    )
    args = parser.parse_args()

    ext = None if args.ext is None else args.ext.lstrip(".")
    convert = None if args.convert is None else args.convert.lstrip(".")

    if not os.path.isdir(args.indir):
        raise FileNotFoundError(f"{args.indir} is not a directory")
    indir = os.path.normpath(args.indir)

    if convert is not None:
        if convert.casefold() == ext.casefold():
            raise ValueError("Input and output extensions are the same")
        if (
            convert.casefold() != "opus"
            and convert.upper() not in sf.available_formats()
        ):
            raise ValueError(f"Output format {convert} is not supported")

    if args.output is not None:
        if "{}" not in args.output and not re.match(r".*\{:0\d+d\}.*", args.output):
            raise ValueError("Output path must contain a '{}' or '{:0xd}' placeholder")
        assert args.output.endswith(".tar"), "Output path does not end with .tar"
        dirname = os.path.dirname(args.output)
        if dirname and not os.path.exists(dirname):
            raise FileNotFoundError(f"Output directory {dirname} does not exist")

    if args.ffmpeg is not None:
        if args.segment is not None:
            raise ValueError("Cannot use --segment with --ffmpeg")
        if args.min_duration is not None:
            raise ValueError("Cannot use --min_duration with --ffmpeg")
        if args.min_rms is not None:
            raise ValueError("Cannot use --min_rms with --ffmpeg")

    if (
        args.segment is not None
        and args.min_duration is not None
        and args.segment < args.min_duration
    ):
        raise ValueError("Segment duration cannot be less than min duration")

    print(f"Scanning {indir}")
    all_files = list(find_files(indir, ext=ext, regex=args.regex))
    if not all_files:
        raise ValueError(f"Found no files with ext .{ext} matching {args.regex}")
    if len(all_files) < args.n_splits:
        raise ValueError("Found less files than the requested number of splits")

    def define_segments(task):
        i_proc, files = task
        segments = []
        for file in files:
            try:
                info = sf.info(file)
            except sf.LibsndfileError:
                if args.skip:
                    print(f"Could not open {file}. Skipping.")
                    continue
                else:
                    raise
            frames = round(args.segment * info.samplerate)
            segments += [
                (file, frames, i * frames)
                for i in range(math.ceil(info.frames / frames))
            ]
        return segments

    if args.segment is None:
        all_segments = [(file, -1, 0) for file in all_files]
    else:
        print(f"Defining segments with duration {args.segment}")
        n_cpu = min(cpu_count(), len(all_files))
        tasks = list(enumerate(split_list(all_files, n_cpu)))
        with Pool(n_cpu) as pool:
            all_segments = sum(pool.map(define_segments, tasks), [])

    if args.shuffle:
        random.shuffle(all_segments)
    if args.shuffle or convert is not None or args.resample is not None or args.mono:
        # infer the number of digits required to represent all segments
        file_digits = int(math.log10(len(all_segments))) + 1

    splits = split_list(list(enumerate(all_segments)), args.n_splits)

    if args.output is None or "{}" in args.output:
        # infer the number of digits required to represent all splits
        tar_digits = int(math.log10(len(splits))) + 1
        if args.output is None:
            output = f"{indir}-{{:0{tar_digits}d}}.tar"
        else:
            output = args.output.format(f"{{:0{tar_digits}d}}")
    else:
        output = args.output
    print(f"Outputting to {output}")

    n_splits = len(splits)
    n_cpu = min(cpu_count(), n_splits)
    tasks = list(enumerate(split_list(list(enumerate(splits)), n_cpu)))

    print(f"Taring {len(all_segments)} segments using {n_cpu} CPUs")

    def process_segments(task):
        i_proc, splits = task
        out_paths = []
        for i_split, (i_tar, split) in enumerate(splits):
            out_path = output.format(i_tar)
            out_paths.append(out_path)
            desc = f"{i_split}/{len(splits)}"
            with tarfile.open(out_path, "w") as tar:
                for i_segment, (file, frames, start) in tqdm(
                    split, position=i_proc, file=sys.stdout, desc=desc
                ):
                    if args.shuffle or args.segment is not None:
                        arcname = f"{i_segment:0{file_digits}d}." + ext
                    else:
                        arcname = os.path.relpath(file, indir)
                    if (
                        args.check
                        or args.ffmpeg is None
                        and (
                            convert is not None
                            or args.resample is not None
                            or args.mono
                            or args.min_duration is not None
                            or args.min_rms is not None
                        )
                    ):
                        try:
                            x, fs = sf.read(file, frames, start)
                        except sf.LibsndfileError:
                            if args.skip:
                                print(f"Could not open {file}. Skipping.")
                                continue
                            else:
                                raise
                        if (
                            args.min_duration is not None
                            and len(x) / fs < args.min_duration
                        ):
                            continue
                    if (
                        convert is None
                        and args.resample is None
                        and not args.mono
                        and args.min_rms is None
                    ):
                        tar.add(file, arcname=arcname)
                        continue
                    out_ext = ext if convert is None else convert
                    tempdir = tempfile.gettempdir()
                    tempname = os.path.join(
                        tempdir, f"{i_segment:0{file_digits}d}." + out_ext
                    )
                    arcname = os.path.splitext(arcname)[0] + "." + out_ext
                    try:
                        if args.ffmpeg is None:
                            if args.resample is not None:
                                x = soxr.resample(x, fs, args.resample)
                                fs = args.resample
                            elif (
                                convert.casefold() == "opus"
                                and args.opus_resample
                                and fs not in SUPPORTED_OPUS_SAMPLING_RATES
                            ):
                                # find smallest supported frequency above fs
                                new_fs = min(
                                    opus_fs
                                    for opus_fs in SUPPORTED_OPUS_SAMPLING_RATES
                                    if opus_fs > fs
                                )
                                x = soxr.resample(x, fs, new_fs)
                                fs = new_fs
                            if args.mono and x.ndim == 2:
                                x = x.mean(axis=1)
                            if args.min_rms is not None:
                                rms = (x**2).mean() ** 0.5
                                if rms <= args.min_rms:
                                    continue
                            sf.write(
                                tempname,
                                x,
                                fs,
                                format="ogg"
                                if convert.casefold() == "opus"
                                else convert,
                                subtype="opus"
                                if convert.casefold() == "opus"
                                else None,
                            )
                        else:
                            ffcommand = args.ffmpeg.split()
                            ffcommand, ffargs = ffcommand[0], ffcommand[1:]
                            if args.resample is not None:
                                ffargs.extend(["-ar", f"{args.resample}"])
                            if args.mono:
                                ffargs.extend(["-ac", "1"])
                            subprocess.run(
                                [
                                    ffcommand,
                                    "-i",
                                    file,
                                    tempname,
                                    "-hide_banner",
                                    "-loglevel",
                                    "error",
                                    "-y",
                                    "-nostdin",
                                    *ffargs,
                                ],
                                check=True,
                            )
                        tar.add(tempname, arcname=arcname)
                    finally:
                        if os.path.exists(tempname):
                            os.remove(tempname)
        return out_paths

    with Pool(n_cpu) as pool:
        results = pool.map(process_segments, tasks)

    # check that the output tarfiles contain the right number of segments
    print("Checking output integrity")

    def count_segments(out_path):
        with tarfile.open(out_path, "r") as tar:
            return len(tar.getnames())

    out_paths = [out_path for out_paths in results for out_path in out_paths]
    with Pool(min(cpu_count(), len(out_paths))) as pool:
        n_segments = sum(pool.map(count_segments, out_paths))

    n_miss = len(all_segments) - n_segments
    if n_miss > 0:
        print(f"{n_miss} segments were not included in output tarfiles")

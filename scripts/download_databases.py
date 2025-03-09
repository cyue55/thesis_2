import argparse
import logging
import os
import zipfile

import requests
from tqdm import tqdm

from mbchl.utils import Registry

COMMANDS = Registry("commands")


def download_file(url, output_dir, check_exists=[]):
    response = requests.get(url, stream=True)
    response.raise_for_status()
    filename = url.split("/")[-1]
    outpath = os.path.join(output_dir, filename)
    if os.path.exists(outpath):
        logging.info(f"{outpath} already exists, skipping download")
    else:
        if not os.path.exists(args.output_path):
            os.makedirs(args.output_path)
        try:
            with tqdm.wrapattr(
                open(outpath, "wb"),
                "write",
                miniters=1,
                desc=filename,
                total=int(response.headers.get("content-length", 0)),
            ) as fout:
                for chunk in response.iter_content(chunk_size=4096):
                    fout.write(chunk)
            fout.close()
        except (Exception, KeyboardInterrupt):
            if os.path.exists(outpath):
                os.remove(outpath)
            raise
        return outpath


def unzip_file(zip_path, output_dir):
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(output_dir)
    os.remove(zip_path)


@COMMANDS.register("vctk")
def download_vctk():
    url = "https://datashare.ed.ac.uk/download/DS_10283_3443.zip"
    outpath = download_file(url, args.output_path)
    unzip_file(outpath, args.output_path)
    subzipfile = os.path.join(args.output_path, "VCTK-Corpus-0.92.zip")
    unzip_file(subzipfile, args.output_path)


@COMMANDS.register("demand")
def download_demand():
    urls = [
        "https://zenodo.org/records/1227121/files/DKITCHEN_16k.zip",
        "https://zenodo.org/records/1227121/files/DKITCHEN_48k.zip",
        "https://zenodo.org/records/1227121/files/DLIVING_16k.zip",
        "https://zenodo.org/records/1227121/files/DLIVING_48k.zip",
        "https://zenodo.org/records/1227121/files/DWASHING_16k.zip",
        "https://zenodo.org/records/1227121/files/DWASHING_48k.zip",
        "https://zenodo.org/records/1227121/files/NFIELD_16k.zip",
        "https://zenodo.org/records/1227121/files/NFIELD_48k.zip",
        "https://zenodo.org/records/1227121/files/NPARK_16k.zip",
        "https://zenodo.org/records/1227121/files/NPARK_48k.zip",
        "https://zenodo.org/records/1227121/files/NRIVER_16k.zip",
        "https://zenodo.org/records/1227121/files/NRIVER_48k.zip",
        "https://zenodo.org/records/1227121/files/OHALLWAY_16k.zip",
        "https://zenodo.org/records/1227121/files/OHALLWAY_48k.zip",
        "https://zenodo.org/records/1227121/files/OMEETING_16k.zip",
        "https://zenodo.org/records/1227121/files/OMEETING_48k.zip",
        "https://zenodo.org/records/1227121/files/OOFFICE_16k.zip",
        "https://zenodo.org/records/1227121/files/OOFFICE_48k.zip",
        "https://zenodo.org/records/1227121/files/PCAFETER_16k.zip",
        "https://zenodo.org/records/1227121/files/PCAFETER_48k.zip",
        "https://zenodo.org/records/1227121/files/PRESTO_16k.zip",
        "https://zenodo.org/records/1227121/files/PRESTO_48k.zip",
        "https://zenodo.org/records/1227121/files/PSTATION_16k.zip",
        "https://zenodo.org/records/1227121/files/PSTATION_48k.zip",
        "https://zenodo.org/records/1227121/files/SCAFE_48k.zip",
        "https://zenodo.org/records/1227121/files/SPSQUARE_16k.zip",
        "https://zenodo.org/records/1227121/files/SPSQUARE_48k.zip",
        "https://zenodo.org/records/1227121/files/STRAFFIC_16k.zip",
        "https://zenodo.org/records/1227121/files/STRAFFIC_48k.zip",
        "https://zenodo.org/records/1227121/files/TBUS_16k.zip",
        "https://zenodo.org/records/1227121/files/TBUS_48k.zip",
        "https://zenodo.org/records/1227121/files/TCAR_16k.zip",
        "https://zenodo.org/records/1227121/files/TCAR_48k.zip",
        "https://zenodo.org/records/1227121/files/TMETRO_16k.zip",
        "https://zenodo.org/records/1227121/files/TMETRO_48k.zip",
    ]
    for url in urls:
        src = os.path.join(args.output_path, url.split("/")[-1][:-8])
        dst = os.path.join(args.output_path, url.split("/")[-1][:-4])
        if os.path.exists(dst):
            logging.info(f"{dst} already exists, skipping download")
            continue
        outpath = download_file(url, args.output_path)
        unzip_file(outpath, args.output_path)
        os.rename(src, dst)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument("which", choices=COMMANDS.keys())
    parser.add_argument("output_path")
    args = parser.parse_args()
    COMMANDS.get(args.which)()

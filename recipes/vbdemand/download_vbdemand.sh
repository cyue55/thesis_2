#!/bin/bash
set -e

WGET_PROGRESS_BAR_FLAG="--show-progress"
if [[ $(wget --version) == *"Wget2"* ]]; then
    WGET_PROGRESS_BAR_FLAG="--force-progress"
fi

wget ${WGET_PROGRESS_BAR_FLAG} -c "https://datashare.ed.ac.uk/download/DS_10283_2791.zip"
mkdir -p "data/vbdemand/"
7z e "DS_10283_2791.zip" -o"data/vbdemand/" -y
rm -f "DS_10283_2791.zip"
for subzipfile in data/vbdemand/*zip ; do
    mkdir -p "${subzipfile%.zip}"
    7z e "${subzipfile}" -o"${subzipfile%.zip}" -y
    rm -f "${subzipfile}"
done

#!/bin/bash
#BSUB -q hpc
#BSUB -J jobname
#BSUB -n 1
#BSUB -W 24:00
#BSUB -R "rusage[mem=4GB]"
#BSUB -R "span[hosts=1]"
#BSUB -oo lsf/logs/%J.out
#BSUB -eo lsf/logs/%J.err
source venv/bin/activate
python ARGS

#!/bin/bash
#BSUB -q gpuv100
#BSUB -J jobname
#BSUB -n 4
#BSUB -W 24:00
#BSUB -R "rusage[mem=4GB]"
#BSUB -R "span[hosts=1]"
#BSUB -oo lsf/logs/%J.out
#BSUB -eo lsf/logs/%J.err
#BSUB -gpu "num=1:mode=exclusive_process"
module load gcc/14.2.0-binutils-2.43
module load cuda/12.6.2
source .venv/bin/activate
python ARGS

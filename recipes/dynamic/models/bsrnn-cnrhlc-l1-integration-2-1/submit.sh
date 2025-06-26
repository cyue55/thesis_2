#!/bin/bash
#BSUB -q gpuv100
#BSUB -J jobname
#BSUB -n 8
#BSUB -W 15:00
#BSUB -R "rusage[mem=8GB]"
#BSUB -R "span[hosts=1]"
#BSUB -oo lsf/logs/%J.out
#BSUB -eo lsf/logs/%J.err
#BSUB -gpu "num=1:mode=exclusive_process"
module load gcc/14.2.0-binutils-2.43
module load cuda/12.6.2
source venv/bin/activate
python recipes/dynamic/evaluate.py recipes/dynamic/models/bsrnn-cnrhlc-l1-integration-2-1/checkpoints/last.ckpt data/fcnrhlc-test   --metrics pesq estoi snr hasqi haspi --write_wav --noisy --noisy_output_dir models/bsrnn-cnrhlc-l1-integration-2-1/audio_outputs --normalizes

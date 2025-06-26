# Controlled joint noise reduction and hearing loss compensation recipe

This recipe corresponds to the paper "Controlled joint noise reduction and hearing loss compensation using a differentiable auditory model".

## Training

Models are trained with noisy and reverberant scenes generated on-the-fly using `mbchl.data.datasets.DynamicAudioDataset`. To get access to the speech and noise files used in the paper, reach out to phigon@dtu.dk. Alternatively, update the model configuration files to point to your own speech and noise files.

To train a model:

```bash
python train.py models/<model_name>/
```

On the HPC cluster, first create a symbolic link to your virtual environment in the top level directory:

```bash
ln -s ../../.venv/ .venv
```

Then submit a training job:

```bash
./lsf/train.sh -n 8 -q gpua100 -W 36 models/<model_name>/
```

Notes:
- 8 CPUs are requested with `-n 8` because `workers: 8` is used in the model configuration files to speed up on-the-fly scene generation.
- The job is submitted to the `gpua100` queue to request a larger A100 GPU and prevent out-of-memory errors when using a V100 GPU.
- Training should take around 26 hours which exceeds the default 24-hour walltime, so a walltime of 36 hours is requested with `-W 36`.

## Evaluation

To get access to the evaluation dataset used in the paper, reach out to phigon@dtu.dk.

To calculate HASPI and HASQI scores, first install `pyclarity`:

```bash
pip install pyclarity
```

Evaluation can be done on CPU. To evaluate a trained model using multiple CPUs:

```bash
python evaluate.py \
    <dataset_path> \
    --ckpt models/<model_name>/checkpoints/<checkpoint_name>.ckpt \
    --n_cpu <n_cpu> \
    --metrics pesq estoi snr haspi hasqi
```

This creates a `scores.npz` file inside `models/<model_name>/` containing scores for each scene and for each standard audiogram. To load the scores:

```python
import numpy as np

data = np.load("models/<model_name>/scores.npz")
subdata = data["haspi.N1"]  # HASPI scores for N1 audiogram
```

Here `subdata` is an array with shape `(n_scenes, n_alpha)`, where `n_scenes` is the number of scenes (1000 in the paper) and `n_alpha` is 1 for all models except those trained for controllable joint noise reduction and hearing loss compensation, for which scores are calculated for different values of the control parameter $`\alpha`$ (11 values between 0.0 and 1.0 by default).

To evaluate on the HPC cluster, first edit `lsf/hpc.sh` to request the right number of CPUs. Then:

```bash
./lsf/submit.sh \
    lsf/hpc.sh \
    evaluate.py \
    <dataset_path> \
    --ckpt models/<model_name>/checkpoints/<checkpoint_name>.ckpt \
    --n_cpu <n_cpu> \
    --metrics pesq estoi snr haspi hasqi
```

_TODO: implement a clean solution to automatically request `n_cpu` CPUs instead of manually editing `lsf/hpc.sh`._

To calculate scores for the noisy signals, use `--noisy --output_dir models/noisy/`:

```bash
./lsf/submit.sh \
    lsf/hpc.sh \
    evaluate.py \
    <dataset_path> \
    --noisy \
    --output_dir models/noisy/ \
    --n_cpu <n_cpu> \
    --metrics pesq estoi snr haspi hasqi
```

The scores will be placed in `models/noisy/scores.npz`.

To calculate scores with NAL-R amplification after the model, use the `--nalr` option:

```bash
./lsf/submit.sh \
    lsf/hpc.sh \
    evaluate.py \
    <dataset_path> \
    --ckpt models/<model_name>/checkpoints/<checkpoint_name>.ckpt \
    --nalr \
    --n_cpu <n_cpu> \
    --metrics pesq estoi snr haspi hasqi
```

The scores will be placed in `models/<model_name>-nalr/scores.npz`.

To calculate scores for NAL-R amplification only, use the `--nalr` option:

```bash
./lsf/submit.sh \
    lsf/hpc.sh \
    evaluate.py \
    <dataset_path> \
    --noisy \
    --nalr \
    --output_dir models/noisy-nalr/ \
    --n_cpu <n_cpu> \
    --metrics pesq estoi snr haspi hasqi
```

The scores will be placed in `models/noisy-nalr/scores.npz`.

To write the output signals on disk for listening, use the `--write_wav` option. It is recommended to use it together with `--normalize` to peak normalize the files, and `--n_files <n_files>` to process only a few files.

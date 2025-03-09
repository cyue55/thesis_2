# Voicebank+DEMAND recipe

## Dataset

To download the dataset:

```bash
./download_vbdemand.sh
```

## Training

To train a model:

```bash
python train.py models/<model_name>/
```

Example:

```bash
python train.py models/convtasnet/
```

To train models on the HPC cluster, first create a symbolic link to your virtual environment:

```bash
ln -s <path/to/venv> venv
```

Example:
```bash
ln -s ../../venv/ venv
```

Then to submit a job:

```bash
./lsf/train.sh models/<model_name>/
```

Example:

```bash
./lsf/train.sh models/convtasnet/
```

## Evaluation

To evaluate a trained model:

```bash
python enhance.py models/<model_name>/checkpoints/<checkpoint_name>.ckpt <output_dir> --evaluate --no_wav --cuda
```

On the HPC cluster:
```bash
./lsf/enhance.sh models/<model_name>/checkpoints/<checkpoint_name>.ckpt -x "<output_dir> --evaluate --no_wav --cuda"
```

If there is a symbolic link in the checkpoint path, the script might not find the model configuration file. In this case, the configuration file can be specified with the `--cfg` option:

```bash
python enhance.py models/<model_name>/checkpoints/<checkpoint_name>.ckpt <output_dir> --evaluate --no_wav --cuda --cfg models/<model_name>/config.yaml
```

On the HPC cluster:
```bash
./lsf/enhance.sh models/<model_name>/checkpoints/<checkpoint_name>.ckpt -x "<output_dir> --evaluate --no_wav --cuda --cfg models/<model_name>/config.yaml"
```

## Results

| Model           | Params | Causal  | Checkpoint  | PESQ | ESTOI | SNR  | MAC/s  |
| --------------- | ------ | ------- | ----------- | ---- | ----- | ---- | ------ |
| Noisy           | -      | -       | -           | 1.97 |  0.74 |  8.4 | -      |
| FFNN            |  1.5 M | &check; | `last.ckpt` | 2.55 |  0.81 | 17.3 |  0.4 G |
| Conv-TasNet     |  4.9 M | &cross; | `last.ckpt` | 2.52 |  0.89 | 18.7 |  9.7 G |
| BSRNN           |  3.3 M | &cross; | `last.ckpt` | 2.64 |  0.90 | 18.3 |  1.3 G |
| TF-GridNet      |  3.8 M | &cross; | `last.ckpt` | 2.69 |  0.92 | 19.4 |  9.7 G |
| SGMSE+M         | 27.8 M | &cross; | `last.ckpt` | 2.67 |  0.86 | 17.5 | 20.0 T |

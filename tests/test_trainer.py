import tempfile

import pytest
import torch

from mbchl.has import HARegistry
from mbchl.training.trainer import AudioTrainer

FS = 16000
BATCH_SIZE = 4
BATCH_SAMPLER = "bucket"
DYNAMIC = True
TRAIN_EXAMPLES = 16
VAL_EXAMPLES = 4
MIN_LENGTH = 8000  # 0.5 s
MAX_LENGTH = 32000  # 2.0 s
EPOCHS = 2


@pytest.mark.parametrize(
    "model, model_kwargs, channels",
    [
        [
            "dummy",
            {
                "input_channels": 2,
                "output_channels": 2,
            },
            [2, 2],
        ],
        [
            "bsrnn",
            {
                "input_channels": 2,
                "reference_channels": [0, 1],
                "layers": 1,
                "base_channels": 1,
            },
            [2, 2],
        ],
        [
            "convtasnet",
            {
                "input_channels": 2,
                "reference_channels": [0, 1],
                "filters": 1,
                "bottleneck_channels": 1,
                "hidden_channels": 1,
                "skip_channels": 1,
                "layers": 1,
                "repeats": 1,
            },
            [2, 2],
        ],
        [
            "ffnn",
            {"input_channels": 2, "reference_channels": [0, 1], "hidden_sizes": [1, 1]},
            [2, 2],
        ],
        [
            "ineube",
            {
                "net1_cls": "bsrnn",
                "net1_kw": {
                    "input_channels": 2,
                    "reference_channels": [0, 1],
                    "layers": 1,
                    "base_channels": 1,
                },
                "net2_cls": "bsrnn",
                "net2_kw": {
                    "input_channels": 6,
                    "reference_channels": [0, 1],
                    "layers": 1,
                    "base_channels": 1,
                },
            },
            [2, 2],
        ],
        [
            "sgmsep",
            {
                "reference_channels": [0, 1],
                "net_kw": {
                    "in_channels": 8,
                    "out_channels": 4,
                    "aux_out_channels": 8,
                    "base_channels": 4,
                    "channel_mult": [1, 1, 1, 1],
                    "num_blocks_per_res": 1,
                    "noise_channel_mult": 1,
                    "emb_channel_mult": 1,
                    "fir_kernel": [1, 1],
                    "attn_resolutions": [],
                    "attn_bottleneck": False,
                },
                "solver_kw": {
                    "num_steps": 1,
                },
            },
            [2, 2],
        ],
        [
            "tcndenseunet",
            {
                "input_channels": 2,
                "output_channels": 2,
                "hidden_channels": 1,
                "hidden_channels_dense": 1,
                "tcn_repeats": 1,
                "tcn_blocks": 1,
                "tcn_channels": 1,
                "stft_kw": {"frame_length": 32, "hop_length": 16},
            },
            [2, 2],
        ],
        [
            "tfgridnet",
            {
                "input_channels": 2,
                "output_channels": 2,
                "layers": 1,
                "lstm_hidden_units": 1,
                "attn_heads": 1,
                "attn_approx_qk_dim": 1,
                "_emb_dim": 1,
                "_emb_ks": 1,
                "_emb_hs": 1,
            },
            [2, 2],
        ],
    ],
    ids=[
        "dummy",
        "bsrnn",
        "convtasnet",
        "ffnn",
        "ineube",
        "sgmsep",
        "tcndenseunet",
        "tfgridnet",
    ],
)
def test_trainer(dummy_model, dummy_dataset, model, model_kwargs, channels):
    if model == "dummy":
        model = dummy_model(**model_kwargs)
    else:
        model = HARegistry.init(model, seed=0, **model_kwargs)

    train_dataset = dummy_dataset(
        examples=TRAIN_EXAMPLES,
        channels=channels,
        min_length=MIN_LENGTH,
        max_length=MAX_LENGTH,
        fs=FS,
        transform=model.transform,
    )

    val_dataset = dummy_dataset(
        examples=VAL_EXAMPLES,
        channels=channels,
        min_length=MIN_LENGTH,
        max_length=MAX_LENGTH,
        fs=FS,
        transform=None,
    )

    parameter_before_training = sample_parameters(model)

    with tempfile.TemporaryDirectory() as tempdir:
        trainer = AudioTrainer(
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            model_dirpath=tempdir,
            train_batch_sampler_kw={
                "batch_size": BATCH_SIZE,
                "dynamic": DYNAMIC,
                "fs": FS,
            },
            val_batch_sampler=BATCH_SAMPLER,
            val_batch_sampler_kw={
                "batch_size": BATCH_SIZE,
                "dynamic": DYNAMIC,
                "fs": FS,
            },
            val_period=1,
            val_metrics={"snr": None},
            model=model,
            epochs=EPOCHS,
            device="cpu",
            preload=True,
            ema="classic",
            ema_kw={"beta": 0.99},
        )
        trainer.run()

        # check that parameters have changed after training
        parameter_after_training = sample_parameters(model)
        assert not torch.allclose(
            parameter_before_training,
            parameter_after_training,
        )

        # test resume from checkpoint device='cuda' if available
        trainer = AudioTrainer(
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            model_dirpath=tempdir,
            train_batch_sampler=BATCH_SAMPLER,
            train_batch_sampler_kw={
                "batch_size": BATCH_SIZE,
                "dynamic": DYNAMIC,
                "fs": FS,
            },
            val_batch_sampler=BATCH_SAMPLER,
            val_batch_sampler_kw={
                "batch_size": BATCH_SIZE,
                "dynamic": DYNAMIC,
                "fs": FS,
            },
            val_period=1,
            val_metrics={"snr": None},
            model=model.cuda() if torch.cuda.is_available() else model,
            epochs=EPOCHS + 1,
            device="cuda" if torch.cuda.is_available() else "cpu",
            preload=False,  # test no preloading this time
            ema="classic",
            ema_kw={"beta": 0.99},
        )
        trainer.run()

        # check that parameters have changed again
        parameter_after_training_again = sample_parameters(model)
        assert not torch.allclose(
            parameter_after_training,
            parameter_after_training_again,
        )


def test_training_with_auditory_loss(dummy_model, dummy_dataset):
    model = "ffnn"
    model_kwargs = {
        "input_channels": 2,
        "reference_channels": [0, 1],
        "hidden_sizes": [1, 1],
        "loss": "auditory",
    }
    channels = [2, 2]
    test_trainer(dummy_model, dummy_dataset, model, model_kwargs, channels)


def sample_parameters(net, n=10):
    parameters = []
    numel = 0
    for next_parameters in net.parameters():
        if not next_parameters.requires_grad:
            continue
        for param in next_parameters.flatten():
            parameters.append(param.item())
            numel += 1
            if numel == n:
                break
        if numel == n:
            break
    return torch.tensor(parameters)

import argparse
import logging
import os
import pprint

import torch
import torch.distributed as dist
import wandb
from dotenv import load_dotenv

from mbchl.data.datasets import DatasetRegistry
from mbchl.has import HARegistry
from mbchl.logging import set_logger
from mbchl.training.trainer import AudioTrainer
from mbchl.utils import parse_args, read_yaml, recursive_dict_update, seed_everything


def main():
    # load environment variables
    load_dotenv()

    # load config
    cfg_path = os.path.join(args.input, "config.yaml")
    cfg = read_yaml(cfg_path)

    # update cfg with command line args
    recursive_dict_update(cfg, parse_args(args.args))

    # seed for reproducibility
    seed_everything(cfg["global_seed"])

    # init distributed group
    rank = cfg["trainer"].pop("rank")
    device = cfg["trainer"].pop("device")
    if cfg["trainer"]["ddp"]:
        dist.init_process_group("nccl")
        rank = dist.get_rank()
        device = rank % torch.cuda.device_count()

    # initialize logger
    log_file = os.path.join(args.input, "log.log")
    set_logger(log_file, cfg["trainer"]["ddp"], rank, args.debug)
    if rank == 0:
        logging.info(f"Training {args.input}")
        logging.info(f"Configuration: \n {pprint.pformat(cfg)}")

    # initialize wandb
    ####
    cfg["trainer"]["use_wandb"] = False
    ####
    if cfg["trainer"]["use_wandb"] and rank == 0:
        missing_vars = [
            var for var in ["WANDB_ENTITY", "WANDB_PROJECT"] if var not in os.environ
        ]
        if missing_vars:
            logging.warning(
                f'{" and ".join(missing_vars)} environment '
                f'variable{"s" if len(missing_vars) > 1 else ""} not set. '
                'Using wandb defaults.'
            )
        configured = wandb.login(timeout=0)
        if not configured:
            raise ValueError(
                "Could not login to wandb. This can be fixed by executing "
                "`wandb login` or by setting the WANDB_API_KEY environment variable in "
                "a .env file in the working directory"
            )
        wandb.init(
            config=cfg,
            name=os.path.basename(os.path.normpath(args.input)),
            dir=args.input,
            id=args.wandb_run_id,
            resume=args.wandb_run_id is not None,
        )

    # initialize model
    logging.debug("Initializing HA")
    model = HARegistry.init(cfg["ha"], **cfg["ha_kw"])

    # initialize datasets
    logging.debug("Initializing datasets")
    train_dataset = DatasetRegistry.get(cfg["dataset"]["train"])(
        transform=model.transform,
        **cfg["dataset"]["train_kw"],
    )
    val_dataset = DatasetRegistry.get(cfg["dataset"]["val"])(
        transform=None,
        **cfg["dataset"]["val_kw"],
    )

    # initialize trainer
    logging.debug("Initializing trainer")
    trainer = AudioTrainer(
        model=model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        model_dirpath=args.input,
        device=device,
        rank=rank,
        ignore_checkpoint=cfg["trainer"].pop("ignore_checkpoint") or args.force,
        **cfg["trainer"],
    )

    # run
    logging.debug("Starting trainer")
    trainer.run()


if __name__ == "__main__":
    custom_args = ["/Users/yue/Documents/Thesis/Codes/thesis/recipes/cnrhlc/models/bsrnn-cnrhlc-l1-test"]
    parser = argparse.ArgumentParser()
    parser.add_argument("input", help="model directory")
    parser.add_argument("args", nargs="*")
    parser.add_argument("-f", "--force", action="store_true", help="train from scratch")
    parser.add_argument("--wandb_run_id", help="id of wandb run to resume")
    parser.add_argument("--debug", help="set log level to DEBUG", action="store_true")
    args = parser.parse_args(custom_args)
    main()

from data import get_train_loader, get_test_loader
from solver_lit import LitBrainMRI
from torch import nn
from munch import Munch
import pandas as pd
import numpy as np
import configparser
from argparser import create_parser
import torch
import os
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint


def main(args):
    # Setting Random Seeds
    np.random.seed(999)
    torch.manual_seed(999)
    torch.cuda.manual_seed(999)

    # Configure GPUs
    torch.backends.cudnn.benchmark = True
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    print("\nTorch GPUs: " + str(torch.cuda.device_count()))


    csv = pd.read_csv(args.csv_path)
    root = args.img_dir
    csv["Domain"] = 0
    wandb_logger = WandbLogger(project=args.experiment, name=args.experiment_tag)
    checkpoint_callback = ModelCheckpoint(
        every_n_train_steps=args.save_every,
    )

    loaders = Munch(
        src=get_train_loader(
            csv,
            root=root,
            which="source",
            img_size=args.img_size,
            batch_size=args.batch_size,
            prob=0,
            num_workers=args.num_workers,
        ),
        ref=get_train_loader(
            csv,
            root=root,
            which="reference",
            img_size=args.img_size,
            batch_size=args.batch_size,
            prob=0,
            num_workers=args.num_workers,
        ),
        val=get_test_loader(
            csv,
            root=root,
            img_size=args.img_size,
            batch_size=args.val_batch_size,
            shuffle=True,
            num_workers=args.num_workers,
        ),
    )

    model = LitBrainMRI(args, loaders)
    print("Trainer")
    trainer = pl.Trainer(
        fast_dev_run=False,
        accelerator=args.device,
        devices=1,
        auto_select_gpus=False,
        max_steps=args.max_steps,
        benchmark=True,
        log_every_n_steps=args.log_every,
        logger=[wandb_logger],
        callbacks=[checkpoint_callback],
    )
    trainer.fit(model=model)


if __name__ == "__main__":
    argparser = create_parser()
    args = argparser.parse_args()

    main(args)

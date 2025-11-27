import argparse
import os
import yaml
import types

import pytorch_lightning as pl
from torchvision import transforms

from ..modules.cyclegan_segment_module import Module
from ..datamodules.datamodule import DataModule


if __name__ == "__main__":

    with open(r"configs\train01.yaml", "r") as f:
        cfg = yaml.safe_load(f)

    checkpoint_path = r"C:\Users\waiho\Coding Projects\Unsupervised-Domain-Adaption\lightning_logs\version_50\checkpoints\joint-epoch=09-val_total_loss=6.0189.ckpt"
    # Load model from checkpoint (this will restore weights + hyperparameters passed in save)
    model = Module.load_from_checkpoint(checkpoint_path, cfg=cfg)

    transform = transforms.Compose(
        [
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
        ]
    )
    # Instantiate the project's DataModule and prepare the test dataloader
    datamodule = DataModule(
        drrs_dir=cfg["data"]["drrs"],
        xrays_dir=cfg["data"]["xrays"],
        masks_dir=cfg["data"]["masks"],
        transform=transform,
        batch_size=cfg["data"]["batch_size"],
        num_workers=cfg["data"]["num_workers"],
    )

    # Run the DataModule setup to prepare datasets
    datamodule.setup()

    # If the Module doesn't implement `test_step`, reuse `validation_step` by binding it

    trainer = pl.Trainer(
        accelerator=cfg["trainer"]["accelerator"],
        devices=cfg["trainer"]["devices"],
        precision=cfg["trainer"]["precision"],
        log_every_n_steps=cfg["trainer"]["log_every_n_steps"],
    )

    # Run test. We already loaded weights into `model`, so pass `ckpt_path=None`.
    results = trainer.test(model, datamodule=datamodule, ckpt_path=None)
    print("Test results:", results)

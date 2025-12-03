from torchvision import transforms
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from ..datamodules.datamodule import DataModule
from ..modules.cyclegan_segment_module import Module
import yaml

if __name__ == "__main__":
    # ----------------------------
    # 1. Data
    # ----------------------------

    with open("configs/train_welsh.yaml", "r") as f:
        cfg = yaml.safe_load(f)
    print(cfg)

    transform = transforms.Compose(
        [
            transforms.Resize((512, 512)),
            transforms.ToTensor(),
        ]
    )

    datamodule = DataModule(
        drrs_dir=cfg["data"]["drrs"],
        xrays_dir=cfg["data"]["xrays"],
        masks_dir=cfg["data"]["masks"],
        transform=transform,
        batch_size=cfg["data"]["batch_size"],
        num_workers=cfg["data"]["num_workers"],
    )

    # ----------------------------
    # 2. Model
    # ----------------------------
    module = Module(cfg)

    # ----------------------------
    # 3. Callbacks
    # ----------------------------
    checkpoint_callback = ModelCheckpoint(
        monitor="val_total_loss",
        mode=cfg["checkpoint"]["mode"],
        save_top_k=cfg["checkpoint"]["save_top_k"],
        filename=cfg["checkpoint"]["filename"],
    )

    lr_monitor = LearningRateMonitor(logging_interval="step")

    # ----------------------------
    # 4. Trainer
    # ----------------------------
    trainer = pl.Trainer(
        accelerator=cfg["trainer"]["accelerator"],  # ensures GPU usage
        devices=cfg["trainer"]["devices"],  # number of GPUs
        max_epochs=cfg["trainer"]["max_epochs"],
        precision=cfg["trainer"]["precision"],  # optional mixed precision for speed
        callbacks=[checkpoint_callback, lr_monitor],
        log_every_n_steps=cfg["trainer"]["log_every_n_steps"],
        check_val_every_n_epoch=cfg["trainer"]["check_val_every_n_epoch"],
    )

    # ----------------------------
    # 5. Fit
    # ----------------------------
    trainer.fit(module, datamodule)

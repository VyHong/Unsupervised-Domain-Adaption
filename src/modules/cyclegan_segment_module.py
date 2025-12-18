import os
import pytorch_lightning as pl
import torch
import torch.nn as nn
from src.models.cyclegan.cyclegan import CycleGANModel
from src.models.hf_segmentation_wrapper import get_hf_model
import wandb
from src.utils.image_manipulation import (
    logits_to_rgb,
    resample_logits,
    save_visualization,
)
from ..utils.config_object import dict_to_simple_object


class Module(pl.LightningModule):
    def __init__(self, cfg):
        super().__init__()
        opt = dict_to_simple_object(cfg["cyclegan"])

        self.cyclegan = CycleGANModel(opt)
        self.segmentation, self.seg_extractor = get_hf_model(
            cfg["segmodel"]["name"], cfg["segmodel"]["num_labels"]
        )

        self.to("cuda")
        self.train()

        self.gan_lr = cfg["cyclegan"]["lr"]
        self.seg_lr = cfg["segmodel"]["lr"]

        self.cfg = cfg
        self.seg_loss = nn.CrossEntropyLoss()
        if opt.isTrain:
            self.run = wandb.init(
                entity="vy_hong", project="Guided Research", config=cfg
            )

    def forward(self, drrs, xrays):
        input_images = {"A": drrs, "B": xrays}
        self.cyclegan.set_input(input_images)
        self.cyclegan.forward()
        images = self.cyclegan.get_images()  # generate synthetic image
        fake_features = self.seg_extractor(images["fake_xray"], return_tensors="pt")
        for k in fake_features:
            fake_features[k] = fake_features[k].to(device="cuda", dtype=torch.float16)

        seg = self.segmentation(**fake_features)  # segment it

        output_tensors = {**images, **{"segmentation": seg}}
        return output_tensors

    def training_step(self, batch, batch_idx):
        gan_loss, seg_loss, total_loss, output_tensors = self.shared_step(
            batch, batch_idx
        )
        optimizer = self.optimizers()
        current_lr = optimizer.param_groups[0]["lr"]
        log_dict = {
            "train_gan_loss": gan_loss,
            "train_seg_loss": seg_loss,
            "train_total_loss": total_loss,
            "lr": current_lr,
        }
        self.run.log(log_dict)
        self.log("train_gan_loss", gan_loss)
        self.log("train_seg_loss", seg_loss)
        self.log("train_total_loss", total_loss)
        self.log("lr", current_lr)
        return total_loss

    def validation_step(self, batch, batch_idx):
        gan_loss, seg_loss, total_loss, output_tensors = self.shared_step(
            batch, batch_idx
        )
        optimizer = self.optimizers()
        current_lr = optimizer.param_groups[0]["lr"]
        log_dict = {
            "val_gan_loss": gan_loss,
            "val_seg_loss": seg_loss,
            "val_total_loss": total_loss,
            "lr": current_lr,
        }
        self.run.log(log_dict)
        self.log("val_gan_loss", gan_loss, prog_bar=True, on_epoch=True)
        self.log("val_seg_loss", seg_loss, prog_bar=True, on_epoch=True)
        self.log("val_total_loss", total_loss, prog_bar=True, on_epoch=True)
        self.log("lr", current_lr)

        if batch_idx == 0:
            drrs = batch["drrs"]
            xrays = batch["xrays"]
            masks = batch["masks"]
            seg = output_tensors["pred_masks"][0]
            drr_grid = [
                drrs[0],
                output_tensors["fake_xray"][0],
                output_tensors["rec_drr"][0],
                masks[0],
                logits_to_rgb(seg, self.segmentation.config.num_labels),
            ]
            os.makedirs(os.path.join(self.logger.log_dir, "images"), exist_ok=True)
            save_visualization(
                drr_grid,
                os.path.join(
                    self.logger.log_dir,
                    "images",
                    f"epoch={self.current_epoch:02d}_val_drr.png",
                ),
            )

            xray_grid = [
                xrays[0],
                output_tensors["fake_drr"][0],
                output_tensors["rec_xray"][0],
            ]
            save_visualization(
                xray_grid,
                os.path.join(
                    self.logger.log_dir,
                    "images",
                    f"epoch={self.current_epoch:02d}_val_xray.png",
                ),
            )

        return total_loss

    def test_step(self, batch, batch_idx):
        gan_loss, seg_loss, total_loss, output_tensors = self.shared_step(
            batch, batch_idx
        )
        optimizer = self.optimizers()
        current_lr = optimizer.param_groups[0]["lr"]
        log_dict = {
            "test_gan_loss": gan_loss,
            "test_seg_loss": seg_loss,
            "test_total_loss": total_loss,
            "lr": current_lr,
        }
        self.run.log(log_dict)
        self.log("test_gan_loss", gan_loss, prog_bar=True, on_epoch=True)
        self.log("test_seg_loss", seg_loss, prog_bar=True, on_epoch=True)
        self.log("test_total_loss", total_loss, prog_bar=True, on_epoch=True)
        self.log("lr", current_lr)

        return total_loss

    def shared_step(self, batch, batch_idx):
        drrs = batch["drrs"]
        xrays = batch["xrays"]
        masks = batch["masks"]

        output_tensors = self(drrs, xrays)

        gan_loss = self.cyclegan.loss()
        target_size = masks.shape[2:]
        upsampled_logits = resample_logits(
            output_tensors["segmentation"].logits, target_size
        )
        output_tensors["pred_masks"] = upsampled_logits
        seg_loss = self.seg_loss(upsampled_logits, masks.long().squeeze(1))

        total_loss = (
            self.cfg["loss_weights"]["gan"] * gan_loss
            + self.cfg["loss_weights"]["seg"] * seg_loss
        )
        return gan_loss, seg_loss, total_loss, output_tensors

    def configure_optimizers(self):
        gan_params = self.cyclegan.parameters()
        seg_params = self.segmentation.parameters()

        all_params = list(gan_params) + list(seg_params)
        optimizer = torch.optim.AdamW(all_params, lr=float(self.gan_lr))

        # Check if scheduler is enabled in config
        if self.cfg.get("scheduler", {}).get("enabled", False):
            scheduler_cfg = self.cfg["scheduler"]
            warmup_epochs = int(scheduler_cfg.get("warmup_epochs", 0))
            decay_epochs = int(scheduler_cfg.get("decay_epochs", 100))
            min_lr = float(scheduler_cfg.get("min_lr", 1e-6))

            # Learning rate scheduler: constant for warmup_epochs, then decay over decay_epochs
            scheduler1 = torch.optim.lr_scheduler.ConstantLR(
                optimizer, factor=1.0, total_iters=warmup_epochs
            )
            scheduler2 = torch.optim.lr_scheduler.PolynomialLR(
                optimizer, total_iters=decay_epochs, end_lr=min_lr
            )
            scheduler = torch.optim.lr_scheduler.SequentialLR(
                optimizer,
                schedulers=[scheduler1, scheduler2],
                milestones=[warmup_epochs],
            )

            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": "epoch",
                    "frequency": 1,
                },
            }

        # Return optimizer only if scheduler is disabled
        return optimizer

    def on_train_epoch_end(self):
        schedulers = self.lr_schedulers()
        if schedulers is not None:
            schedulers.step()

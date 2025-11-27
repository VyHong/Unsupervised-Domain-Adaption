import pytorch_lightning as pl
import torch
import torch.nn as nn
from src.models.cyclegan import CycleGANModel
from src.models.hf_segmentation_wrapper import get_hf_model
import wandb
from ..utils.config_object import dict_to_simple_object
import torch.nn.functional as F


class Module(pl.LightningModule):
    def __init__(self, cfg):
        super().__init__()
        opt = dict_to_simple_object(cfg["cyclegan"])

        self.cyclegan = CycleGANModel(opt)
        self.segmentation, self.seg_extractor = get_hf_model(cfg["segmodel"]["name"])

        self.to("cuda")
        self.train()

        self.gan_lr = cfg["cyclegan"]["lr"]
        self.seg_lr = cfg["segmodel"]["lr"]

        self.l1_loss = nn.L1Loss()
        self.seg_loss = nn.CrossEntropyLoss()
        if opt.isTrain:
            self.run = wandb.init(
                entity="vy_hong", project="Guided Research", config=cfg
            )

    def forward(self, drrs, xrays):
        input_images = {"A": drrs, "B": xrays}
        self.cyclegan.set_input(input_images)
        self.cyclegan.forward()
        fake = self.cyclegan.get_to_segment_data()  # generate synthetic image
        fake_features = self.seg_extractor(fake, return_tensors="pt")
        for k in fake_features:
            fake_features[k] = fake_features[k].to(device="cuda", dtype=torch.float16)

        seg = self.segmentation(**fake_features)  # segment it
        return fake, seg

    def training_step(self, batch, batch_idx):
        gan_loss, seg_loss, total_loss, fake_img = self.shared_step(batch, batch_idx)
        log_dict = {
            "train_gan_loss": gan_loss,
            "train_seg_loss": seg_loss,
            "train_total_loss": total_loss,
        }
        self.run.log(log_dict)
        self.log("train_gan_loss", gan_loss)
        self.log("train_seg_loss", seg_loss)
        self.log("train_total_loss", total_loss)
        return total_loss

    def validation_step(self, batch, batch_idx):
        gan_loss, seg_loss, total_loss, fake_img = self.shared_step(batch, batch_idx)
        log_dict = {
            "val_gan_loss": gan_loss,
            "val_seg_loss": seg_loss,
            "val_total_loss": total_loss,
        }
        self.run.log(log_dict)
        self.log("val_gan_loss", gan_loss, prog_bar=True, on_epoch=True)
        self.log("val_seg_loss", seg_loss, prog_bar=True, on_epoch=True)
        self.log("val_total_loss", total_loss, prog_bar=True, on_epoch=True)

        drrs = batch["drrs"]
        if batch_idx == 100:
            grid = torch.cat([drrs[:4], fake_img[:4]], dim=-1)
            self.logger.experiment.add_images("val_real_fake", grid, self.current_epoch)

        return total_loss

    def shared_step(self, batch, batch_idx):
        drrs = batch["drrs"]
        xrays = batch["xrays"]
        masks = batch["mask"]

        fake_img, pred_mask = self(drrs, xrays)

        gan_loss = self.cyclegan.loss()
        target_size = masks.shape[2:]
        upsampled_logits = F.interpolate(
            pred_mask.logits,
            size=target_size,
            mode="bilinear",
            align_corners=False,  # Always set to False for segmentation upsampling
        )
        seg_loss = self.seg_loss(upsampled_logits, masks.long().squeeze(1))
        total_loss = gan_loss + seg_loss
        return gan_loss, seg_loss, total_loss, fake_img

    def configure_optimizers(self):
        gan_params = self.cyclegan.parameters()
        seg_params = self.segmentation.parameters()

        # 2. Combine the iterables of parameters into one list
        # The '*' operator unpacks the iterables, but list concatenation is cleaner for parameter generators:
        all_params = list(gan_params) + list(seg_params)
        optimizer = torch.optim.Adam(all_params, lr=float(self.gan_lr))
        return optimizer

import os
from .mapped_dataset import MappedDataset
from .cached_dataset import CachedMappedDataset
from torch.utils.data import DataLoader
from torchvision import transforms
import pytorch_lightning as pl


class DataModule(pl.LightningDataModule):
    def __init__(
        self,
        drrs_dir,
        xrays_dir,
        masks_dir,
        transform=None,
        batch_size=4,
        num_workers=4,
    ):
        super().__init__()
        self.drrs_dir = drrs_dir
        self.xrays_dir = xrays_dir
        self.masks_dir = masks_dir
        self.batch_size = batch_size
        self.num_workers = num_workers

        if transform is None:
            self.transform = transforms.Compose(
                [
                    transforms.ToTensor(),
                ]
            )
        else:
            self.transform = transform

    def setup(self, stage=None):
        if stage == "fit":
            self.train_dataset = CachedMappedDataset(
                os.path.join(self.drrs_dir, "train"),
                os.path.join(self.xrays_dir, "train"),
                os.path.join(self.masks_dir, "train"),
                transform=self.transform,
            )
            self.val_dataset = CachedMappedDataset(
                os.path.join(self.drrs_dir, "val"),
                os.path.join(self.xrays_dir, "val"),
                os.path.join(self.masks_dir, "val"),
                transform=self.transform,
            )
        if stage == "test":
            self.dataset = CachedMappedDataset(
                os.path.join(self.drrs_dir, "test"),
                os.path.join(self.xrays_dir, "test"),
                os.path.join(self.masks_dir, "test"),
                transform=self.transform,
            )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )

    def test_dataloader(self):
        # Prefer an explicit test dataset; otherwise reuse the validation dataset
        return DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )

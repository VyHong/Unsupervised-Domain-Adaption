import os
from .dataset import MappedDataset
from .cached_dataset import CachedMappedDataset    
from torch.utils.data import DataLoader
from torchvision import transforms
import pytorch_lightning as pl


class DataModule(pl.LightningDataModule):
    def __init__(self, drrs_dir, xrays_dir,masks_dir, batch_size=4, num_workers=4):
        super().__init__()
        self.drrs_dir = drrs_dir
        self.xrays_dir = xrays_dir
        self.masks_dir = masks_dir
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
        ])

    def setup(self, stage=None):
        self.train_dataset = CachedMappedDataset(
            os.path.join(self.drrs_dir, "train"),
            os.path.join(self.xrays_dir, "train"),
            os.path.join(self.masks_dir, "train"),
            transform=self.transform
        )
        self.val_dataset = CachedMappedDataset(
            os.path.join(self.drrs_dir, "val"),
            os.path.join(self.xrays_dir, "val"),
            os.path.join(self.masks_dir, "val"),
            transform=self.transform
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers
        )

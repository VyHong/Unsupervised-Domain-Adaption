import os
from .mapped_dataset import MappedDataset
from .cached_dataset import CachedMappedDataset
from torch.utils.data import DataLoader
from torchvision import transforms
from src.utils.functions import normalize_path_for_os
import pytorch_lightning as pl


class DataModule(pl.LightningDataModule):
    def __init__(
        self,
        train_json,
        val_json,
        test_json,
        transform=None,
        batch_size=4,
        num_workers=4,
    ):
        super().__init__()
        self.train_json = normalize_path_for_os(train_json)
        self.val_json = normalize_path_for_os(val_json)
        self.test_json = normalize_path_for_os(test_json)
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
            self.train_dataset = MappedDataset(
                self.train_json,
                transform=self.transform,
            )
            self.val_dataset = MappedDataset(
                self.val_json,
                transform=self.transform,
            )
        if stage == "test":
            self.dataset = MappedDataset(
                self.test_json,
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
        loader = DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )
        print(len(loader))
        return loader

    def test_dataloader(self):
        # Prefer an explicit test dataset; otherwise reuse the validation dataset
        return DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )

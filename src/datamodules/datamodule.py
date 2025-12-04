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
        trains_dir,
        vals_dir,
        tests_dir,
        transform=None,
        batch_size=4,
        num_workers=4,
    ):
        super().__init__()
        self.trains_dir = normalize_path_for_os(trains_dir)
        self.vals_dir = normalize_path_for_os(vals_dir)
        self.tests_dir = normalize_path_for_os(tests_dir)
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
                os.path.join(self.trains_dir, "drr"),
                os.path.join(self.vals_dir, "xray"),
                os.path.join(self.tests_dir, "mask"),
                transform=self.transform,
            )
            self.val_dataset = MappedDataset(
                os.path.join(self.trains_dir, "drr"),
                os.path.join(self.vals_dir, "xray"),
                os.path.join(self.tests_dir, "mask"),
                transform=self.transform,
            )
        if stage == "test":
            self.dataset = MappedDataset(
                os.path.join(self.trains_dir, "drr"),
                os.path.join(self.vals_dir, "xray"),
                os.path.join(self.tests_dir, "mask"),
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

import os
import json
from PIL import Image
from torch.utils.data import Dataset


class MappedDataset(Dataset):
    """
    A PyTorch Dataset that loads image paths from a JSON file.
    Files are loaded from disk on-the-fly during __getitem__.
    """

    def __init__(self, json_path, transform=None):
        # 1. Load the JSON split file
        if not os.path.exists(json_path):
            raise FileNotFoundError(f"JSON file not found: {json_path}")

        with open(json_path, "r") as f:
            self.data_samples = json.load(f)

        self.transform = transform

    def __len__(self):
        return len(self.data_samples)

    def __getitem__(self, idx):
        # 2. Extract paths from the JSON entry
        sample = self.data_samples[idx]

        drr_path = sample["drr"]
        xray_path = sample["xray"]
        mask_path = sample["mask"]

        # 3. Open images
        drr = Image.open(drr_path).convert("RGB")
        xray = Image.open(xray_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")  # Assuming mask is grayscale/index

        # 4. Apply transforms
        if self.transform:
            drr = self.transform(drr)
            xray = self.transform(xray)
            mask = self.transform(mask)

        # 5. Return tensors moved to GPU as per your original logic
        # Note: Be careful with .to("cuda") here if using num_workers > 0 in DataLoader
        return {
            "drrs": drr.to("cuda"),
            "xrays": xray.to("cuda"),
            "masks": mask.to("cuda"),
        }

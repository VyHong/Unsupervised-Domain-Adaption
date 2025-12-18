import os
import json
from PIL import Image
from torch.utils.data import Dataset


class CachedMappedDataset(Dataset):
    """
    A PyTorch Dataset class that loads paths from a JSON file and caches
    tensors in memory.
    """

    def __init__(self, json_path, transform=None):
        # 1. Load the JSON data
        if not os.path.exists(json_path):
            raise FileNotFoundError(f"JSON file not found at: {json_path}")

        with open(json_path, "r") as f:
            self.data_list = json.load(f)

        self.transform = transform
        self.length = len(self.data_list)
        self._cached_data = []

        print(
            f"Caching {self.length} samples into memory from {os.path.basename(json_path)}..."
        )

        # 2. Cache Data
        for entry in self.data_list:
            try:
                xray = Image.open(entry["xray"]).convert("RGB")
                mask = Image.open(entry["mask"]).convert(
                    "L"
                )  # L for single-channel mask
                drr = Image.open(entry["drr"]).convert("RGB")
                if self.transform:
                    xray_tensor = self.transform(xray)
                    mask_tensor = self.transform(mask)
                    drr_tensor = self.transform(drr)

                # Store the processed tensors
                self._cached_data.append(
                    {"xrays": xray_tensor, "masks": mask_tensor, "drrs": drr_tensor}
                )
            except Exception as e:
                print(f"Error loading paths from JSON entry: {entry}. Error: {e}")

        print("Caching complete.")

    def __len__(self):
        return len(self._cached_data)

    def __getitem__(self, idx):
        return self._cached_data[idx]

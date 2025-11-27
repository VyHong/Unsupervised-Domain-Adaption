import os
from glob import glob
from PIL import Image
from torch.utils.data import Dataset
import torch

class CachedMappedDataset(Dataset):
    """
    A PyTorch Dataset class that caches all image and mask data in memory
    during initialization to speed up training by eliminating disk I/O latency
    during training epochs.
    
    NOTE: This requires enough RAM to hold all images and masks simultaneously.
    """
    def __init__(self, drrs_dir, xrays_dir, masks_dir, transform=None):
        # 1. Store file paths
        drr_paths = sorted(glob(os.path.join(drrs_dir, '*')))
        xray_paths = sorted(glob(os.path.join(xrays_dir, '*')))
        mask_paths = sorted(glob(os.path.join(masks_dir, '*')))
        
        # Ensure all sets have the same length
        if not (len(drr_paths) == len(xray_paths) == len(mask_paths)):
             raise ValueError("The number of DRR, X-ray, and Mask files must be equal.")
             
        self.length = len(drr_paths)
        self.transform = transform
        
        # 2. Initialize in-memory storage (lists to hold the cached tensors)
        self._cached_data = []

        print(f"Caching {self.length} samples into memory. This may take a moment...")
        
        # 3. Cache Data: Iterate through all paths, open, process, and apply transform
        for idx in range(self.length):
            # Open and convert images/masks (PIL)
            drr = Image.open(drr_paths[idx]).convert("RGB")
            xray = Image.open(xray_paths[idx]).convert("RGB")
            mask = Image.open(mask_paths[idx]) # Mask is usually single-channel/L

            # Apply transform and convert to Tensor (Tensors are required for the cache)
            if self.transform:
                drr_tensor = self.transform(drr)
                xray_tensor = self.transform(xray)
                # Apply transform to mask. If the transform includes ToTensor, 
                # ensure it keeps the required shape for the mask.
                mask_tensor = self.transform(mask) 
            else:
                # If no transform is defined, must manually convert to Tensor (e.g., for segmentation)
                # Note: This is a placeholder; you'd need a specific transform here 
                # (like ToTensor and potentially resizing) for a real segmentation pipeline.
                drr_tensor = torch.tensor(drr).float().permute(2, 0, 1) / 255.0 
                xray_tensor = torch.tensor(xray).float().permute(2, 0, 1) / 255.0
                mask_tensor = torch.tensor(mask).long()
            
            # Store the processed tensors
            self._cached_data.append({
                "drrs": drr_tensor,
                "xrays": xray_tensor,
                "mask": mask_tensor
            })
        
        print("Caching complete.")

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        # Simply retrieve the pre-processed tensors from memory
        return self._cached_data[idx]
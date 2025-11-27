import os
from glob import glob
from PIL import Image
from torch.utils.data import Dataset

class MappedDataset(Dataset):
    def __init__(self, drrs_dir, xrays_dir, masks_dir, transform=None):
        self.drrs = sorted(glob(os.path.join(drrs_dir, '*')))
        self.xrays = sorted(glob(os.path.join(xrays_dir, '*')))
        self.masks = sorted(glob(os.path.join(masks_dir, '*')))
        self.transform = transform

    def __len__(self):
        return len(self.drrs)

    def __getitem__(self, idx):
        drr = Image.open(self.drrs[idx]).convert("RGB")
        xray = Image.open(self.xrays[idx]).convert("RGB")
        mask = Image.open(self.masks[idx])
        if self.transform:
            drr = self.transform(drr)
            xray = self.transform(xray)
            mask = self.transform(mask)
        return {"drrs": drr,"xrays": xray, "mask": mask}
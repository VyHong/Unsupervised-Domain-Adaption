import pytorch_lightning as pl
from ..modules.module import Module  # Replace with your actual import
import yaml
from ..datamodules.cached_dataset import CachedMappedDataset
import os
from torchvision import transforms
import torch
import torchvision.utils as vutils
import torch.nn.functional as F

CHECKPOINT_PATH = r"C:\Users\waiho\Coding Projects\Unsupervised-Domain-Adaption\lightning_logs\version_40\checkpoints\joint-epoch=89-val_total_loss=1.4417.ckpt"
if __name__ == "__main__":
    # Load the model weights into a new instance of your module class
    # NOTE: If your __init__ requires 'cfg', you must pass it here.
    with open("configs/trial01.yaml", "r") as f:
        cfg = yaml.safe_load(f)
    print(cfg)
    loaded_module = Module.load_from_checkpoint(CHECKPOINT_PATH, cfg=cfg)

    drrs_dir = os.path.join(cfg["data"]["drrs"], "train")
    xrays_dir = os.path.join(cfg["data"]["xrays"], "train")
    masks_dir = os.path.join(cfg["data"]["masks"], "train")
    transform = transforms.Compose(
        [
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
        ]
    )

    test_dataset = CachedMappedDataset(
        drrs_dir=drrs_dir, xrays_dir=xrays_dir, masks_dir=masks_dir, transform=transform
    )

    input_tensors = test_dataset.__getitem__(0)
    loaded_module.eval()
    loaded_module.half()
    fake_img, pred_mask = loaded_module(
        input_tensors["drrs"].to(dtype=torch.float16),
        input_tensors["xrays"].to(dtype=torch.float16),
    )
    grid_data = [
        # Col 2: Generated Output (Fake X-ray)
        fake_img,
        # Col 4: Segmentation Prediction from Fake X-ray
        pred_mask.logits,
    ]

    vis_tensors = torch.cat(grid_data, dim=0)

    # Make the grid (e.g., max_images rows, 4 columns)
    image_grid = vutils.make_grid(
        vis_tensors,
        nrow=len(grid_data),  # Sets the number of columns to 4
        normalize=True,  # Important: Scales [-1, 1] images and [0, N] masks to [0, 1]
        scale_each=True,  # Ensures each tensor (image or mask) is scaled independently
    )

    save_path = r".\test_1.png"
    vutils.save_image(image_grid, save_path)
    # predictions = loaded_module.segmentation(input_tensor)

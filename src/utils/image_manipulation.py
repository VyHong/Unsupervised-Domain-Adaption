import torch.nn.functional as F
import torch
import torchvision.utils as vutils
import numpy as np
import matplotlib.pyplot as plt


def resample_logits(tensor, target_size):
    resampled_logits = F.interpolate(
        tensor,
        size=target_size,
        mode="bilinear",
        align_corners=False,  # Always set to False for segmentation upsampling
    )
    return resampled_logits


def save_visualization(grid, save_path):
    prepared = []
    for image in grid:
        img = image
        # Move to cpu and ensure float
        img = img.detach().cpu().float()

        # If single-channel, convert to 3 channels
        if img.ndim == 3 and img.shape[0] == 1:
            img = img.repeat_interleave(3, dim=0)

        # If image values look like they're in [-1, 1], rescale to [0, 1]
        minv = float(img.min())
        maxv = float(img.max())

        if minv < 0.0 or maxv > 1.0:
            # Common case: model outputs in [-1, 1]
            if minv >= -1.0 - 1e-3 and maxv <= 1.0 + 1e-3:
                img = (img + 1.0) / 2.0

        prepared.append(img)

    # Do NOT normalize here: normalization would change the colors of colorized
    # segmentation maps. We assume items in `prepared` are already in [0,1].
    image_grid = vutils.make_grid(prepared, nrow=len(prepared), normalize=False)
    vutils.save_image(image_grid, save_path)


def auto_palette(num_classes):
    cmap = plt.get_cmap("tab20")
    return {i: tuple(float(c) for c in cmap(i)[:3]) for i in range(num_classes)}


def logits_to_rgb(logits, num_classes):
    """
    logits:  (C, H, W) tensor
    returns: (H, W, 3) uint8 RGB mask
    """

    # 1) logits → predicted class indices
    preds = torch.argmax(logits, dim=0)  # (B, H, W)

    # 2) define color palette per class (C classes)
    # adjust to your number of classes
    palette = auto_palette(num_classes)
    # 3) create RGB image
    H, W = preds.shape
    rgb = torch.zeros((H, W, 3), dtype=torch.float32)

    for cls_id, color in palette.items():
        mask = preds == cls_id
        rgb[mask] = torch.tensor(color, dtype=torch.float32)
    rgb_output_tensor = rgb.permute(2, 0, 1)
    return rgb_output_tensor

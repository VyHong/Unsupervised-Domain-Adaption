import torch.nn.functional as F
import torch
import torchvision.utils as vutils


def resample_logits(tensor, target_size):
    resampled_logits = F.interpolate(
        tensor,
        size=target_size,
        mode="bilinear",
        align_corners=False,  # Always set to False for segmentation upsampling
    )
    return resampled_logits


def save_visualization(grid, save_path):
    for i, image in enumerate(grid):
        if image.shape[0] == 1:
            # Use repeat_interleave to copy the single channel 3 times
            # This transforms [1, 512, 512] to [3, 512, 512]
            grayscale_rgb = image.repeat_interleave(3, dim=0)
            grid[i] = grayscale_rgb

    image_grid = vutils.make_grid(
        grid,
        nrow=len(grid),  # Sets the number of columns to 4
        normalize=True,  # Important: Scales [-1, 1] images and [0, N] masks to [0, 1]
        scale_each=True,  # Ensures each tensor (image or mask) is scaled independently
    )
    vutils.save_image(image_grid, save_path)

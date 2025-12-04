import torch.nn.functional as F
import torch
import torchvision.utils as vutils
import numpy as np
import matplotlib.cm as cm


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
        try:
            minv = float(img.min())
            maxv = float(img.max())
        except Exception:
            minv, maxv = 0.0, 1.0

        if minv < 0.0 or maxv > 1.0:
            # Common case: model outputs in [-1, 1]
            if minv >= -1.0 - 1e-3 and maxv <= 1.0 + 1e-3:
                img = (img + 1.0) / 2.0
            else:
                # Otherwise clamp to [0,1]
                img = img.clamp(0.0, 1.0)

        prepared.append(img)

    # Do NOT normalize here: normalization would change the colors of colorized
    # segmentation maps. We assume items in `prepared` are already in [0,1].
    image_grid = vutils.make_grid(prepared, nrow=len(prepared), normalize=False)
    vutils.save_image(image_grid, save_path)


def colorize_segmentation(
    mask_input: torch.Tensor,
    num_classes: int,
    cmap_name: str = "viridis",
    channel_first: bool = True,
) -> torch.Tensor:
    """
    Converts a single-channel segmentation mask (or batch) into an RGB image tensor.

    Automatically detects and handles single images ([H, W] or [1, H, W])
    and batches ([B, H, W] or [B, 1, H, W]).

    Args:
        mask_input (torch.Tensor): The input mask(s).
        num_classes (int): The total count of classes (used for the color palette).
        cmap_name (str): The name of the Matplotlib colormap to use.
        channel_first (bool): If True, returns shape [B, 3, H, W] (or [3, H, W]).

    Returns:
        torch.Tensor: The resulting RGB image tensor batch.
    """
    original_device = mask_input.device
    original_ndim = mask_input.ndim

    # --- 1. Standardize Input to Batch Format [B, H, W] ---

    mask_batch_np = mask_input.cpu().numpy().astype(np.int64)

    # Convert single image to batch of size 1
    if original_ndim == 2:
        # [H, W] -> [1, H, W]
        mask_batch_np = np.expand_dims(mask_batch_np, axis=0)
    elif original_ndim == 3 and mask_batch_np.shape[0] == 1:
        # [1, H, W] is already a batch of 1, keep as is
        pass
    elif mask_batch_np.ndim == 4 and mask_batch_np.shape[1] == 1:
        # [B, 1, H, W] -> [B, H, W]
        mask_batch_np = mask_batch_np.squeeze(1)

    if mask_batch_np.ndim != 3:
        raise ValueError(
            f"Input mask could not be standardized to [B, H, W]. Final shape: {mask_batch_np.shape}"
        )

    # --- 2. Colorization Logic (Applied to Batch) ---

    # a) Identify ALL unique labels across the entire batch
    unique_labels_all = np.unique(mask_batch_np)
    n_unique_actual = len(unique_labels_all)

    if num_classes < n_unique_actual:
        raise ValueError(
            f"The 'num_classes' ({num_classes}) is less than the actual unique labels found ({n_unique_actual})."
        )

    # b) Generate a FIXED Colormap
    cmap = cm.get_cmap(cmap_name, num_classes)
    class_colors = cmap(np.arange(num_classes))[:, :3].astype(np.float32)

    # c) Create Remapping Lookup Table: Maps actual label value to sequential index
    label_to_index_map = {label: i for i, label in enumerate(unique_labels_all)}

    # d) Process the batch
    colored_images_list = []

    for mask in mask_batch_np:
        # Remap the mask's values
        remapped_mask = np.vectorize(label_to_index_map.get)(mask)
        # Look up colors: Output shape is [H, W, 3]
        rgb_image_np = class_colors[remapped_mask]
        colored_images_list.append(rgb_image_np)

    # --- 3. Final Stack, Conversion, and Output Shape ---

    # Stack the list of [H, W, 3] images into a [B, H, W, 3] NumPy array
    rgb_batch_output_np = np.stack(colored_images_list, axis=0)

    # Convert back to PyTorch tensor and move to the original device
    rgb_output_tensor = torch.from_numpy(rgb_batch_output_np).to(original_device)

    # Handle final output dimensions based on original input type and channel_first flag

    # If the original input was a single image (ndim 2 or 3 with size 1),
    # and not explicitly channel_first, squeeze the batch dimension.
    if original_ndim < 4 and not channel_first:
        rgb_output_tensor = rgb_output_tensor.squeeze(0)  # [1, H, W, 3] -> [H, W, 3]

    if channel_first:
        # Handle both batch and single image squeeze (the squeeze(0) is undone by the permute for single image)
        if original_ndim < 4:
            # Single image: [H, W, 3] -> [3, H, W]
            rgb_output_tensor = rgb_output_tensor.squeeze(0).permute(2, 0, 1)
        else:
            # Batch: [B, H, W, 3] -> [B, 3, H, W]
            rgb_output_tensor = rgb_output_tensor.permute(0, 3, 1, 2)

    return rgb_output_tensor

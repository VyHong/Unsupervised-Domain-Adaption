import os
import shutil
import argparse
from glob import glob
from sklearn.model_selection import (
    train_test_split,
)  # Used for reliable shuffle and split

# --- Configuration ---
TRAIN_RATIO = 0.80  # 80% for training
VAL_RATIO = 0.10  # 10% for validation
TEST_RATIO = 0.10  # 10% for testing (Note: Ratios must sum to 1.0)
RANDOM_STATE = 42  # Seed for reproducibility of shuffle/split
# ---------------------


def get_case_id(filename):
    """
    Extracts the unique case ID from the file name.

    Examples:
    - '167_xrayimage_view_2.png' -> '167_view_2'
    - '171_semantic_label_view_2.png' -> '171_view_2'
    """
    base = os.path.basename(filename).replace(".png", "")  # Remove extension

    # Split by underscore
    parts = base.split("_")

    if len(parts) >= 3:
        # Assumes structure is ID_type_view_ID/ID_type_view_ID
        # We want to capture the initial ID and the view part: '167_view_2'
        case_id = f"{parts[0]}_{parts[-2]}_{parts[-1]}"
        return case_id.replace("view_", "view")  # Clean up if needed

    # Fallback for unexpected naming, using the whole base name
    return base


def split_segmentation_data(
    xray_dir, mask_dir, dest_dir, train_ratio, val_ratio, test_ratio
):
    """
    Splits paired X-ray and Mask files based on unique case IDs and copies them
    into the standard train/val/test structure.
    """
    # 1. Check Ratios
    if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-6:
        print("Error: Train, Val, and Test ratios must sum up to 1.0.")
        return

    print(
        f"--- Starting Segmentation Data Split ({train_ratio*100:.0f}%/{val_ratio*100:.0f}%/{test_ratio*100:.0f}%) ---"
    )
    print(f"X-ray Source: {xray_dir}")
    print(f"Mask Source: {mask_dir}")
    print(f"Destination Root: {dest_dir}")

    # 2. Collect all X-ray file paths
    all_xray_paths = sorted(
        glob(os.path.join(xray_dir, "*.png"))
    )  # Use *.png or appropriate extension

    if not all_xray_paths:
        print(f"Error: No X-ray files found in {xray_dir}")
        return

    # 3. Map Case IDs to full paths
    # We use a dictionary to store all paths based on their extracted ID
    xray_id_to_path = {get_case_id(p): p for p in all_xray_paths}

    # Check for corresponding mask
    unique_ids = list(xray_id_to_path.keys())
    paired_ids = []

    for case_id in unique_ids:
        # Infer mask filename pattern based on the X-ray file's pattern
        # This requires some knowledge of the naming convention logic:
        # 167_xrayimage_view_2.png <-> 167_semantic_label_view_2.png
        xray_path = xray_id_to_path[case_id]
        xray_filename = os.path.basename(xray_path)

        # We need to construct the corresponding mask filename.
        # This is a critical step depending on your exact naming.
        # Assuming only the middle part changes:
        mask_filename = xray_filename.replace("xrayimage", "semantic_label")

        mask_path = os.path.join(mask_dir, mask_filename)

        if os.path.exists(mask_path):
            # If a matching mask is found, we keep this ID for the split
            paired_ids.append(case_id)
        else:
            print(f"Warning: No matching mask found for X-ray ID: {case_id}. Skipping.")

    if not paired_ids:
        print("Error: No paired X-ray/Mask samples found to split.")
        return

    print(f"Found {len(paired_ids)} unique, paired samples to split.")

    # 4. Shuffle and Split the paired IDs (This is the core step!)

    # Split into initial train set and a temporary remaining set (val + test)
    train_ids, remaining_ids = train_test_split(
        paired_ids,
        test_size=(VAL_RATIO + TEST_RATIO),
        shuffle=True,  # Perform initial shuffle
        random_state=RANDOM_STATE,
    )

    # Calculate the new ratio for val/test split from the remaining set
    val_test_ratio = VAL_RATIO / (VAL_RATIO + TEST_RATIO)

    # Split the remaining set into validation and test
    val_ids, test_ids = train_test_split(
        remaining_ids,
        test_size=(1 - val_test_ratio),
        shuffle=False,  # No need to shuffle again
        random_state=RANDOM_STATE,
    )

    split_ids = {"train": train_ids, "val": val_ids, "test": test_ids}

    print(
        f"Split counts: Train={len(train_ids)}, Validation={len(val_ids)}, Test={len(test_ids)}"
    )

    # 5. Define and Create Destination Folders
    for split_name in split_ids.keys():
        os.makedirs(os.path.join(dest_dir, split_name, "xray"), exist_ok=True)
        os.makedirs(os.path.join(dest_dir, split_name, "mask"), exist_ok=True)

    # 6. Copy the Paired Files
    for split_name, case_ids in split_ids.items():
        dest_img_path = os.path.join(dest_dir, split_name, "xray")
        dest_mask_path = os.path.join(dest_dir, split_name, "mask")

        print(f"\nCopying {len(case_ids)} samples to {split_name}...")

        for case_id in case_ids:
            xray_path = xray_id_to_path[case_id]

            # Re-derive the mask filename based on the stored X-ray path
            xray_filename = os.path.basename(xray_path)
            mask_filename = xray_filename.replace("xrayimage", "semantic_label")
            mask_path = os.path.join(mask_dir, mask_filename)

            # Copy X-ray (Image)
            shutil.copy2(xray_path, os.path.join(dest_img_path, xray_filename))
            # Copy Mask
            shutil.copy2(mask_path, os.path.join(dest_mask_path, mask_filename))

    print("\n✅ Segmentation Data splitting complete!")
    print(
        f"Data has been saved to '{dest_dir}' with the structure: /Train/Images, /Train/Masks, etc."
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Randomly splits paired X-ray and Mask files into train, val, and test folders based on unique case IDs."
    )

    parser.add_argument(
        "xray_dir",
        type=str,
        help="The path to the directory containing all X-ray images.",
    )

    parser.add_argument(
        "mask_dir",
        type=str,
        help="The path to the directory containing all corresponding Mask files.",
    )

    parser.add_argument(
        "dest_dir",
        type=str,
        help="The path to the directory where the 'train', 'val', and 'test' folders will be created.",
    )

    args = parser.parse_args()

    # You must install scikit-learn for this to work: pip install scikit-learn
    split_segmentation_data(
        args.xray_dir, args.mask_dir, args.dest_dir, TRAIN_RATIO, VAL_RATIO, TEST_RATIO
    )

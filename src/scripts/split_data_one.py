import os
import shutil
import random
import argparse
from glob import glob

# --- Configuration ---
TRAIN_RATIO = 0.80  # 80% for training
VAL_RATIO = 0.10  # 10% for validation
TEST_RATIO = 0.10  # 10% for testing (Note: Ratios must sum to 1.0)
SUB_FOLDER_NAME = "drr"  # The required intermediate folder name (e.g., 'drr')
# ---------------------


def split_single_folder_with_subfolder(
    source_dir, dest_dir, train_ratio, val_ratio, test_ratio, sub_folder
):
    """
    Splits all files from the source directory and copies them into
    {dest_dir}/{train, val, test}/{sub_folder}/{filename}.
    """
    # 1. Check Ratios
    if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-6:
        print("Error: Train, Val, and Test ratios must sum up to 1.0.")
        return

    print(
        f"--- Starting Single Folder Data Split ({train_ratio*100:.0f}%/{val_ratio*100:.0f}%/{test_ratio*100:.0f}%) ---"
    )
    print(f"Source: {source_dir}")
    print(f"Destination: {dest_dir}/.../{sub_folder}/")

    # 2. Get all file paths
    all_files = [
        os.path.abspath(f)
        for f in glob(os.path.join(source_dir, "*"))
        if os.path.isfile(f)
    ]

    if not all_files:
        print(f"Error: No files found in the source directory: {source_dir}")
        return

    print(f"Found {len(all_files)} files to split.")

    # 3. Shuffle the list of files randomly
    random.seed(42)  # Set seed for reproducibility
    random.shuffle(all_files)

    # 4. Calculate split indices
    total_count = len(all_files)
    train_end = int(total_count * train_ratio)
    val_end = train_end + int(total_count * val_ratio)

    # Assign files to lists
    train_files = all_files[:train_end]
    val_files = all_files[train_end:val_end]
    test_files = all_files[val_end:]

    print(
        f"Split counts: Train={len(train_files)}, Validation={len(val_files)}, Test={len(test_files)}"
    )

    # 5. Define and Create Destination Folders (MODIFIED HERE)
    sets = {"train": train_files, "val": val_files, "test": test_files}

    # Create the top-level destination directory
    os.makedirs(dest_dir, exist_ok=True)

    # Create the train/val/test subdirectories, each containing the 'drr' subfolder
    for folder in sets.keys():
        dest_folder = os.path.join(dest_dir, folder, sub_folder)  # Nested structure
        os.makedirs(dest_folder, exist_ok=True)

    # 6. Copy the files (MODIFIED HERE)
    for folder_name, file_list in sets.items():
        # Destination path now includes the required sub_folder
        dest_path = os.path.join(dest_dir, folder_name, sub_folder)
        print(
            f"Copying {len(file_list)} files to {os.path.basename(dest_dir)}/{folder_name}/{sub_folder}/..."
        )

        for file_path in file_list:
            file_name = os.path.basename(file_path)
            # Copy the file to the new nested destination path
            shutil.copy2(file_path, os.path.join(dest_path, file_name))

    print("\n✅ Single folder data splitting complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Randomly splits files from a source directory into train, val, and test folders with a specific subfolder (e.g., 'drr')."
    )

    parser.add_argument(
        "source_dir",
        type=str,
        help="The path to the directory containing all source data.",
    )

    parser.add_argument(
        "dest_dir",
        type=str,
        help="The path to the root directory where the 'train', 'val', and 'test' folders will be created.",
    )

    args = parser.parse_args()

    split_single_folder_with_subfolder(
        args.source_dir,
        args.dest_dir,
        TRAIN_RATIO,
        VAL_RATIO,
        TEST_RATIO,
        SUB_FOLDER_NAME,
    )

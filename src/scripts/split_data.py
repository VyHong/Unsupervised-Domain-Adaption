import os
import shutil
import random
import argparse
from glob import glob

# --- Configuration ---
TRAIN_RATIO = 0.70  # 70% for training
VAL_RATIO = 0.15  # 15% for validation
TEST_RATIO = 0.15  # 15% for testing (Note: Ratios must sum to 1.0)
# ---------------------


def split_data(source_dir, dest_dir, train_ratio, val_ratio, test_ratio):
    """
    Splits all files from the source directory and moves them into train,
    val, and test folders created inside the destination directory.
    """
    # 1. Check Ratios
    if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-6:
        print("Error: Train, Val, and Test ratios must sum up to 1.0.")
        return

    print(
        f"--- Starting Data Split ({train_ratio*100:.0f}%/{val_ratio*100:.0f}%/{test_ratio*100:.0f}%) ---"
    )
    print(f"Source: {source_dir}")
    print(f"Destination: {dest_dir}")

    # 2. Get all file paths
    # We find all files in the source directory (excluding subdirectories)
    # The absolute path ensures shutil.move works reliably
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
    random.shuffle(all_files)

    # 4. Calculate split indices
    total_count = len(all_files)
    train_end = int(total_count * train_ratio)
    val_end = train_end + int(total_count * val_ratio)

    # Ensure all files are included, accounting for rounding
    train_files = all_files[:train_end]
    val_files = all_files[train_end:val_end]
    test_files = all_files[val_end:]

    print(
        f"Split counts: Train={len(train_files)}, Validation={len(val_files)}, Test={len(test_files)}"
    )

    # 5. Define and Create Destination Folders
    sets = {"train": train_files, "val": val_files, "test": test_files}

    # Create the top-level destination directory if needed
    os.makedirs(dest_dir, exist_ok=True)

    # Create the train/val/test subdirectories
    for folder in sets.keys():
        dest_folder = os.path.join(dest_dir, folder)
        os.makedirs(dest_folder, exist_ok=True)

    # 6. Move the files
    for folder_name, file_list in sets.items():
        dest_path = os.path.join(dest_dir, folder_name)
        print(
            f"Moving {len(file_list)} files to {os.path.basename(dest_dir)}/{folder_name}/..."
        )

        for file_path in file_list:
            file_name = os.path.basename(file_path)
            # shutil.move moves the file from the source to the destination path
            shutil.copy2(file_path, os.path.join(dest_path, file_name))

    print("\n✅ Data splitting complete!")
    print(
        f"Data has been moved from '{os.path.basename(source_dir)}' to '{os.path.basename(dest_dir)}'."
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Randomly splits files from a source directory into train, val, and test folders in a specified destination directory."
    )

    parser.add_argument(
        "source_dir",
        type=str,
        help="The absolute path to the directory containing the source data (e.g., /home/user/data/raw).",
    )

    parser.add_argument(
        "dest_dir",
        type=str,
        help="The absolute path to the directory where the 'train', 'val', and 'test' folders will be created (e.g., /home/user/data/split).",
    )

    args = parser.parse_args()

    split_data(args.source_dir, args.dest_dir, TRAIN_RATIO, VAL_RATIO, TEST_RATIO)

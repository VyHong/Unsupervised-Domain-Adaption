import os
import shutil
import random
from glob import glob

# --- Configuration ---
SOURCE_DIR = "/home/vy/welsh_dragon/xray"  # CHANGE THIS to the absolute path of your data folder (e.g., "/home/vy/welsh_dragon/xray")
TRAIN_RATIO = 0.80  # 80% for training
VAL_RATIO = 0.10  # 10% for validation
TEST_RATIO = 0.10  # 10% for testing (Note: Ratios must sum to 1.0)
# ---------------------


def split_data(source_dir, train_ratio, val_ratio, test_ratio):
    """
    Splits all files in the source directory into train, val, and test folders.
    """
    # 1. Check Ratios
    if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-6:
        print("Error: Train, Val, and Test ratios must sum up to 1.0.")
        return

    print(
        f"--- Starting Data Split ({train_ratio*100:.0f}%/{val_ratio*100:.0f}%/{test_ratio*100:.0f}%) ---"
    )

    # 2. Get all file paths
    # We use glob to find all files (excluding subdirectories)
    all_files = [f for f in glob(os.path.join(source_dir, "*")) if os.path.isfile(f)]

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

    # Create the destination directories if they don't exist
    for folder in sets.keys():
        dest_folder = os.path.join(source_dir, folder)
        os.makedirs(dest_folder, exist_ok=True)

    # 6. Move the files
    for folder_name, file_list in sets.items():
        dest_path = os.path.join(source_dir, folder_name)
        print(f"Moving {len(file_list)} files to ./{folder_name}/...")

        for file_path in file_list:
            # os.path.basename extracts just the file name from the full path
            file_name = os.path.basename(file_path)
            # shutil.move moves the file from the source to the destination path
            shutil.move(file_path, os.path.join(dest_path, file_name))

    print("\n✅ Data splitting complete!")
    print(
        f"Source directory '{os.path.basename(source_dir)}' now contains 'train', 'val', and 'test' folders."
    )


if __name__ == "__main__":
    split_data(SOURCE_DIR, TRAIN_RATIO, VAL_RATIO, TEST_RATIO)

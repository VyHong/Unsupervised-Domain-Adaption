import os
from pathlib import Path


def normalize_path_for_os(incompatible_path_string):
    """
    Normalizes a path string (potentially containing mixed separators)
    for the current operating system.
    """
    temp_path = incompatible_path_string.replace("\\", "/")

    return os.path.normpath(temp_path)

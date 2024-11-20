def is_dir_writeable(dir_path):
    """
    Check if a directory is writeable.

    Args:
        dir_path (str) or (Path): The path to the directory.

    Returns:
        bool: True if the directory is writeable, False otherwise.
    """
    return os.access(str(dir_path), os.W_OK)

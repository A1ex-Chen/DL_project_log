def check_is_path_safe(basedir, path):
    """
    Check if the resolved path is under the intended directory to prevent path traversal.

    Args:
        basedir (Path | str): The intended directory.
        path (Path | str): The path to check.

    Returns:
        (bool): True if the path is safe, False otherwise.
    """
    base_dir_resolved = Path(basedir).resolve()
    path_resolved = Path(path).resolve()
    return path_resolved.exists() and path_resolved.parts[:len(
        base_dir_resolved.parts)] == base_dir_resolved.parts

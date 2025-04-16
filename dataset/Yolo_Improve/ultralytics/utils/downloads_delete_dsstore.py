def delete_dsstore(path, files_to_delete=('.DS_Store', '__MACOSX')):
    """
    Deletes all ".DS_store" files under a specified directory.

    Args:
        path (str, optional): The directory path where the ".DS_store" files should be deleted.
        files_to_delete (tuple): The files to be deleted.

    Example:
        ```python
        from ultralytics.utils.downloads import delete_dsstore

        delete_dsstore('path/to/dir')
        ```

    Note:
        ".DS_store" files are created by the Apple operating system and contain metadata about folders and files. They
        are hidden system files and can cause issues when transferring files between different operating systems.
    """
    for file in files_to_delete:
        matches = list(Path(path).rglob(file))
        LOGGER.info(f'Deleting {file} files: {matches}')
        for f in matches:
            f.unlink()

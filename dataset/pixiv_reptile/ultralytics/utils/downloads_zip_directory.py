def zip_directory(directory, compress=True, exclude=('.DS_Store',
    '__MACOSX'), progress=True):
    """
    Zips the contents of a directory, excluding files containing strings in the exclude list. The resulting zip file is
    named after the directory and placed alongside it.

    Args:
        directory (str | Path): The path to the directory to be zipped.
        compress (bool): Whether to compress the files while zipping. Default is True.
        exclude (tuple, optional): A tuple of filename strings to be excluded. Defaults to ('.DS_Store', '__MACOSX').
        progress (bool, optional): Whether to display a progress bar. Defaults to True.

    Returns:
        (Path): The path to the resulting zip file.

    Example:
        ```python
        from ultralytics.utils.downloads import zip_directory

        file = zip_directory('path/to/dir')
        ```
    """
    from zipfile import ZIP_DEFLATED, ZIP_STORED, ZipFile
    delete_dsstore(directory)
    directory = Path(directory)
    if not directory.is_dir():
        raise FileNotFoundError(f"Directory '{directory}' does not exist.")
    files_to_zip = [f for f in directory.rglob('*') if f.is_file() and all(
        x not in f.name for x in exclude)]
    zip_file = directory.with_suffix('.zip')
    compression = ZIP_DEFLATED if compress else ZIP_STORED
    with ZipFile(zip_file, 'w', compression) as f:
        for file in TQDM(files_to_zip, desc=
            f'Zipping {directory} to {zip_file}...', unit='file', disable=
            not progress):
            f.write(file, file.relative_to(directory))
    return zip_file

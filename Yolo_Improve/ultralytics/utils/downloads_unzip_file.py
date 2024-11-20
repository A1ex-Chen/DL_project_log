def unzip_file(file, path=None, exclude=('.DS_Store', '__MACOSX'), exist_ok
    =False, progress=True):
    """
    Unzips a *.zip file to the specified path, excluding files containing strings in the exclude list.

    If the zipfile does not contain a single top-level directory, the function will create a new
    directory with the same name as the zipfile (without the extension) to extract its contents.
    If a path is not provided, the function will use the parent directory of the zipfile as the default path.

    Args:
        file (str): The path to the zipfile to be extracted.
        path (str, optional): The path to extract the zipfile to. Defaults to None.
        exclude (tuple, optional): A tuple of filename strings to be excluded. Defaults to ('.DS_Store', '__MACOSX').
        exist_ok (bool, optional): Whether to overwrite existing contents if they exist. Defaults to False.
        progress (bool, optional): Whether to display a progress bar. Defaults to True.

    Raises:
        BadZipFile: If the provided file does not exist or is not a valid zipfile.

    Returns:
        (Path): The path to the directory where the zipfile was extracted.

    Example:
        ```python
        from ultralytics.utils.downloads import unzip_file

        dir = unzip_file('path/to/file.zip')
        ```
    """
    from zipfile import BadZipFile, ZipFile, is_zipfile
    if not (Path(file).exists() and is_zipfile(file)):
        raise BadZipFile(f"File '{file}' does not exist or is a bad zip file.")
    if path is None:
        path = Path(file).parent
    with ZipFile(file) as zipObj:
        files = [f for f in zipObj.namelist() if all(x not in f for x in
            exclude)]
        top_level_dirs = {Path(f).parts[0] for f in files}
        unzip_as_dir = len(top_level_dirs) == 1
        if unzip_as_dir:
            extract_path = path
            path = Path(path) / list(top_level_dirs)[0]
        else:
            path = extract_path = Path(path) / Path(file).stem
        if path.exists() and any(path.iterdir()) and not exist_ok:
            LOGGER.warning(
                f'WARNING ⚠️ Skipping {file} unzip as destination directory {path} is not empty.'
                )
            return path
        for f in TQDM(files, desc=
            f'Unzipping {file} to {Path(path).resolve()}...', unit='file',
            disable=not progress):
            if '..' in Path(f).parts:
                LOGGER.warning(
                    f'Potentially insecure file path: {f}, skipping extraction.'
                    )
                continue
            zipObj.extract(f, extract_path)
    return path

def unzip_file(file, path=None, exclude=('.DS_Store', '__MACOSX'), exist_ok
    =False):
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

    Raises:
        BadZipFile: If the provided file does not exist or is not a valid zipfile.

    Returns:
        (Path): The path to the directory where the zipfile was extracted.
    """
    if not (Path(file).exists() and is_zipfile(file)):
        raise BadZipFile(f"File '{file}' does not exist or is a bad zip file.")
    if path is None:
        path = Path(file).parent
    with ZipFile(file) as zipObj:
        file_list = [f for f in zipObj.namelist() if all(x not in f for x in
            exclude)]
        top_level_dirs = {Path(f).parts[0] for f in file_list}
        if len(top_level_dirs) > 1 or not file_list[0].endswith('/'):
            path = Path(path) / Path(file).stem
        extract_path = Path(path) / list(top_level_dirs)[0]
        if extract_path.exists() and any(extract_path.iterdir()
            ) and not exist_ok:
            LOGGER.info(f'Skipping {file} unzip (already unzipped)')
            return path
        for f in file_list:
            zipObj.extract(f, path=path)
    return path

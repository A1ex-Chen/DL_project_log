def increment_path(path: (str | Path), exist_ok: bool=False, mkdir: bool=False
    ) ->Path:
    """Increments a path (appends a suffix) if it already exists.

    Args:
        path (Union[str, Path]): The path to increment.
        exist_ok (bool, optional): If set to True, no increment will be done. Defaults to False.
        mkdir (bool, optional): If set to True, the directory will be created. Defaults to False.

    Returns:
        Path: The incremented path.
    """
    path = Path(path)
    if exist_ok:
        return path
    suffix = 1
    new_path = path
    while new_path.exists():
        new_path = Path(f'{path}_{suffix}')
        suffix += 1
    if mkdir:
        new_path.mkdir(parents=True, exist_ok=True)
    return new_path

@contextmanager
def spaces_in_path(path):
    """
    Context manager to handle paths with spaces in their names. If a path contains spaces, it replaces them with
    underscores, copies the file/directory to the new path, executes the context code block, then copies the
    file/directory back to its original location.

    Args:
        path (str | Path): The original path.

    Yields:
        (Path): Temporary path with spaces replaced by underscores if spaces were present, otherwise the original path.

    Example:
        ```python
        with ultralytics.utils.files import spaces_in_path

        with spaces_in_path('/path/with spaces') as new_path:
            # Your code here
        ```
    """
    if ' ' in str(path):
        string = isinstance(path, str)
        path = Path(path)
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir) / path.name.replace(' ', '_')
            if path.is_dir():
                shutil.copytree(path, tmp_path)
            elif path.is_file():
                tmp_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(path, tmp_path)
            try:
                yield str(tmp_path) if string else tmp_path
            finally:
                if tmp_path.is_dir():
                    shutil.copytree(tmp_path, path, dirs_exist_ok=True)
                elif tmp_path.is_file():
                    shutil.copy2(tmp_path, path)
    else:
        yield path

@staticmethod
def _unzip(path):
    """Unzip data.zip."""
    if not str(path).endswith('.zip'):
        return False, None, path
    unzip_dir = unzip_file(path, path=path.parent)
    assert unzip_dir.is_dir(
        ), f'Error unzipping {path}, {unzip_dir} not found. path/to/abc.zip MUST unzip to path/to/abc/'
    return True, str(unzip_dir), find_dataset_yaml(unzip_dir)

def _unzip(self, path):
    if not str(path).endswith('.zip'):
        return False, None, path
    assert Path(path).is_file(), f'Error unzipping {path}, file not found'
    ZipFile(path).extractall(path=path.parent)
    dir = path.with_suffix('')
    assert dir.is_dir(
        ), f'Error unzipping {path}, {dir} not found. path/to/abc.zip MUST unzip to path/to/abc/'
    return True, str(dir), self._find_yaml(dir)

def _open(self, path, mode='r', **kwargs):
    return PathManager.open(self._get_local_path(path), mode, **kwargs)

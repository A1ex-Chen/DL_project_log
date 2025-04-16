def _loadtxt(fn):
    return np.loadtxt(os.path.join(self._data_dir, fn), dtype=str)

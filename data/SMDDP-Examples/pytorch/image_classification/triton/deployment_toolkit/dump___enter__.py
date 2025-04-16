def __enter__(self):
    if self._output_dir.exists() and len(list(self._output_dir.iterdir())):
        raise ValueError(f'{self._output_dir.as_posix()} is not empty')
    self._output_dir.mkdir(parents=True, exist_ok=True)
    return self

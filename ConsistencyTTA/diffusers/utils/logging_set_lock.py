def set_lock(self, *args, **kwargs):
    self._lock = None
    if _tqdm_active:
        return tqdm_lib.tqdm.set_lock(*args, **kwargs)

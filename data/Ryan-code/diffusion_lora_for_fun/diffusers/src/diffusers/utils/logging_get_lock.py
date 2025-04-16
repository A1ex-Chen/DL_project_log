def get_lock(self):
    if _tqdm_active:
        return tqdm_lib.tqdm.get_lock()

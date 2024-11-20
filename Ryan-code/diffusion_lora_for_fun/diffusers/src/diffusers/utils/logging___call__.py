def __call__(self, *args, **kwargs):
    if _tqdm_active:
        return tqdm_lib.tqdm(*args, **kwargs)
    else:
        return EmptyTqdm(*args, **kwargs)

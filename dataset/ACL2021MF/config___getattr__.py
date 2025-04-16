def __getattr__(self, attr: str):
    return self._C.__getattr__(attr)

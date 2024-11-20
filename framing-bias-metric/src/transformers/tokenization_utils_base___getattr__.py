def __getattr__(self, item: str):
    try:
        return self.data[item]
    except KeyError:
        raise AttributeError

def __new__(cls, dataset, map_func):
    is_iterable = isinstance(dataset, data.IterableDataset)
    if is_iterable:
        return _MapIterableDataset(dataset, map_func)
    else:
        return super().__new__(cls)

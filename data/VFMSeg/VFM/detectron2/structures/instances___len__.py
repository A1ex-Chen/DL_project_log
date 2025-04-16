def __len__(self) ->int:
    for v in self._fields.values():
        return v.__len__()
    raise NotImplementedError('Empty Instances does not support __len__!')

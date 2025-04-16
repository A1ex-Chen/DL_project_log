def __setitem__(self, index: Union[list, int], value: Union[list, int]) ->None:
    """Retrieve a specific transform or a set of transforms using indexing."""
    assert isinstance(index, (int, list)
        ), f'The indices should be either list or int type but got {type(index)}'
    if isinstance(index, list):
        assert isinstance(value, list
            ), f'The indices should be the same type as values, but got {type(index)} and {type(value)}'
    if isinstance(index, int):
        index, value = [index], [value]
    for i, v in zip(index, value):
        assert i < len(self.transforms
            ), f'list index {i} out of range {len(self.transforms)}.'
        self.transforms[i] = v

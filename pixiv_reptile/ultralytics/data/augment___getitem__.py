def __getitem__(self, index: Union[list, int]) ->'Compose':
    """Retrieve a specific transform or a set of transforms using indexing."""
    assert isinstance(index, (int, list)
        ), f'The indices should be either list or int type but got {type(index)}'
    index = [index] if isinstance(index, int) else index
    return Compose([self.transforms[i] for i in index])

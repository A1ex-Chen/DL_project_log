def __handle_indices__(self, indices: Union[int, float, List[Union[int,
    float]], torch.Tensor, slice], indice_max_length: int=None):
    if indice_max_length is None:
        indice_max_length = len(self.__data__)
    if isinstance(indices, slice):
        indices = [x for x in range(*indices.indices(indice_max_length))]
    elif not hasattr(indices, '__len__'):
        indices = [indices]
    return indices

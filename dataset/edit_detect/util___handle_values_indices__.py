def __handle_values_indices__(self, values, indices: Union[int, float, List
    [Union[int, float]], torch.Tensor, slice], indice_max_length: int=None):
    return self.__handle_values__(values=values, indices=indices
        ), self.__handle_indices__(indices=indices, indice_max_length=
        indice_max_length)

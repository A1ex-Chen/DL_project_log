def update_by_key(self, key: Union[str, int], values, indices: Union[int,
    float, List[Union[int, float]], torch.Tensor, slice], indice_max_length:
    int=None, err_if_replace: bool=False, replace: bool=False) ->None:
    if err_if_replace and replace:
        raise ValueError(
            f"Arguement err_if_replace and replace shouldn't be true at the same time."
            )
    self.__init_data_by_key__(key=key)
    values, indices = self.__handle_values_indices__(values=values, indices
        =indices, indice_max_length=indice_max_length)
    if len(indices) != len(values):
        raise ValueError(f'values and indices should have the same length.')
    self.__update_by_indices__(key=key, values=values, indices=indices,
        err_if_replace=err_if_replace, replace=replace)

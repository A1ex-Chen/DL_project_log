def __update_by_indices__(self, key: Union[str, int], values, indices:
    Union[int, float, List[Union[int, float]], torch.Tensor, slice],
    err_if_replace: bool=False, replace: bool=False):
    if err_if_replace and replace:
        raise ValueError(
            f"Arguement err_if_replace and replace shouldn't be true at the same time."
            )
    for i, idx in enumerate(indices):
        if idx in self.__data__[key]:
            if replace and not err_if_replace:
                self.__data__[key][idx] = values[i]
            elif not replace and err_if_replace:
                raise ValueError(
                    f'Cannot update existing value with key: {key} and indice: {idx}'
                    )
        else:
            self.__data__[key][idx] = values[i]

def get_by_key(self, key: Union[str, int], indices: Union[int, float, List[
    Union[int, float]], torch.Tensor, slice], indice_max_length: int=None,
    empty_handler: str=EMPTY_HANDLER_DEFAULT, default_val=None):
    self.__init_data_by_key__(key=key)
    indices = self.__handle_indices__(indices=indices, indice_max_length=
        indice_max_length)
    ret_ls = []
    for idx in indices:
        if idx in self.__data__[key]:
            ret_ls.append(self.__data__[key][idx])
        elif empty_handler == Recorder.EMPTY_HANDLER_DEFAULT:
            ret_ls.append(default_val)
        elif empty_handler == Recorder.EMPTY_HANDLER_ERR:
            raise ValueError(f'Value at [{key}][{idx}] is empty.')
    return ret_ls

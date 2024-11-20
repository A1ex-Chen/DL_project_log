def __setitem__(self, key: torch.Tensor, value: any):
    self.__data__[self.__get_key__(key)] = value

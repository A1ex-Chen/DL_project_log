def __init_data_by_key__(self, key: Union[str, int]) ->None:
    if not key in self.__data__:
        self.__data__[key] = {}

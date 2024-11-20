def __repeat_data__(self, data: List):
    new_ls: List = []
    if self.__repeat__ > 1:
        if self.__repeat_type__ == MultiDataset.REPEAT_BY_INPLACE:
            for e in data:
                new_ls = new_ls + [e] * self.__repeat__
            return new_ls
        elif self.__repeat_type__ == MultiDataset.REPEAT_BY_APPEND:
            return data * self.__repeat__
        else:
            raise ValueError(
                f'No such repeat type, {self.__repeat_type__}, should be {MultiDataset.REPEAT_BY_INPLACE} or {MultiDataset.REPEAT_BY_APPEND}'
                )

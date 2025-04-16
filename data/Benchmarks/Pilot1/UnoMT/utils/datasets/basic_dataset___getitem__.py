def __getitem__(self, index):
    return self.__data[index].astype(self.__out_dtype)

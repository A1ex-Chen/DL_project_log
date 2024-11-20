def __getitem__(self, idx):
    return self.__trans__(self.__images__[idx]), self.__labels__[idx]

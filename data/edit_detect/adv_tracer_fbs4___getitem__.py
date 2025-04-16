def __getitem__(self, idx):
    src_ls: List = self.__src__[idx]
    if isinstance(src_ls, list):
        return [self.__loader__(data=data) for data in src_ls], self.__noise__[
            idx], self.__src_idx__[idx]
    return self.__loader__(data=src_ls), self.__noise__[idx], self.__src_idx__[
        idx]

def __scan_mdid_by__(self, archs: Union[str, List[str], Set[str], None]=
    None, datasets: Union[str, List[str], Set[str], None]=None, sizes:
    Union[int, List[int], Set[int], None]=None):
    if isinstance(sizes, int):
        sizes = [sizes]
    res: List[str] = []
    key: str = 'MDID_'
    for var in self.STATIC_VARS:
        if var[:len(key)] == key:
            if self.__scan_by__(var=var, conds=archs) and self.__scan_by__(var
                =var, conds=datasets) and self.__scan_by__(var=var, conds=sizes
                ):
                res.append(getattr(self, var))
    return list(set(res))

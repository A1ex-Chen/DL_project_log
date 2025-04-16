def __scan_ds_by__(self, datasets: Union[int, List[int], Set[int], None]=
    None, sizes: Union[int, List[int], Set[int], None]=None):
    if isinstance(sizes, int):
        sizes = [sizes]
    res: List[str] = []
    key: str = 'DS_'
    for var in self.STATIC_VARS:
        if var[:len(key)] == key:
            if self.__scan_by__(var=var, conds=datasets) and self.__scan_by__(
                var=var, conds=sizes):
                res.append(getattr(self, var))
    return list(set(res))

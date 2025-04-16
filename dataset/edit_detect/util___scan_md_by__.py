def __scan_md_by__(self, archs: Union[str, List[str], Set[str], None]=None):
    if isinstance(sizes, int):
        sizes = [sizes]
    res: List[str] = []
    key: str = 'MD_'
    for var in self.STATIC_VARS:
        if var[:len(key)] == key:
            if self.__scan_by__(var=var, conds=archs):
                res.append(getattr(self, var))
    return list(set(res))

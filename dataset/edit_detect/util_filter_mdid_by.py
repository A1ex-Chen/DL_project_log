def filter_mdid_by(self, archs: Union[str, List[str], Set[str], None]=None,
    datasets: Union[str, List[str], Set[str], None]=None, sizes: Union[int,
    List[int], Set[int], None]=None):
    return self.__scan_mdid_by__(archs=archs, datasets=datasets, sizes=sizes)

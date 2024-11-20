def filter_ds_by(self, datasets: Union[str, List[str], Set[str], None]=None,
    sizes: Union[int, List[int], Set[int], None]=None):
    return self.__scan_ds_by__(datasets=datasets, sizes=sizes)

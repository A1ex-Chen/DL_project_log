def get_dataset(root: str, size: Union[int, Tuple[int], List[int]], repeat:
    int=1, vmin_out: int=-1, vmax_out: int=1):
    return MultiDataset(src=root, size=size, repeat=repeat, vmin_out=
        vmin_out, vmax_out=vmax_out)

def get_freq_circle(shape: Union[Tuple[int], int], in_diamiter: int,
    out_diamiter: int):
    return nd_circle(shape=shape, diamiter=out_diamiter) - nd_circle(shape=
        shape, diamiter=in_diamiter)

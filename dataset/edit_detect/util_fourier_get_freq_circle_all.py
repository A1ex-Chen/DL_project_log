def get_freq_circle_all(shape: Union[Tuple[int], int]):
    n: int = max(shape)
    freq_circles: List[torch.Tensor] = []
    for i in range(n):
        freq_circles.append(get_freq_circle(shape=shape, in_diamiter=i,
            out_diamiter=i + 1))
    return freq_circles

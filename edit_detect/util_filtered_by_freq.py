def filtered_by_freq(x: torch, dim: Union[Tuple[int], int], in_diamiter:
    int, out_diamiter: int):
    if isinstance(dim, int):
        dim = [dim]
    filter_shape = tuple(x.shape[i] for i in dim)
    freq_circle: torch.Tensor = get_freq_circle(shape=filter_shape,
        in_diamiter=in_diamiter, out_diamiter=out_diamiter)
    print(f'x: {x.shape}')

    def reshape_repeat(circle: torch.Tensor, target_shape: Union[Tuple[int],
        List[int]], dim: Union[Tuple[int], int]):
        circle_reshape = [1] * len(target_shape)
        circle_repeat = list(target_shape)
        for i, elem in enumerate(dim):
            circle_reshape[elem] = target_shape[elem]
            circle_repeat[elem] = 1
        print(
            f'dim: {dim}, circle: {circle.shape}, {circle.isfinite().all()}, circle_reshape: {circle_reshape}, circle_repeat: {circle_repeat}'
            )
        return circle.reshape(circle_reshape).repeat(circle_repeat)
    return fft_nd(x=x, dim=dim) * reshape_repeat(circle=freq_circle,
        target_shape=x.shape, dim=dim)

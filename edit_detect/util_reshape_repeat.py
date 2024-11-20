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

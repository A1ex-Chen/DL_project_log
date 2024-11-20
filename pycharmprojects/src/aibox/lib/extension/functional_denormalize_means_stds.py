def denormalize_means_stds(input: Tensor, means: List[float], stds: List[float]
    ) ->Tensor:
    assert input.ndim in [3, 4]
    num_channels = input.shape[-3]
    assert len(means) == len(stds) == num_channels
    if input.ndim == 3:
        output = normalize(normalize(input, mean=(0, 0, 0), std=[(1 / v) for
            v in stds]), mean=[(-v) for v in means], std=(1, 1, 1))
        return output
    else:
        return torch.stack([normalize(normalize(it, mean=(0, 0, 0), std=[(1 /
            v) for v in stds]), mean=[(-v) for v in means], std=(1, 1, 1)) for
            it in input], dim=0)

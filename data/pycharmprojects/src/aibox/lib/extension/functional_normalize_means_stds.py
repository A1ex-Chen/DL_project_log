def normalize_means_stds(input: Tensor, means: List[float], stds: List[float]
    ) ->Tensor:
    assert input.ndim in [3, 4]
    num_channels = input.shape[-3]
    assert len(means) == len(stds) == num_channels
    if input.ndim == 3:
        return normalize(input, means, stds)
    else:
        return torch.stack([normalize(it, means, stds) for it in input], dim=0)

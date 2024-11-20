def __force_image_channel_last__(self, data: torch.Tensor):
    dim: int = data.dim()
    perm: List[int] = [i for i in range(dim)]
    channel_dim = perm[-3]
    perm = perm[:-3] + perm[-2:]
    perm = perm + [channel_dim]
    return data.permute(*perm)

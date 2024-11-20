def __force_image_channel_first__(self, data: torch.Tensor):
    dim: int = data.dim()
    perm: List[int] = [i for i in range(dim)]
    channel_dim = perm[-1]
    perm = perm[:-1]
    perm.insert(-2, channel_dim)
    return data.permute(*perm)

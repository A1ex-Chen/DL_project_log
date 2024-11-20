def _stable_argsort(vector, dim):
    scale_offset = torch.arange(vector.shape[dim], device=vector.device).view(
        1, 1, -1)
    scale_offset = scale_offset.expand(vector.shape)
    scaled_vector = vector.shape[dim] * vector + scale_offset % vector.shape[
        dim]
    return torch.argsort(scaled_vector, dim=dim)

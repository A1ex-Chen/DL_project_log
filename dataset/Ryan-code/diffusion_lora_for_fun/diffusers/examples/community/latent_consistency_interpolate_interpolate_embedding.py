@torch.no_grad()
def interpolate_embedding(self, start_embedding: torch.Tensor,
    end_embedding: torch.Tensor, num_interpolation_steps: Union[int, List[
    int]], interpolation_type: str) ->torch.Tensor:
    if interpolation_type == 'lerp':
        interpolation_fn = lerp
    elif interpolation_type == 'slerp':
        interpolation_fn = slerp
    else:
        raise ValueError(
            f"embedding_interpolation_type must be one of ['lerp', 'slerp'], got {interpolation_type}."
            )
    embedding = torch.cat([start_embedding, end_embedding])
    steps = torch.linspace(0, 1, num_interpolation_steps, dtype=embedding.dtype
        ).cpu().numpy()
    steps = np.expand_dims(steps, axis=tuple(range(1, embedding.ndim)))
    interpolations = []
    for i in range(embedding.shape[0] - 1):
        interpolations.append(interpolation_fn(embedding[i], embedding[i + 
            1], steps).squeeze(dim=1))
    interpolations = torch.cat(interpolations)
    return interpolations

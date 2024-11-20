@torch.no_grad()
def interpolate_latent(self, start_latent: torch.Tensor, end_latent: torch.
    Tensor, num_interpolation_steps: Union[int, List[int]],
    interpolation_type: str) ->torch.Tensor:
    if interpolation_type == 'lerp':
        interpolation_fn = lerp
    elif interpolation_type == 'slerp':
        interpolation_fn = slerp
    latent = torch.cat([start_latent, end_latent])
    steps = torch.linspace(0, 1, num_interpolation_steps, dtype=latent.dtype
        ).cpu().numpy()
    steps = np.expand_dims(steps, axis=tuple(range(1, latent.ndim)))
    interpolations = []
    for i in range(latent.shape[0] - 1):
        interpolations.append(interpolation_fn(latent[i], latent[i + 1],
            steps).squeeze(dim=1))
    return torch.cat(interpolations)

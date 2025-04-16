def get_xt(x0: torch.Tensor, noise_scale: Union[int, torch.FloatTensor],
    noise: torch.Tensor):
    if x0.dim() != 4:
        x0 = x0.unsqueeze(0)
    noise_scale = noise_scale.reshape((-1, *([1] * len(x0.shape[1:]))))
    return x0 + noise_scale * noise

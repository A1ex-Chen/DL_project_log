def get_fixed_latents(self, seed=0, device='cpu', dtype=torch.float32,
    shape=(1, 3, 64, 64)):
    if isinstance(device, str):
        device = torch.device(device)
    generator = torch.Generator(device=device).manual_seed(seed)
    latents = randn_tensor(shape, generator=generator, device=device, dtype
        =dtype)
    return latents

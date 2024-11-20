def get_fixed_latents(self, device, seed=0):
    if isinstance(device, str):
        device = torch.device(device)
    latent_device = torch.device('cpu')
    generator = torch.Generator(device=latent_device).manual_seed(seed)
    prompt_latents = randn_tensor((1, 77, 768), generator=generator, device
        =device, dtype=torch.float32)
    vae_latents = randn_tensor((1, 4, 64, 64), generator=generator, device=
        device, dtype=torch.float32)
    clip_latents = randn_tensor((1, 1, 512), generator=generator, device=
        device, dtype=torch.float32)
    prompt_latents = prompt_latents.to(device)
    vae_latents = vae_latents.to(device)
    clip_latents = clip_latents.to(device)
    latents = {'prompt_latents': prompt_latents, 'vae_latents': vae_latents,
        'clip_latents': clip_latents}
    return latents

def _backward(self, latent, mask, steps, t0, noises, hook_latents,
    lora_scales, cfg_scales, text_embeddings, generator):
    t0 = int(t0 * steps)
    t_begin = steps - t0
    hook_latent = hook_latents.pop()
    latent = torch.where(mask > 128, latent, hook_latent)
    for t in self.scheduler.timesteps[t_begin - 1:-1]:
        latent = self._sample(t, latent, cfg_scales.pop(), text_embeddings,
            steps, sde=True, noise=noises.pop(), lora_scale=lora_scales.pop
            (), generator=generator)
        hook_latent = hook_latents.pop()
        latent = torch.where(mask > 128, latent, hook_latent)
    return latent

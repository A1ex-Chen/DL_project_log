@torch.no_grad()
def prepare_latents(self, image: Image.Image, text_embeds: torch.Tensor,
    timesteps: torch.Tensor, guidance_scale: float, generator: Optional[
    torch.Generator]=None):
    do_classifier_free_guidance = guidance_scale > 1.0
    image = image.to(device=self.device, dtype=text_embeds.dtype)
    latent = self.vae.encode(image).latent_dist.sample(generator)
    latent = self.vae.config.scaling_factor * latent
    coupled_latents = [latent.clone(), latent.clone()]
    for i, t in tqdm(enumerate(timesteps), total=len(timesteps)):
        coupled_latents = self.noise_mixing_layer(x=coupled_latents[0], y=
            coupled_latents[1])
        for j in range(2):
            k = j ^ 1
            if self.leapfrog_steps:
                if i % 2 == 0:
                    k, j = j, k
            model_input = coupled_latents[j]
            base = coupled_latents[k]
            latent_model_input = torch.cat([model_input] * 2
                ) if do_classifier_free_guidance else model_input
            noise_pred = self.unet(latent_model_input, t,
                encoder_hidden_states=text_embeds).sample
            if do_classifier_free_guidance:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (
                    noise_pred_text - noise_pred_uncond)
            base, model_input = self.noise_step(base=base, model_input=
                model_input, model_output=noise_pred, timestep=t)
            coupled_latents[k] = model_input
    return coupled_latents

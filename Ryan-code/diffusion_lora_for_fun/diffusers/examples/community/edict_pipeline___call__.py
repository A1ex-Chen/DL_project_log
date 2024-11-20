@torch.no_grad()
def __call__(self, base_prompt: str, target_prompt: str, image: Image.Image,
    guidance_scale: float=3.0, num_inference_steps: int=50, strength: float
    =0.8, negative_prompt: Optional[str]=None, generator: Optional[torch.
    Generator]=None, output_type: Optional[str]='pil'):
    do_classifier_free_guidance = guidance_scale > 1.0
    image = self.image_processor.preprocess(image)
    base_embeds = self._encode_prompt(base_prompt, negative_prompt,
        do_classifier_free_guidance)
    target_embeds = self._encode_prompt(target_prompt, negative_prompt,
        do_classifier_free_guidance)
    self.scheduler.set_timesteps(num_inference_steps, self.device)
    t_limit = num_inference_steps - int(num_inference_steps * strength)
    fwd_timesteps = self.scheduler.timesteps[t_limit:]
    bwd_timesteps = fwd_timesteps.flip(0)
    coupled_latents = self.prepare_latents(image, base_embeds,
        bwd_timesteps, guidance_scale, generator)
    for i, t in tqdm(enumerate(fwd_timesteps), total=len(fwd_timesteps)):
        for k in range(2):
            j = k ^ 1
            if self.leapfrog_steps:
                if i % 2 == 1:
                    k, j = j, k
            model_input = coupled_latents[j]
            base = coupled_latents[k]
            latent_model_input = torch.cat([model_input] * 2
                ) if do_classifier_free_guidance else model_input
            noise_pred = self.unet(latent_model_input, t,
                encoder_hidden_states=target_embeds).sample
            if do_classifier_free_guidance:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (
                    noise_pred_text - noise_pred_uncond)
            base, model_input = self.denoise_step(base=base, model_input=
                model_input, model_output=noise_pred, timestep=t)
            coupled_latents[k] = model_input
        coupled_latents = self.denoise_mixing_layer(x=coupled_latents[0], y
            =coupled_latents[1])
    final_latent = coupled_latents[0]
    if output_type not in ['latent', 'pt', 'np', 'pil']:
        deprecation_message = (
            f'the output_type {output_type} is outdated. Please make sure to set it to one of these instead: `pil`, `np`, `pt`, `latent`'
            )
        deprecate('Unsupported output_type', '1.0.0', deprecation_message,
            standard_warn=False)
        output_type = 'np'
    if output_type == 'latent':
        image = final_latent
    else:
        image = self.decode_latents(final_latent)
        image = self.image_processor.postprocess(image, output_type=output_type
            )
    return image

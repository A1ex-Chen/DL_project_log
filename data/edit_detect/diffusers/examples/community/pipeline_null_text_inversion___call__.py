@torch.no_grad()
def __call__(self, prompt, uncond_embeddings, inverted_latent,
    num_inference_steps: int=50, timesteps=None, guidance_scale=7.5,
    negative_prompt=None, num_images_per_prompt=1, generator=None, latents=
    None, prompt_embeds=None, negative_prompt_embeds=None, output_type='pil'):
    self._guidance_scale = guidance_scale
    height = self.unet.config.sample_size * self.vae_scale_factor
    width = self.unet.config.sample_size * self.vae_scale_factor
    callback_steps = None
    self.check_inputs(prompt, height, width, callback_steps,
        negative_prompt, prompt_embeds, negative_prompt_embeds)
    device = self._execution_device
    prompt_embeds, _ = self.encode_prompt(prompt, device,
        num_images_per_prompt, self.do_classifier_free_guidance,
        negative_prompt, prompt_embeds=prompt_embeds,
        negative_prompt_embeds=negative_prompt_embeds)
    timesteps, num_inference_steps = retrieve_timesteps(self.scheduler,
        num_inference_steps, device, timesteps)
    latents = inverted_latent
    with self.progress_bar(total=num_inference_steps) as progress_bar:
        for i, t in enumerate(timesteps):
            noise_pred_uncond = self.unet(latents, t, encoder_hidden_states
                =uncond_embeddings[i])['sample']
            noise_pred = self.unet(latents, t, encoder_hidden_states=
                prompt_embeds)['sample']
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred -
                noise_pred_uncond)
            latents = self.scheduler.step(noise_pred, t, latents,
                return_dict=False)[0]
            progress_bar.update()
    if not output_type == 'latent':
        image = self.vae.decode(latents / self.vae.config.scaling_factor,
            return_dict=False, generator=generator)[0]
    else:
        image = latents
    image = self.image_processor.postprocess(image, output_type=output_type,
        do_denormalize=[True] * image.shape[0])
    self.maybe_free_model_hooks()
    return StableDiffusionPipelineOutput(images=image,
        nsfw_content_detected=False)

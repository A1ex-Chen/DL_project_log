@torch.no_grad()
def __call__(self, prompt: Union[str, List[str]]=None, height: Optional[int
    ]=768, width: Optional[int]=768, guidance_scale: float=7.5,
    num_images_per_prompt: Optional[int]=1, latents: Optional[torch.Tensor]
    =None, num_inference_steps: int=4, lcm_origin_steps: int=50,
    prompt_embeds: Optional[torch.Tensor]=None, output_type: Optional[str]=
    'pil', return_dict: bool=True, cross_attention_kwargs: Optional[Dict[
    str, Any]]=None):
    height = height or self.unet.config.sample_size * self.vae_scale_factor
    width = width or self.unet.config.sample_size * self.vae_scale_factor
    if prompt is not None and isinstance(prompt, str):
        batch_size = 1
    elif prompt is not None and isinstance(prompt, list):
        batch_size = len(prompt)
    else:
        batch_size = prompt_embeds.shape[0]
    device = self._execution_device
    prompt_embeds = self._encode_prompt(prompt, device,
        num_images_per_prompt, prompt_embeds=prompt_embeds)
    self.scheduler.set_timesteps(num_inference_steps, lcm_origin_steps)
    timesteps = self.scheduler.timesteps
    num_channels_latents = self.unet.config.in_channels
    latents = self.prepare_latents(batch_size * num_images_per_prompt,
        num_channels_latents, height, width, prompt_embeds.dtype, device,
        latents)
    bs = batch_size * num_images_per_prompt
    w = torch.tensor(guidance_scale).repeat(bs)
    w_embedding = self.get_w_embedding(w, embedding_dim=256).to(device=
        device, dtype=latents.dtype)
    with self.progress_bar(total=num_inference_steps) as progress_bar:
        for i, t in enumerate(timesteps):
            ts = torch.full((bs,), t, device=device, dtype=torch.long)
            latents = latents.to(prompt_embeds.dtype)
            model_pred = self.unet(latents, ts, timestep_cond=w_embedding,
                encoder_hidden_states=prompt_embeds, cross_attention_kwargs
                =cross_attention_kwargs, return_dict=False)[0]
            latents, denoised = self.scheduler.step(model_pred, i, t,
                latents, return_dict=False)
            progress_bar.update()
    denoised = denoised.to(prompt_embeds.dtype)
    if not output_type == 'latent':
        image = self.vae.decode(denoised / self.vae.config.scaling_factor,
            return_dict=False)[0]
        image, has_nsfw_concept = self.run_safety_checker(image, device,
            prompt_embeds.dtype)
    else:
        image = denoised
        has_nsfw_concept = None
    if has_nsfw_concept is None:
        do_denormalize = [True] * image.shape[0]
    else:
        do_denormalize = [(not has_nsfw) for has_nsfw in has_nsfw_concept]
    image = self.image_processor.postprocess(image, output_type=output_type,
        do_denormalize=do_denormalize)
    if not return_dict:
        return image, has_nsfw_concept
    return StableDiffusionPipelineOutput(images=image,
        nsfw_content_detected=has_nsfw_concept)

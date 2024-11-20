@torch.no_grad()
def __call__(self, prompt: Union[str, List[str]], source_prompt: Union[str,
    List[str]], image: Union[torch.FloatTensor, PIL.Image.Image]=None,
    strength: float=0.8, num_inference_steps: Optional[int]=50,
    guidance_scale: Optional[float]=7.5, source_guidance_scale: Optional[
    float]=1, num_images_per_prompt: Optional[int]=1, eta: Optional[float]=
    0.1, generator: Optional[Union[torch.Generator, List[torch.Generator]]]
    =None, prompt_embeds: Optional[torch.FloatTensor]=None, output_type:
    Optional[str]='pil', return_dict: bool=True, callback: Optional[
    Callable[[int, int, torch.FloatTensor], None]]=None, callback_steps: int=1
    ):
    """
        Function invoked when calling the pipeline for generation.

        Args:
            prompt (`str` or `List[str]`):
                The prompt or prompts to guide the image generation.
            image (`torch.FloatTensor` or `PIL.Image.Image`):
                `Image`, or tensor representing an image batch, that will be used as the starting point for the
                process.
            strength (`float`, *optional*, defaults to 0.8):
                Conceptually, indicates how much to transform the reference `image`. Must be between 0 and 1. `image`
                will be used as a starting point, adding more noise to it the larger the `strength`. The number of
                denoising steps depends on the amount of noise initially added. When `strength` is 1, added noise will
                be maximum and the denoising process will run for the full number of iterations specified in
                `num_inference_steps`. A value of 1, therefore, essentially ignores `image`.
            num_inference_steps (`int`, *optional*, defaults to 50):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference. This parameter will be modulated by `strength`.
            guidance_scale (`float`, *optional*, defaults to 7.5):
                Guidance scale as defined in [Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598).
                `guidance_scale` is defined as `w` of equation 2. of [Imagen
                Paper](https://arxiv.org/pdf/2205.11487.pdf). Guidance scale is enabled by setting `guidance_scale >
                1`. Higher guidance scale encourages to generate images that are closely linked to the text `prompt`,
                usually at the expense of lower image quality.
            source_guidance_scale (`float`, *optional*, defaults to 1):
                Guidance scale for the source prompt. This is useful to control the amount of influence the source
                prompt for encoding.
            num_images_per_prompt (`int`, *optional*, defaults to 1):
                The number of images to generate per prompt.
            eta (`float`, *optional*, defaults to 0.1):
                Corresponds to parameter eta (η) in the DDIM paper: https://arxiv.org/abs/2010.02502. Only applies to
                [`schedulers.DDIMScheduler`], will be ignored for others.
            generator (`torch.Generator`, *optional*):
                One or a list of [torch generator(s)](https://pytorch.org/docs/stable/generated/torch.Generator.html)
                to make generation deterministic.
            prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not
                provided, text embeddings will be generated from `prompt` input argument.
            negative_prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated negative text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt
                weighting. If not provided, negative_prompt_embeds will be generated from `negative_prompt` input
                argument.
            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generate image. Choose between
                [PIL](https://pillow.readthedocs.io/en/stable/): `PIL.Image.Image` or `np.array`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] instead of a
                plain tuple.
            callback (`Callable`, *optional*):
                A function that will be called every `callback_steps` steps during inference. The function will be
                called with the following arguments: `callback(step: int, timestep: int, latents: torch.FloatTensor)`.
            callback_steps (`int`, *optional*, defaults to 1):
                The frequency at which the `callback` function will be called. If not specified, the callback will be
                called at every step.

        Returns:
            [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] or `tuple`:
            [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] if `return_dict` is True, otherwise a `tuple.
            When returning a tuple, the first element is a list with the generated images, and the second element is a
            list of `bool`s denoting whether the corresponding generated image likely represents "not-safe-for-work"
            (nsfw) content, according to the `safety_checker`.
        """
    self.check_inputs(prompt, strength, callback_steps)
    batch_size = 1 if isinstance(prompt, str) else len(prompt)
    device = self._execution_device
    do_classifier_free_guidance = guidance_scale > 1.0
    prompt_embeds = self._encode_prompt(prompt, device,
        num_images_per_prompt, do_classifier_free_guidance, prompt_embeds=
        prompt_embeds)
    source_prompt_embeds = self._encode_prompt(source_prompt, device,
        num_images_per_prompt, do_classifier_free_guidance, None)
    image = preprocess(image)
    self.scheduler.set_timesteps(num_inference_steps, device=device)
    timesteps, num_inference_steps = self.get_timesteps(num_inference_steps,
        strength, device)
    latent_timestep = timesteps[:1].repeat(batch_size * num_images_per_prompt)
    latents, clean_latents = self.prepare_latents(image, latent_timestep,
        batch_size, num_images_per_prompt, prompt_embeds.dtype, device,
        generator)
    source_latents = latents
    extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)
    generator = extra_step_kwargs.pop('generator', None)
    num_warmup_steps = len(timesteps
        ) - num_inference_steps * self.scheduler.order
    with self.progress_bar(total=num_inference_steps) as progress_bar:
        for i, t in enumerate(timesteps):
            latent_model_input = torch.cat([latents] * 2)
            source_latent_model_input = torch.cat([source_latents] * 2)
            latent_model_input = self.scheduler.scale_model_input(
                latent_model_input, t)
            source_latent_model_input = self.scheduler.scale_model_input(
                source_latent_model_input, t)
            concat_latent_model_input = torch.stack([
                source_latent_model_input[0], latent_model_input[0],
                source_latent_model_input[1], latent_model_input[1]], dim=0)
            concat_prompt_embeds = torch.stack([source_prompt_embeds[0],
                prompt_embeds[0], source_prompt_embeds[1], prompt_embeds[1]
                ], dim=0)
            concat_noise_pred = self.unet(concat_latent_model_input, t,
                encoder_hidden_states=concat_prompt_embeds).sample
            (source_noise_pred_uncond, noise_pred_uncond,
                source_noise_pred_text, noise_pred_text
                ) = concat_noise_pred.chunk(4, dim=0)
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text
                 - noise_pred_uncond)
            source_noise_pred = (source_noise_pred_uncond + 
                source_guidance_scale * (source_noise_pred_text -
                source_noise_pred_uncond))
            prev_source_latents = posterior_sample(self.scheduler,
                source_latents, t, clean_latents, generator=generator, **
                extra_step_kwargs)
            noise = compute_noise(self.scheduler, prev_source_latents,
                source_latents, t, source_noise_pred, **extra_step_kwargs)
            source_latents = prev_source_latents
            latents = self.scheduler.step(noise_pred, t, latents,
                variance_noise=noise, **extra_step_kwargs).prev_sample
            if i == len(timesteps) - 1 or i + 1 > num_warmup_steps and (i + 1
                ) % self.scheduler.order == 0:
                progress_bar.update()
                if callback is not None and i % callback_steps == 0:
                    callback(i, t, latents)
    image = self.decode_latents(latents)
    image, has_nsfw_concept = self.run_safety_checker(image, device,
        prompt_embeds.dtype)
    if output_type == 'pil':
        image = self.numpy_to_pil(image)
    if not return_dict:
        return image, has_nsfw_concept
    return StableDiffusionPipelineOutput(images=image,
        nsfw_content_detected=has_nsfw_concept)

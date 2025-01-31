@torch.no_grad()
def __call__(self, prompt: Union[str, List[str]]=None, height: Optional[int
    ]=None, width: Optional[int]=None, num_inference_steps: int=50,
    guidance_scale: float=7.5, negative_prompt: Optional[Union[str, List[
    str]]]=None, num_images_per_prompt: Optional[int]=1, eta: float=0.0,
    generator: Optional[Union[torch.Generator, List[torch.Generator]]]=None,
    latents: Optional[torch.FloatTensor]=None, prompt_embeds: Optional[
    torch.FloatTensor]=None, negative_prompt_embeds: Optional[torch.
    FloatTensor]=None, output_type: Optional[str]='pil', return_dict: bool=
    True, callback: Optional[Callable[[int, int, torch.FloatTensor], None]]
    =None, callback_steps: int=1, use_karras_sigmas: Optional[bool]=False):
    """
        Function invoked when calling the pipeline for generation.

        Args:
            prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts to guide the image generation. If not defined, one has to pass `prompt_embeds`.
                instead.
            height (`int`, *optional*, defaults to self.unet.config.sample_size * self.vae_scale_factor):
                The height in pixels of the generated image.
            width (`int`, *optional*, defaults to self.unet.config.sample_size * self.vae_scale_factor):
                The width in pixels of the generated image.
            num_inference_steps (`int`, *optional*, defaults to 50):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            guidance_scale (`float`, *optional*, defaults to 7.5):
                Guidance scale as defined in [Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598).
                `guidance_scale` is defined as `w` of equation 2. of [Imagen
                Paper](https://arxiv.org/pdf/2205.11487.pdf). Guidance scale is enabled by setting `guidance_scale >
                1`. Higher guidance scale encourages to generate images that are closely linked to the text `prompt`,
                usually at the expense of lower image quality.
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the image generation. If not defined, one has to pass
                `negative_prompt_embeds`. instead. Ignored when not using guidance (i.e., ignored if `guidance_scale`
                is less than `1`).
            num_images_per_prompt (`int`, *optional*, defaults to 1):
                The number of images to generate per prompt.
            eta (`float`, *optional*, defaults to 0.0):
                Corresponds to parameter eta (η) in the DDIM paper: https://arxiv.org/abs/2010.02502. Only applies to
                [`schedulers.DDIMScheduler`], will be ignored for others.
            generator (`torch.Generator`, *optional*):
                One or a list of [torch generator(s)](https://pytorch.org/docs/stable/generated/torch.Generator.html)
                to make generation deterministic.
            latents (`torch.FloatTensor`, *optional*):
                Pre-generated noisy latents, sampled from a Gaussian distribution, to be used as inputs for image
                generation. Can be used to tweak the same generation with different prompts. If not provided, a latents
                tensor will ge generated by sampling using the supplied random `generator`.
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
            use_karras_sigmas (`bool`, *optional*, defaults to `False`):
                Use karras sigmas. For example, specifying `sample_dpmpp_2m` to `set_scheduler` will be equivalent to
                `DPM++2M` in stable-diffusion-webui. On top of that, setting this option to True will make it `DPM++2M
                Karras`.
        Returns:
            [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] or `tuple`:
            [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] if `return_dict` is True, otherwise a `tuple.
            When returning a tuple, the first element is a list with the generated images, and the second element is a
            list of `bool`s denoting whether the corresponding generated image likely represents "not-safe-for-work"
            (nsfw) content, according to the `safety_checker`.
        """
    height = height or self.unet.config.sample_size * self.vae_scale_factor
    width = width or self.unet.config.sample_size * self.vae_scale_factor
    self.check_inputs(prompt, height, width, callback_steps)
    batch_size = 1 if isinstance(prompt, str) else len(prompt)
    device = self._execution_device
    do_classifier_free_guidance = True
    if guidance_scale <= 1.0:
        raise ValueError('has to use guidance_scale')
    prompt_embeds = self._encode_prompt(prompt, device,
        num_images_per_prompt, do_classifier_free_guidance, negative_prompt,
        prompt_embeds=prompt_embeds, negative_prompt_embeds=
        negative_prompt_embeds)
    self.scheduler.set_timesteps(num_inference_steps, device=prompt_embeds.
        device)
    if use_karras_sigmas:
        sigma_min: float = self.k_diffusion_model.sigmas[0].item()
        sigma_max: float = self.k_diffusion_model.sigmas[-1].item()
        sigmas = get_sigmas_karras(n=num_inference_steps, sigma_min=
            sigma_min, sigma_max=sigma_max)
        sigmas = sigmas.to(device)
    else:
        sigmas = self.scheduler.sigmas
    sigmas = sigmas.to(prompt_embeds.dtype)
    num_channels_latents = self.unet.in_channels
    latents = self.prepare_latents(batch_size * num_images_per_prompt,
        num_channels_latents, height, width, prompt_embeds.dtype, device,
        generator, latents)
    latents = latents * sigmas[0]
    self.k_diffusion_model.sigmas = self.k_diffusion_model.sigmas.to(latents
        .device)
    self.k_diffusion_model.log_sigmas = self.k_diffusion_model.log_sigmas.to(
        latents.device)

    def model_fn(x, t):
        latent_model_input = torch.cat([x] * 2)
        t = torch.cat([t] * 2)
        noise_pred = self.k_diffusion_model(latent_model_input, t, cond=
            prompt_embeds)
        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text -
            noise_pred_uncond)
        return noise_pred
    latents = self.sampler(model_fn, latents, sigmas)
    image = self.decode_latents(latents)
    image, has_nsfw_concept = self.run_safety_checker(image, device,
        prompt_embeds.dtype)
    if output_type == 'pil':
        image = self.numpy_to_pil(image)
    if hasattr(self, 'final_offload_hook'
        ) and self.final_offload_hook is not None:
        self.final_offload_hook.offload()
    if not return_dict:
        return image, has_nsfw_concept
    return StableDiffusionPipelineOutput(images=image,
        nsfw_content_detected=has_nsfw_concept)

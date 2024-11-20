@torch.no_grad()
def __call__(self, prompt: Union[str, List[str]], height: Optional[int]=
    None, width: Optional[int]=None, num_inference_steps: int=50,
    guidance_scale: float=7.5, negative_prompt: Optional[Union[str, List[
    str]]]=None, num_images_per_prompt: Optional[int]=1, eta: float=0.0,
    generator: Optional[torch.Generator]=None, latents: Optional[torch.
    Tensor]=None, output_type: Optional[str]='pil', return_dict: bool=True,
    callback: Optional[Callable[[int, int, torch.Tensor], None]]=None,
    callback_steps: int=1, weights: Optional[str]=''):
    """
        Function invoked when calling the pipeline for generation.

        Args:
            prompt (`str` or `List[str]`):
                The prompt or prompts to guide the image generation.
            height (`int`, *optional*, defaults to self.unet.config.sample_size * self.vae_scale_factor):
                The height in pixels of the generated image.
            width (`int`, *optional*, defaults to self.unet.config.sample_size * self.vae_scale_factor):
                The width in pixels of the generated image.
            num_inference_steps (`int`, *optional*, defaults to 50):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            guidance_scale (`float`, *optional*, defaults to 5.0):
                Guidance scale as defined in [Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598).
                `guidance_scale` is defined as `w` of equation 2. of [Imagen
                Paper](https://arxiv.org/pdf/2205.11487.pdf). Guidance scale is enabled by setting `guidance_scale >
                1`. Higher guidance scale encourages to generate images that are closely linked to the text `prompt`,
                usually at the expense of lower image quality.
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the image generation. Ignored when not using guidance (i.e., ignored
                if `guidance_scale` is less than `1`).
            num_images_per_prompt (`int`, *optional*, defaults to 1):
                The number of images to generate per prompt.
            eta (`float`, *optional*, defaults to 0.0):
                Corresponds to parameter eta (η) in the DDIM paper: https://arxiv.org/abs/2010.02502. Only applies to
                [`schedulers.DDIMScheduler`], will be ignored for others.
            generator (`torch.Generator`, *optional*):
                A [torch generator](https://pytorch.org/docs/stable/generated/torch.Generator.html) to make generation
                deterministic.
            latents (`torch.Tensor`, *optional*):
                Pre-generated noisy latents, sampled from a Gaussian distribution, to be used as inputs for image
                generation. Can be used to tweak the same generation with different prompts. If not provided, a latents
                tensor will ge generated by sampling using the supplied random `generator`.
            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generate image. Choose between
                [PIL](https://pillow.readthedocs.io/en/stable/): `PIL.Image.Image` or `np.array`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] instead of a
                plain tuple.
            callback (`Callable`, *optional*):
                A function that will be called every `callback_steps` steps during inference. The function will be
                called with the following arguments: `callback(step: int, timestep: int, latents: torch.Tensor)`.
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
    height = height or self.unet.config.sample_size * self.vae_scale_factor
    width = width or self.unet.config.sample_size * self.vae_scale_factor
    self.check_inputs(prompt, height, width, callback_steps)
    batch_size = 1 if isinstance(prompt, str) else len(prompt)
    device = self._execution_device
    do_classifier_free_guidance = guidance_scale > 1.0
    if '|' in prompt:
        prompt = [x.strip() for x in prompt.split('|')]
        print(f'composing {prompt}...')
        if not weights:
            print(
                'using equal positive weights (conjunction) for all prompts...'
                )
            weights = torch.tensor([guidance_scale] * len(prompt), device=
                self.device).reshape(-1, 1, 1, 1)
        else:
            num_prompts = len(prompt) if isinstance(prompt, list) else 1
            weights = [float(w.strip()) for w in weights.split('|')]
            if len(weights) < num_prompts:
                weights.append(guidance_scale)
            else:
                weights = weights[:num_prompts]
            assert len(weights) == len(prompt
                ), 'weights specified are not equal to the number of prompts'
            weights = torch.tensor(weights, device=self.device).reshape(-1,
                1, 1, 1)
    else:
        weights = guidance_scale
    text_embeddings = self._encode_prompt(prompt, device,
        num_images_per_prompt, do_classifier_free_guidance, negative_prompt)
    self.scheduler.set_timesteps(num_inference_steps, device=device)
    timesteps = self.scheduler.timesteps
    num_channels_latents = self.unet.config.in_channels
    latents = self.prepare_latents(batch_size * num_images_per_prompt,
        num_channels_latents, height, width, text_embeddings.dtype, device,
        generator, latents)
    if isinstance(prompt, list) and batch_size == 1:
        text_embeddings = text_embeddings[len(prompt) - 1:]
    extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)
    num_warmup_steps = len(timesteps
        ) - num_inference_steps * self.scheduler.order
    with self.progress_bar(total=num_inference_steps) as progress_bar:
        for i, t in enumerate(timesteps):
            latent_model_input = torch.cat([latents] * 2
                ) if do_classifier_free_guidance else latents
            latent_model_input = self.scheduler.scale_model_input(
                latent_model_input, t)
            noise_pred = []
            for j in range(text_embeddings.shape[0]):
                noise_pred.append(self.unet(latent_model_input[:1], t,
                    encoder_hidden_states=text_embeddings[j:j + 1]).sample)
            noise_pred = torch.cat(noise_pred, dim=0)
            if do_classifier_free_guidance:
                noise_pred_uncond, noise_pred_text = noise_pred[:1
                    ], noise_pred[1:]
                noise_pred = noise_pred_uncond + (weights * (
                    noise_pred_text - noise_pred_uncond)).sum(dim=0,
                    keepdims=True)
            latents = self.scheduler.step(noise_pred, t, latents, **
                extra_step_kwargs).prev_sample
            if i == len(timesteps) - 1 or i + 1 > num_warmup_steps and (i + 1
                ) % self.scheduler.order == 0:
                progress_bar.update()
                if callback is not None and i % callback_steps == 0:
                    step_idx = i // getattr(self.scheduler, 'order', 1)
                    callback(step_idx, t, latents)
    image = self.decode_latents(latents)
    image, has_nsfw_concept = self.run_safety_checker(image, device,
        text_embeddings.dtype)
    if output_type == 'pil':
        image = self.numpy_to_pil(image)
    if not return_dict:
        return image, has_nsfw_concept
    return StableDiffusionPipelineOutput(images=image,
        nsfw_content_detected=has_nsfw_concept)

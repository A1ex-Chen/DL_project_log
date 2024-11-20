@torch.no_grad()
@replace_example_docstring(EXAMPLE_DOC_STRING)
def __call__(self, prompt: Union[str, List[str]]=None, image: Union[torch.
    Tensor, PIL.Image.Image]=None, controlnet_conditioning_image: Union[
    torch.Tensor, PIL.Image.Image, List[torch.Tensor], List[PIL.Image.Image
    ]]=None, strength: float=0.8, height: Optional[int]=None, width:
    Optional[int]=None, num_inference_steps: int=50, guidance_scale: float=
    7.5, negative_prompt: Optional[Union[str, List[str]]]=None,
    num_images_per_prompt: Optional[int]=1, eta: float=0.0, generator:
    Optional[Union[torch.Generator, List[torch.Generator]]]=None, latents:
    Optional[torch.Tensor]=None, prompt_embeds: Optional[torch.Tensor]=None,
    negative_prompt_embeds: Optional[torch.Tensor]=None, output_type:
    Optional[str]='pil', return_dict: bool=True, callback: Optional[
    Callable[[int, int, torch.Tensor], None]]=None, callback_steps: int=1,
    cross_attention_kwargs: Optional[Dict[str, Any]]=None,
    controlnet_conditioning_scale: Union[float, List[float]]=1.0,
    controlnet_guidance_start: float=0.0, controlnet_guidance_end: float=1.0):
    """
        Function invoked when calling the pipeline for generation.

        Args:
            prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts to guide the image generation. If not defined, one has to pass `prompt_embeds`.
                instead.
            image (`torch.Tensor` or `PIL.Image.Image`):
                `Image`, or tensor representing an image batch which will be inpainted, *i.e.* parts of the image will
                be masked out with `mask_image` and repainted according to `prompt`.
            controlnet_conditioning_image (`torch.Tensor`, `PIL.Image.Image`, `List[torch.Tensor]` or `List[PIL.Image.Image]`):
                The ControlNet input condition. ControlNet uses this input condition to generate guidance to Unet. If
                the type is specified as `torch.Tensor`, it is passed to ControlNet as is. PIL.Image.Image` can
                also be accepted as an image. The control image is automatically resized to fit the output image.
            strength (`float`, *optional*):
                Conceptually, indicates how much to transform the reference `image`. Must be between 0 and 1. `image`
                will be used as a starting point, adding more noise to it the larger the `strength`. The number of
                denoising steps depends on the amount of noise initially added. When `strength` is 1, added noise will
                be maximum and the denoising process will run for the full number of iterations specified in
                `num_inference_steps`. A value of 1, therefore, essentially ignores `image`.
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
                `negative_prompt_embeds` instead. Ignored when not using guidance (i.e., ignored if `guidance_scale` is less than `1`).
            num_images_per_prompt (`int`, *optional*, defaults to 1):
                The number of images to generate per prompt.
            eta (`float`, *optional*, defaults to 0.0):
                Corresponds to parameter eta (η) in the DDIM paper: https://arxiv.org/abs/2010.02502. Only applies to
                [`schedulers.DDIMScheduler`], will be ignored for others.
            generator (`torch.Generator` or `List[torch.Generator]`, *optional*):
                One or a list of [torch generator(s)](https://pytorch.org/docs/stable/generated/torch.Generator.html)
                to make generation deterministic.
            latents (`torch.Tensor`, *optional*):
                Pre-generated noisy latents, sampled from a Gaussian distribution, to be used as inputs for image
                generation. Can be used to tweak the same generation with different prompts. If not provided, a latents
                tensor will ge generated by sampling using the supplied random `generator`.
            prompt_embeds (`torch.Tensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not
                provided, text embeddings will be generated from `prompt` input argument.
            negative_prompt_embeds (`torch.Tensor`, *optional*):
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
                called with the following arguments: `callback(step: int, timestep: int, latents: torch.Tensor)`.
            callback_steps (`int`, *optional*, defaults to 1):
                The frequency at which the `callback` function will be called. If not specified, the callback will be
                called at every step.
            cross_attention_kwargs (`dict`, *optional*):
                A kwargs dictionary that if specified is passed along to the `AttentionProcessor` as defined under
                `self.processor` in
                [diffusers.models.attention_processor](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py).
            controlnet_conditioning_scale (`float`, *optional*, defaults to 1.0):
                The outputs of the controlnet are multiplied by `controlnet_conditioning_scale` before they are added
                to the residual in the original unet.
            controlnet_guidance_start ('float', *optional*, defaults to 0.0):
                The percentage of total steps the controlnet starts applying. Must be between 0 and 1.
            controlnet_guidance_end ('float', *optional*, defaults to 1.0):
                The percentage of total steps the controlnet ends applying. Must be between 0 and 1. Must be greater
                than `controlnet_guidance_start`.

        Examples:

        Returns:
            [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] or `tuple`:
            [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] if `return_dict` is True, otherwise a `tuple.
            When returning a tuple, the first element is a list with the generated images, and the second element is a
            list of `bool`s denoting whether the corresponding generated image likely represents "not-safe-for-work"
            (nsfw) content, according to the `safety_checker`.
        """
    height, width = self._default_height_width(height, width,
        controlnet_conditioning_image)
    self.check_inputs(prompt, image, controlnet_conditioning_image, height,
        width, callback_steps, negative_prompt, prompt_embeds,
        negative_prompt_embeds, strength, controlnet_guidance_start,
        controlnet_guidance_end, controlnet_conditioning_scale)
    if prompt is not None and isinstance(prompt, str):
        batch_size = 1
    elif prompt is not None and isinstance(prompt, list):
        batch_size = len(prompt)
    else:
        batch_size = prompt_embeds.shape[0]
    device = self._execution_device
    do_classifier_free_guidance = guidance_scale > 1.0
    if isinstance(self.controlnet, MultiControlNetModel) and isinstance(
        controlnet_conditioning_scale, float):
        controlnet_conditioning_scale = [controlnet_conditioning_scale] * len(
            self.controlnet.nets)
    prompt_embeds = self._encode_prompt(prompt, device,
        num_images_per_prompt, do_classifier_free_guidance, negative_prompt,
        prompt_embeds=prompt_embeds, negative_prompt_embeds=
        negative_prompt_embeds)
    image = prepare_image(image)
    if isinstance(self.controlnet, ControlNetModel):
        controlnet_conditioning_image = prepare_controlnet_conditioning_image(
            controlnet_conditioning_image=controlnet_conditioning_image,
            width=width, height=height, batch_size=batch_size *
            num_images_per_prompt, num_images_per_prompt=
            num_images_per_prompt, device=device, dtype=self.controlnet.
            dtype, do_classifier_free_guidance=do_classifier_free_guidance)
    elif isinstance(self.controlnet, MultiControlNetModel):
        controlnet_conditioning_images = []
        for image_ in controlnet_conditioning_image:
            image_ = prepare_controlnet_conditioning_image(
                controlnet_conditioning_image=image_, width=width, height=
                height, batch_size=batch_size * num_images_per_prompt,
                num_images_per_prompt=num_images_per_prompt, device=device,
                dtype=self.controlnet.dtype, do_classifier_free_guidance=
                do_classifier_free_guidance)
            controlnet_conditioning_images.append(image_)
        controlnet_conditioning_image = controlnet_conditioning_images
    else:
        assert False
    self.scheduler.set_timesteps(num_inference_steps, device=device)
    timesteps, num_inference_steps = self.get_timesteps(num_inference_steps,
        strength, device)
    latent_timestep = timesteps[:1].repeat(batch_size * num_images_per_prompt)
    if latents is None:
        latents = self.prepare_latents(image, latent_timestep, batch_size,
            num_images_per_prompt, prompt_embeds.dtype, device, generator)
    extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)
    num_warmup_steps = len(timesteps
        ) - num_inference_steps * self.scheduler.order
    with self.progress_bar(total=num_inference_steps) as progress_bar:
        for i, t in enumerate(timesteps):
            latent_model_input = torch.cat([latents] * 2
                ) if do_classifier_free_guidance else latents
            latent_model_input = self.scheduler.scale_model_input(
                latent_model_input, t)
            current_sampling_percent = i / len(timesteps)
            if (current_sampling_percent < controlnet_guidance_start or 
                current_sampling_percent > controlnet_guidance_end):
                down_block_res_samples = None
                mid_block_res_sample = None
            else:
                down_block_res_samples, mid_block_res_sample = self.controlnet(
                    latent_model_input, t, encoder_hidden_states=
                    prompt_embeds, controlnet_cond=
                    controlnet_conditioning_image, conditioning_scale=
                    controlnet_conditioning_scale, return_dict=False)
            noise_pred = self.unet(latent_model_input, t,
                encoder_hidden_states=prompt_embeds, cross_attention_kwargs
                =cross_attention_kwargs, down_block_additional_residuals=
                down_block_res_samples, mid_block_additional_residual=
                mid_block_res_sample).sample
            if do_classifier_free_guidance:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (
                    noise_pred_text - noise_pred_uncond)
            latents = self.scheduler.step(noise_pred, t, latents, **
                extra_step_kwargs).prev_sample
            if i == len(timesteps) - 1 or i + 1 > num_warmup_steps and (i + 1
                ) % self.scheduler.order == 0:
                progress_bar.update()
                if callback is not None and i % callback_steps == 0:
                    step_idx = i // getattr(self.scheduler, 'order', 1)
                    callback(step_idx, t, latents)
    if hasattr(self, 'final_offload_hook'
        ) and self.final_offload_hook is not None:
        self.unet.to('cpu')
        self.controlnet.to('cpu')
        torch.cuda.empty_cache()
    if output_type == 'latent':
        image = latents
        has_nsfw_concept = None
    elif output_type == 'pil':
        image = self.decode_latents(latents)
        image, has_nsfw_concept = self.run_safety_checker(image, device,
            prompt_embeds.dtype)
        image = self.numpy_to_pil(image)
    else:
        image = self.decode_latents(latents)
        image, has_nsfw_concept = self.run_safety_checker(image, device,
            prompt_embeds.dtype)
    if hasattr(self, 'final_offload_hook'
        ) and self.final_offload_hook is not None:
        self.final_offload_hook.offload()
    if not return_dict:
        return image, has_nsfw_concept
    return StableDiffusionPipelineOutput(images=image,
        nsfw_content_detected=has_nsfw_concept)
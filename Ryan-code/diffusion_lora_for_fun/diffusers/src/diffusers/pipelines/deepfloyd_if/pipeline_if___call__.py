@torch.no_grad()
@replace_example_docstring(EXAMPLE_DOC_STRING)
def __call__(self, prompt: Union[str, List[str]]=None, num_inference_steps:
    int=100, timesteps: List[int]=None, guidance_scale: float=7.0,
    negative_prompt: Optional[Union[str, List[str]]]=None,
    num_images_per_prompt: Optional[int]=1, height: Optional[int]=None,
    width: Optional[int]=None, eta: float=0.0, generator: Optional[Union[
    torch.Generator, List[torch.Generator]]]=None, prompt_embeds: Optional[
    torch.Tensor]=None, negative_prompt_embeds: Optional[torch.Tensor]=None,
    output_type: Optional[str]='pil', return_dict: bool=True, callback:
    Optional[Callable[[int, int, torch.Tensor], None]]=None, callback_steps:
    int=1, clean_caption: bool=True, cross_attention_kwargs: Optional[Dict[
    str, Any]]=None):
    """
        Function invoked when calling the pipeline for generation.

        Args:
            prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts to guide the image generation. If not defined, one has to pass `prompt_embeds`.
                instead.
            num_inference_steps (`int`, *optional*, defaults to 100):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            timesteps (`List[int]`, *optional*):
                Custom timesteps to use for the denoising process. If not defined, equal spaced `num_inference_steps`
                timesteps are used. Must be in descending order.
            guidance_scale (`float`, *optional*, defaults to 7.0):
                Guidance scale as defined in [Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598).
                `guidance_scale` is defined as `w` of equation 2. of [Imagen
                Paper](https://arxiv.org/pdf/2205.11487.pdf). Guidance scale is enabled by setting `guidance_scale >
                1`. Higher guidance scale encourages to generate images that are closely linked to the text `prompt`,
                usually at the expense of lower image quality.
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the image generation. If not defined, one has to pass
                `negative_prompt_embeds` instead. Ignored when not using guidance (i.e., ignored if `guidance_scale` is
                less than `1`).
            num_images_per_prompt (`int`, *optional*, defaults to 1):
                The number of images to generate per prompt.
            height (`int`, *optional*, defaults to self.unet.config.sample_size):
                The height in pixels of the generated image.
            width (`int`, *optional*, defaults to self.unet.config.sample_size):
                The width in pixels of the generated image.
            eta (`float`, *optional*, defaults to 0.0):
                Corresponds to parameter eta (η) in the DDIM paper: https://arxiv.org/abs/2010.02502. Only applies to
                [`schedulers.DDIMScheduler`], will be ignored for others.
            generator (`torch.Generator` or `List[torch.Generator]`, *optional*):
                One or a list of [torch generator(s)](https://pytorch.org/docs/stable/generated/torch.Generator.html)
                to make generation deterministic.
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
                Whether or not to return a [`~pipelines.stable_diffusion.IFPipelineOutput`] instead of a plain tuple.
            callback (`Callable`, *optional*):
                A function that will be called every `callback_steps` steps during inference. The function will be
                called with the following arguments: `callback(step: int, timestep: int, latents: torch.Tensor)`.
            callback_steps (`int`, *optional*, defaults to 1):
                The frequency at which the `callback` function will be called. If not specified, the callback will be
                called at every step.
            clean_caption (`bool`, *optional*, defaults to `True`):
                Whether or not to clean the caption before creating embeddings. Requires `beautifulsoup4` and `ftfy` to
                be installed. If the dependencies are not installed, the embeddings will be created from the raw
                prompt.
            cross_attention_kwargs (`dict`, *optional*):
                A kwargs dictionary that if specified is passed along to the `AttentionProcessor` as defined under
                `self.processor` in
                [diffusers.models.attention_processor](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py).

        Examples:

        Returns:
            [`~pipelines.stable_diffusion.IFPipelineOutput`] or `tuple`:
            [`~pipelines.stable_diffusion.IFPipelineOutput`] if `return_dict` is True, otherwise a `tuple. When
            returning a tuple, the first element is a list with the generated images, and the second element is a list
            of `bool`s denoting whether the corresponding generated image likely represents "not-safe-for-work" (nsfw)
            or watermarked content, according to the `safety_checker`.
        """
    self.check_inputs(prompt, callback_steps, negative_prompt,
        prompt_embeds, negative_prompt_embeds)
    height = height or self.unet.config.sample_size
    width = width or self.unet.config.sample_size
    if prompt is not None and isinstance(prompt, str):
        batch_size = 1
    elif prompt is not None and isinstance(prompt, list):
        batch_size = len(prompt)
    else:
        batch_size = prompt_embeds.shape[0]
    device = self._execution_device
    do_classifier_free_guidance = guidance_scale > 1.0
    prompt_embeds, negative_prompt_embeds = self.encode_prompt(prompt,
        do_classifier_free_guidance, num_images_per_prompt=
        num_images_per_prompt, device=device, negative_prompt=
        negative_prompt, prompt_embeds=prompt_embeds,
        negative_prompt_embeds=negative_prompt_embeds, clean_caption=
        clean_caption)
    if do_classifier_free_guidance:
        prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])
    if timesteps is not None:
        self.scheduler.set_timesteps(timesteps=timesteps, device=device)
        timesteps = self.scheduler.timesteps
        num_inference_steps = len(timesteps)
    else:
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps
    if hasattr(self.scheduler, 'set_begin_index'):
        self.scheduler.set_begin_index(0)
    intermediate_images = self.prepare_intermediate_images(batch_size *
        num_images_per_prompt, self.unet.config.in_channels, height, width,
        prompt_embeds.dtype, device, generator)
    extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)
    if hasattr(self, 'text_encoder_offload_hook'
        ) and self.text_encoder_offload_hook is not None:
        self.text_encoder_offload_hook.offload()
    num_warmup_steps = len(timesteps
        ) - num_inference_steps * self.scheduler.order
    with self.progress_bar(total=num_inference_steps) as progress_bar:
        for i, t in enumerate(timesteps):
            model_input = torch.cat([intermediate_images] * 2
                ) if do_classifier_free_guidance else intermediate_images
            model_input = self.scheduler.scale_model_input(model_input, t)
            noise_pred = self.unet(model_input, t, encoder_hidden_states=
                prompt_embeds, cross_attention_kwargs=
                cross_attention_kwargs, return_dict=False)[0]
            if do_classifier_free_guidance:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred_uncond, _ = noise_pred_uncond.split(model_input.
                    shape[1], dim=1)
                noise_pred_text, predicted_variance = noise_pred_text.split(
                    model_input.shape[1], dim=1)
                noise_pred = noise_pred_uncond + guidance_scale * (
                    noise_pred_text - noise_pred_uncond)
                noise_pred = torch.cat([noise_pred, predicted_variance], dim=1)
            if self.scheduler.config.variance_type not in ['learned',
                'learned_range']:
                noise_pred, _ = noise_pred.split(model_input.shape[1], dim=1)
            intermediate_images = self.scheduler.step(noise_pred, t,
                intermediate_images, **extra_step_kwargs, return_dict=False)[0]
            if i == len(timesteps) - 1 or i + 1 > num_warmup_steps and (i + 1
                ) % self.scheduler.order == 0:
                progress_bar.update()
                if callback is not None and i % callback_steps == 0:
                    callback(i, t, intermediate_images)
    image = intermediate_images
    if output_type == 'pil':
        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.cpu().permute(0, 2, 3, 1).float().numpy()
        image, nsfw_detected, watermark_detected = self.run_safety_checker(
            image, device, prompt_embeds.dtype)
        image = self.numpy_to_pil(image)
        if self.watermarker is not None:
            image = self.watermarker.apply_watermark(image, self.unet.
                config.sample_size)
    elif output_type == 'pt':
        nsfw_detected = None
        watermark_detected = None
        if hasattr(self, 'unet_offload_hook'
            ) and self.unet_offload_hook is not None:
            self.unet_offload_hook.offload()
    else:
        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.cpu().permute(0, 2, 3, 1).float().numpy()
        image, nsfw_detected, watermark_detected = self.run_safety_checker(
            image, device, prompt_embeds.dtype)
    self.maybe_free_model_hooks()
    if not return_dict:
        return image, nsfw_detected, watermark_detected
    return IFPipelineOutput(images=image, nsfw_detected=nsfw_detected,
        watermark_detected=watermark_detected)

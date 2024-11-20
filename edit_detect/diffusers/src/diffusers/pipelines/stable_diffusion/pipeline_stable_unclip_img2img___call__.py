@torch.no_grad()
@replace_example_docstring(EXAMPLE_DOC_STRING)
def __call__(self, image: Union[torch.Tensor, PIL.Image.Image]=None, prompt:
    Union[str, List[str]]=None, height: Optional[int]=None, width: Optional
    [int]=None, num_inference_steps: int=20, guidance_scale: float=10,
    negative_prompt: Optional[Union[str, List[str]]]=None,
    num_images_per_prompt: Optional[int]=1, eta: float=0.0, generator:
    Optional[torch.Generator]=None, latents: Optional[torch.Tensor]=None,
    prompt_embeds: Optional[torch.Tensor]=None, negative_prompt_embeds:
    Optional[torch.Tensor]=None, output_type: Optional[str]='pil',
    return_dict: bool=True, callback: Optional[Callable[[int, int, torch.
    Tensor], None]]=None, callback_steps: int=1, cross_attention_kwargs:
    Optional[Dict[str, Any]]=None, noise_level: int=0, image_embeds:
    Optional[torch.Tensor]=None, clip_skip: Optional[int]=None):
    """
        The call function to the pipeline for generation.

        Args:
            prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts to guide the image generation. If not defined, either `prompt_embeds` will be
                used or prompt is initialized to `""`.
            image (`torch.Tensor` or `PIL.Image.Image`):
                `Image` or tensor representing an image batch. The image is encoded to its CLIP embedding which the
                `unet` is conditioned on. The image is _not_ encoded by the `vae` and then used as the latents in the
                denoising process like it is in the standard Stable Diffusion text-guided image variation process.
            height (`int`, *optional*, defaults to `self.unet.config.sample_size * self.vae_scale_factor`):
                The height in pixels of the generated image.
            width (`int`, *optional*, defaults to `self.unet.config.sample_size * self.vae_scale_factor`):
                The width in pixels of the generated image.
            num_inference_steps (`int`, *optional*, defaults to 20):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            guidance_scale (`float`, *optional*, defaults to 10.0):
                A higher guidance scale value encourages the model to generate images closely linked to the text
                `prompt` at the expense of lower image quality. Guidance scale is enabled when `guidance_scale > 1`.
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts to guide what to not include in image generation. If not defined, you need to
                pass `negative_prompt_embeds` instead. Ignored when not using guidance (`guidance_scale < 1`).
            num_images_per_prompt (`int`, *optional*, defaults to 1):
                The number of images to generate per prompt.
            eta (`float`, *optional*, defaults to 0.0):
                Corresponds to parameter eta (η) from the [DDIM](https://arxiv.org/abs/2010.02502) paper. Only applies
                to the [`~schedulers.DDIMScheduler`], and is ignored in other schedulers.
            generator (`torch.Generator` or `List[torch.Generator]`, *optional*):
                A [`torch.Generator`](https://pytorch.org/docs/stable/generated/torch.Generator.html) to make
                generation deterministic.
            latents (`torch.Tensor`, *optional*):
                Pre-generated noisy latents sampled from a Gaussian distribution, to be used as inputs for image
                generation. Can be used to tweak the same generation with different prompts. If not provided, a latents
                tensor is generated by sampling using the supplied random `generator`.
            prompt_embeds (`torch.Tensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs (prompt weighting). If not
                provided, text embeddings are generated from the `prompt` input argument.
            negative_prompt_embeds (`torch.Tensor`, *optional*):
                Pre-generated negative text embeddings. Can be used to easily tweak text inputs (prompt weighting). If
                not provided, `negative_prompt_embeds` are generated from the `negative_prompt` input argument.
            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generated image. Choose between `PIL.Image` or `np.array`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~pipelines.ImagePipelineOutput`] instead of a plain tuple.
            callback (`Callable`, *optional*):
                A function that calls every `callback_steps` steps during inference. The function is called with the
                following arguments: `callback(step: int, timestep: int, latents: torch.Tensor)`.
            callback_steps (`int`, *optional*, defaults to 1):
                The frequency at which the `callback` function is called. If not specified, the callback is called at
                every step.
            cross_attention_kwargs (`dict`, *optional*):
                A kwargs dictionary that if specified is passed along to the [`AttentionProcessor`] as defined in
                [`self.processor`](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py).
            noise_level (`int`, *optional*, defaults to `0`):
                The amount of noise to add to the image embeddings. A higher `noise_level` increases the variance in
                the final un-noised images. See [`StableUnCLIPPipeline.noise_image_embeddings`] for more details.
            image_embeds (`torch.Tensor`, *optional*):
                Pre-generated CLIP embeddings to condition the `unet` on. These latents are not used in the denoising
                process. If you want to provide pre-generated latents, pass them to `__call__` as `latents`.
            clip_skip (`int`, *optional*):
                Number of layers to be skipped from CLIP while computing the prompt embeddings. A value of 1 means that
                the output of the pre-final layer will be used for computing the prompt embeddings.

        Examples:

        Returns:
            [`~pipelines.ImagePipelineOutput`] or `tuple`:
                [`~ pipeline_utils.ImagePipelineOutput`] if `return_dict` is True, otherwise a `tuple`. When returning
                a tuple, the first element is a list with the generated images.
        """
    height = height or self.unet.config.sample_size * self.vae_scale_factor
    width = width or self.unet.config.sample_size * self.vae_scale_factor
    if prompt is None and prompt_embeds is None:
        prompt = len(image) * [''] if isinstance(image, list) else ''
    self.check_inputs(prompt=prompt, image=image, height=height, width=
        width, callback_steps=callback_steps, noise_level=noise_level,
        negative_prompt=negative_prompt, prompt_embeds=prompt_embeds,
        negative_prompt_embeds=negative_prompt_embeds, image_embeds=
        image_embeds)
    if prompt is not None and isinstance(prompt, str):
        batch_size = 1
    elif prompt is not None and isinstance(prompt, list):
        batch_size = len(prompt)
    else:
        batch_size = prompt_embeds.shape[0]
    batch_size = batch_size * num_images_per_prompt
    device = self._execution_device
    do_classifier_free_guidance = guidance_scale > 1.0
    text_encoder_lora_scale = cross_attention_kwargs.get('scale', None
        ) if cross_attention_kwargs is not None else None
    prompt_embeds, negative_prompt_embeds = self.encode_prompt(prompt=
        prompt, device=device, num_images_per_prompt=num_images_per_prompt,
        do_classifier_free_guidance=do_classifier_free_guidance,
        negative_prompt=negative_prompt, prompt_embeds=prompt_embeds,
        negative_prompt_embeds=negative_prompt_embeds, lora_scale=
        text_encoder_lora_scale)
    if do_classifier_free_guidance:
        prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])
    noise_level = torch.tensor([noise_level], device=device)
    image_embeds = self._encode_image(image=image, device=device,
        batch_size=batch_size, num_images_per_prompt=num_images_per_prompt,
        do_classifier_free_guidance=do_classifier_free_guidance,
        noise_level=noise_level, generator=generator, image_embeds=image_embeds
        )
    self.scheduler.set_timesteps(num_inference_steps, device=device)
    timesteps = self.scheduler.timesteps
    num_channels_latents = self.unet.config.in_channels
    if latents is None:
        latents = self.prepare_latents(batch_size=batch_size,
            num_channels_latents=num_channels_latents, height=height, width
            =width, dtype=prompt_embeds.dtype, device=device, generator=
            generator, latents=latents)
    extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)
    for i, t in enumerate(self.progress_bar(timesteps)):
        latent_model_input = torch.cat([latents] * 2
            ) if do_classifier_free_guidance else latents
        latent_model_input = self.scheduler.scale_model_input(
            latent_model_input, t)
        noise_pred = self.unet(latent_model_input, t, encoder_hidden_states
            =prompt_embeds, class_labels=image_embeds,
            cross_attention_kwargs=cross_attention_kwargs, return_dict=False)[0
            ]
        if do_classifier_free_guidance:
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text
                 - noise_pred_uncond)
        latents = self.scheduler.step(noise_pred, t, latents, **
            extra_step_kwargs, return_dict=False)[0]
        if callback is not None and i % callback_steps == 0:
            step_idx = i // getattr(self.scheduler, 'order', 1)
            callback(step_idx, t, latents)
    if not output_type == 'latent':
        image = self.vae.decode(latents / self.vae.config.scaling_factor,
            return_dict=False)[0]
    else:
        image = latents
    image = self.image_processor.postprocess(image, output_type=output_type)
    self.maybe_free_model_hooks()
    if not return_dict:
        return image,
    return ImagePipelineOutput(images=image)
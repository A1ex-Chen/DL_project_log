@torch.no_grad()
@replace_example_docstring(EXAMPLE_DOC_STRING)
def __call__(self, prompt: Union[str, List[str]]=None, height: Optional[int
    ]=None, width: Optional[int]=None, num_inference_steps: int=50,
    guidance_scale: float=7.5, gligen_scheduled_sampling_beta: float=0.3,
    gligen_phrases: List[str]=None, gligen_images: List[PIL.Image.Image]=
    None, input_phrases_mask: Union[int, List[int]]=None, input_images_mask:
    Union[int, List[int]]=None, gligen_boxes: List[List[float]]=None,
    gligen_inpaint_image: Optional[PIL.Image.Image]=None, negative_prompt:
    Optional[Union[str, List[str]]]=None, num_images_per_prompt: Optional[
    int]=1, eta: float=0.0, generator: Optional[Union[torch.Generator, List
    [torch.Generator]]]=None, latents: Optional[torch.Tensor]=None,
    prompt_embeds: Optional[torch.Tensor]=None, negative_prompt_embeds:
    Optional[torch.Tensor]=None, output_type: Optional[str]='pil',
    return_dict: bool=True, callback: Optional[Callable[[int, int, torch.
    Tensor], None]]=None, callback_steps: int=1, cross_attention_kwargs:
    Optional[Dict[str, Any]]=None, gligen_normalize_constant: float=28.7,
    clip_skip: int=None):
    """
        The call function to the pipeline for generation.

        Args:
            prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts to guide image generation. If not defined, you need to pass `prompt_embeds`.
            height (`int`, *optional*, defaults to `self.unet.config.sample_size * self.vae_scale_factor`):
                The height in pixels of the generated image.
            width (`int`, *optional*, defaults to `self.unet.config.sample_size * self.vae_scale_factor`):
                The width in pixels of the generated image.
            num_inference_steps (`int`, *optional*, defaults to 50):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            guidance_scale (`float`, *optional*, defaults to 7.5):
                A higher guidance scale value encourages the model to generate images closely linked to the text
                `prompt` at the expense of lower image quality. Guidance scale is enabled when `guidance_scale > 1`.
            gligen_phrases (`List[str]`):
                The phrases to guide what to include in each of the regions defined by the corresponding
                `gligen_boxes`. There should only be one phrase per bounding box.
            gligen_images (`List[PIL.Image.Image]`):
                The images to guide what to include in each of the regions defined by the corresponding `gligen_boxes`.
                There should only be one image per bounding box
            input_phrases_mask (`int` or `List[int]`):
                pre phrases mask input defined by the correspongding `input_phrases_mask`
            input_images_mask (`int` or `List[int]`):
                pre images mask input defined by the correspongding `input_images_mask`
            gligen_boxes (`List[List[float]]`):
                The bounding boxes that identify rectangular regions of the image that are going to be filled with the
                content described by the corresponding `gligen_phrases`. Each rectangular box is defined as a
                `List[float]` of 4 elements `[xmin, ymin, xmax, ymax]` where each value is between [0,1].
            gligen_inpaint_image (`PIL.Image.Image`, *optional*):
                The input image, if provided, is inpainted with objects described by the `gligen_boxes` and
                `gligen_phrases`. Otherwise, it is treated as a generation task on a blank input image.
            gligen_scheduled_sampling_beta (`float`, defaults to 0.3):
                Scheduled Sampling factor from [GLIGEN: Open-Set Grounded Text-to-Image
                Generation](https://arxiv.org/pdf/2301.07093.pdf). Scheduled Sampling factor is only varied for
                scheduled sampling during inference for improved quality and controllability.
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
                Whether or not to return a [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] instead of a
                plain tuple.
            callback (`Callable`, *optional*):
                A function that calls every `callback_steps` steps during inference. The function is called with the
                following arguments: `callback(step: int, timestep: int, latents: torch.Tensor)`.
            callback_steps (`int`, *optional*, defaults to 1):
                The frequency at which the `callback` function is called. If not specified, the callback is called at
                every step.
            cross_attention_kwargs (`dict`, *optional*):
                A kwargs dictionary that if specified is passed along to the [`AttentionProcessor`] as defined in
                [`self.processor`](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py).
            gligen_normalize_constant (`float`, *optional*, defaults to 28.7):
                The normalize value of the image embedding.
            clip_skip (`int`, *optional*):
                Number of layers to be skipped from CLIP while computing the prompt embeddings. A value of 1 means that
                the output of the pre-final layer will be used for computing the prompt embeddings.

        Examples:

        Returns:
            [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] or `tuple`:
                If `return_dict` is `True`, [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] is returned,
                otherwise a `tuple` is returned where the first element is a list with the generated images and the
                second element is a list of `bool`s indicating whether the corresponding generated image contains
                "not-safe-for-work" (nsfw) content.
        """
    height = height or self.unet.config.sample_size * self.vae_scale_factor
    width = width or self.unet.config.sample_size * self.vae_scale_factor
    self.check_inputs(prompt, height, width, callback_steps,
        negative_prompt, prompt_embeds, negative_prompt_embeds)
    if prompt is not None and isinstance(prompt, str):
        batch_size = 1
    elif prompt is not None and isinstance(prompt, list):
        batch_size = len(prompt)
    else:
        batch_size = prompt_embeds.shape[0]
    device = self._execution_device
    do_classifier_free_guidance = guidance_scale > 1.0
    prompt_embeds, negative_prompt_embeds = self.encode_prompt(prompt,
        device, num_images_per_prompt, do_classifier_free_guidance,
        negative_prompt, prompt_embeds=prompt_embeds,
        negative_prompt_embeds=negative_prompt_embeds, clip_skip=clip_skip)
    if do_classifier_free_guidance:
        prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])
    self.scheduler.set_timesteps(num_inference_steps, device=device)
    timesteps = self.scheduler.timesteps
    num_channels_latents = self.unet.config.in_channels
    latents = self.prepare_latents(batch_size * num_images_per_prompt,
        num_channels_latents, height, width, prompt_embeds.dtype, device,
        generator, latents)
    max_objs = 30
    if len(gligen_boxes) > max_objs:
        warnings.warn(
            f'More that {max_objs} objects found. Only first {max_objs} objects will be processed.'
            , FutureWarning)
        gligen_phrases = gligen_phrases[:max_objs]
        gligen_boxes = gligen_boxes[:max_objs]
        gligen_images = gligen_images[:max_objs]
    repeat_batch = batch_size * num_images_per_prompt
    if do_classifier_free_guidance:
        repeat_batch = repeat_batch * 2
    if cross_attention_kwargs is None:
        cross_attention_kwargs = {}
    hidden_size = prompt_embeds.shape[2]
    cross_attention_kwargs['gligen'
        ] = self.get_cross_attention_kwargs_with_grounded(hidden_size=
        hidden_size, gligen_phrases=gligen_phrases, gligen_images=
        gligen_images, gligen_boxes=gligen_boxes, input_phrases_mask=
        input_phrases_mask, input_images_mask=input_images_mask,
        repeat_batch=repeat_batch, normalize_constant=
        gligen_normalize_constant, max_objs=max_objs, device=device)
    cross_attention_kwargs_without_grounded = {}
    cross_attention_kwargs_without_grounded['gligen'
        ] = self.get_cross_attention_kwargs_without_grounded(hidden_size=
        hidden_size, repeat_batch=repeat_batch, max_objs=max_objs, device=
        device)
    if gligen_inpaint_image is not None:
        if gligen_inpaint_image.size != (self.vae.sample_size, self.vae.
            sample_size):
            gligen_inpaint_image = self.target_size_center_crop(
                gligen_inpaint_image, self.vae.sample_size)
        gligen_inpaint_image = self.image_processor.preprocess(
            gligen_inpaint_image)
        gligen_inpaint_image = gligen_inpaint_image.to(dtype=self.vae.dtype,
            device=self.vae.device)
        gligen_inpaint_latent = self.vae.encode(gligen_inpaint_image
            ).latent_dist.sample()
        gligen_inpaint_latent = (self.vae.config.scaling_factor *
            gligen_inpaint_latent)
        gligen_inpaint_mask = self.draw_inpaint_mask_from_boxes(gligen_boxes,
            gligen_inpaint_latent.shape[2:])
        gligen_inpaint_mask = gligen_inpaint_mask.to(dtype=
            gligen_inpaint_latent.dtype, device=gligen_inpaint_latent.device)
        gligen_inpaint_mask = gligen_inpaint_mask[None, None]
        gligen_inpaint_mask_addition = torch.cat((gligen_inpaint_latent *
            gligen_inpaint_mask, gligen_inpaint_mask), dim=1)
        gligen_inpaint_mask_addition = gligen_inpaint_mask_addition.expand(
            repeat_batch, -1, -1, -1).clone()
    int(gligen_scheduled_sampling_beta * len(timesteps))
    self.enable_fuser(True)
    extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)
    num_warmup_steps = len(timesteps
        ) - num_inference_steps * self.scheduler.order
    with self.progress_bar(total=num_inference_steps) as progress_bar:
        for i, t in enumerate(timesteps):
            if latents.shape[1] != 4:
                latents = torch.randn_like(latents[:, :4])
            if gligen_inpaint_image is not None:
                gligen_inpaint_latent_with_noise = self.scheduler.add_noise(
                    gligen_inpaint_latent, torch.randn_like(
                    gligen_inpaint_latent), torch.tensor([t])).expand(latents
                    .shape[0], -1, -1, -1).clone()
                latents = (gligen_inpaint_latent_with_noise *
                    gligen_inpaint_mask + latents * (1 - gligen_inpaint_mask))
            latent_model_input = torch.cat([latents] * 2
                ) if do_classifier_free_guidance else latents
            latent_model_input = self.scheduler.scale_model_input(
                latent_model_input, t)
            if gligen_inpaint_image is not None:
                latent_model_input = torch.cat((latent_model_input,
                    gligen_inpaint_mask_addition), dim=1)
            noise_pred_with_grounding = self.unet(latent_model_input, t,
                encoder_hidden_states=prompt_embeds, cross_attention_kwargs
                =cross_attention_kwargs).sample
            noise_pred_without_grounding = self.unet(latent_model_input, t,
                encoder_hidden_states=prompt_embeds, cross_attention_kwargs
                =cross_attention_kwargs_without_grounded).sample
            if do_classifier_free_guidance:
                _, noise_pred_text = noise_pred_with_grounding.chunk(2)
                noise_pred_uncond, _ = noise_pred_without_grounding.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (
                    noise_pred_text - noise_pred_uncond)
            else:
                noise_pred = noise_pred_with_grounding
            latents = self.scheduler.step(noise_pred, t, latents, **
                extra_step_kwargs).prev_sample
            if i == len(timesteps) - 1 or i + 1 > num_warmup_steps and (i + 1
                ) % self.scheduler.order == 0:
                progress_bar.update()
                if callback is not None and i % callback_steps == 0:
                    step_idx = i // getattr(self.scheduler, 'order', 1)
                    callback(step_idx, t, latents)
    if not output_type == 'latent':
        image = self.vae.decode(latents / self.vae.config.scaling_factor,
            return_dict=False)[0]
        image, has_nsfw_concept = self.run_safety_checker(image, device,
            prompt_embeds.dtype)
    else:
        image = latents
        has_nsfw_concept = None
    if has_nsfw_concept is None:
        do_denormalize = [True] * image.shape[0]
    else:
        do_denormalize = [(not has_nsfw) for has_nsfw in has_nsfw_concept]
    image = self.image_processor.postprocess(image, output_type=output_type,
        do_denormalize=do_denormalize)
    self.maybe_free_model_hooks()
    if not return_dict:
        return image, has_nsfw_concept
    return StableDiffusionPipelineOutput(images=image,
        nsfw_content_detected=has_nsfw_concept)

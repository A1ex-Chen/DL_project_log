@torch.no_grad()
def __call__(self, prompt: Union[str, List[str]], prompt_2: Optional[Union[
    str, List[str]]]=None, video_length: Optional[int]=8, height: Optional[
    int]=None, width: Optional[int]=None, num_inference_steps: int=50,
    denoising_end: Optional[float]=None, guidance_scale: float=7.5,
    negative_prompt: Optional[Union[str, List[str]]]=None,
    negative_prompt_2: Optional[Union[str, List[str]]]=None,
    num_videos_per_prompt: Optional[int]=1, eta: float=0.0, generator:
    Optional[Union[torch.Generator, List[torch.Generator]]]=None, frame_ids:
    Optional[List[int]]=None, prompt_embeds: Optional[torch.Tensor]=None,
    negative_prompt_embeds: Optional[torch.Tensor]=None,
    pooled_prompt_embeds: Optional[torch.Tensor]=None,
    negative_pooled_prompt_embeds: Optional[torch.Tensor]=None, latents:
    Optional[torch.Tensor]=None, motion_field_strength_x: float=12,
    motion_field_strength_y: float=12, output_type: Optional[str]='tensor',
    return_dict: bool=True, callback: Optional[Callable[[int, int, torch.
    Tensor], None]]=None, callback_steps: int=1, cross_attention_kwargs:
    Optional[Dict[str, Any]]=None, guidance_rescale: float=0.0,
    original_size: Optional[Tuple[int, int]]=None, crops_coords_top_left:
    Tuple[int, int]=(0, 0), target_size: Optional[Tuple[int, int]]=None, t0:
    int=44, t1: int=47):
    """
        Function invoked when calling the pipeline for generation.

        Args:
            prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts to guide the image generation. If not defined, one has to pass `prompt_embeds`.
                instead.
            prompt_2 (`str` or `List[str]`, *optional*):
                The prompt or prompts to be sent to the `tokenizer_2` and `text_encoder_2`. If not defined, `prompt` is
                used in both text-encoders
            video_length (`int`, *optional*, defaults to 8):
                The number of generated video frames.
            height (`int`, *optional*, defaults to self.unet.config.sample_size * self.vae_scale_factor):
                The height in pixels of the generated image.
            width (`int`, *optional*, defaults to self.unet.config.sample_size * self.vae_scale_factor):
                The width in pixels of the generated image.
            num_inference_steps (`int`, *optional*, defaults to 50):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            denoising_end (`float`, *optional*):
                When specified, determines the fraction (between 0.0 and 1.0) of the total denoising process to be
                completed before it is intentionally prematurely terminated. As a result, the returned sample will
                still retain a substantial amount of noise as determined by the discrete timesteps selected by the
                scheduler. The denoising_end parameter should ideally be utilized when this pipeline forms a part of a
                "Mixture of Denoisers" multi-pipeline setup, as elaborated in [**Refining the Image
                Output**](https://huggingface.co/docs/diffusers/api/pipelines/stable_diffusion/stable_diffusion_xl#refining-the-image-output)
            guidance_scale (`float`, *optional*, defaults to 7.5):
                Guidance scale as defined in [Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598).
                `guidance_scale` is defined as `w` of equation 2. of [Imagen
                Paper](https://arxiv.org/pdf/2205.11487.pdf). Guidance scale is enabled by setting `guidance_scale >
                1`. Higher guidance scale encourages to generate images that are closely linked to the text `prompt`,
                usually at the expense of lower image quality.
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the image generation. If not defined, one has to pass
                `negative_prompt_embeds` instead. Ignored when not using guidance (i.e., ignored if `guidance_scale` is
                less than `1`).
            negative_prompt_2 (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the image generation to be sent to `tokenizer_2` and
                `text_encoder_2`. If not defined, `negative_prompt` is used in both text-encoders
            num_videos_per_prompt (`int`, *optional*, defaults to 1):
                The number of videos to generate per prompt.
            eta (`float`, *optional*, defaults to 0.0):
                Corresponds to parameter eta (η) in the DDIM paper: https://arxiv.org/abs/2010.02502. Only applies to
                [`schedulers.DDIMScheduler`], will be ignored for others.
            generator (`torch.Generator` or `List[torch.Generator]`, *optional*):
                One or a list of [torch generator(s)](https://pytorch.org/docs/stable/generated/torch.Generator.html)
                to make generation deterministic.
            frame_ids (`List[int]`, *optional*):
                Indexes of the frames that are being generated. This is used when generating longer videos
                chunk-by-chunk.
            prompt_embeds (`torch.Tensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not
                provided, text embeddings will be generated from `prompt` input argument.
            negative_prompt_embeds (`torch.Tensor`, *optional*):
                Pre-generated negative text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt
                weighting. If not provided, negative_prompt_embeds will be generated from `negative_prompt` input
                argument.
            pooled_prompt_embeds (`torch.Tensor`, *optional*):
                Pre-generated pooled text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting.
                If not provided, pooled text embeddings will be generated from `prompt` input argument.
            negative_pooled_prompt_embeds (`torch.Tensor`, *optional*):
                Pre-generated negative pooled text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt
                weighting. If not provided, pooled negative_prompt_embeds will be generated from `negative_prompt`
                input argument.
            latents (`torch.Tensor`, *optional*):
                Pre-generated noisy latents, sampled from a Gaussian distribution, to be used as inputs for image
                generation. Can be used to tweak the same generation with different prompts. If not provided, a latents
                tensor will ge generated by sampling using the supplied random `generator`.
            motion_field_strength_x (`float`, *optional*, defaults to 12):
                Strength of motion in generated video along x-axis. See the [paper](https://arxiv.org/abs/2303.13439),
                Sect. 3.3.1.
            motion_field_strength_y (`float`, *optional*, defaults to 12):
                Strength of motion in generated video along y-axis. See the [paper](https://arxiv.org/abs/2303.13439),
                Sect. 3.3.1.
            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generate image. Choose between
                [PIL](https://pillow.readthedocs.io/en/stable/): `PIL.Image.Image` or `np.array`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~pipelines.stable_diffusion_xl.StableDiffusionXLPipelineOutput`] instead
                of a plain tuple.
            callback (`Callable`, *optional*):
                A function that will be called every `callback_steps` steps during inference. The function will be
                called with the following arguments: `callback(step: int, timestep: int, latents: torch.Tensor)`.
            callback_steps (`int`, *optional*, defaults to 1):
                The frequency at which the `callback` function will be called. If not specified, the callback will be
                called at every step.
            cross_attention_kwargs (`dict`, *optional*):
                A kwargs dictionary that if specified is passed along to the `AttentionProcessor` as defined under
                `self.processor` in
                [diffusers.cross_attention](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/cross_attention.py).
            guidance_rescale (`float`, *optional*, defaults to 0.7):
                Guidance rescale factor proposed by [Common Diffusion Noise Schedules and Sample Steps are
                Flawed](https://arxiv.org/pdf/2305.08891.pdf) `guidance_scale` is defined as `φ` in equation 16. of
                [Common Diffusion Noise Schedules and Sample Steps are Flawed](https://arxiv.org/pdf/2305.08891.pdf).
                Guidance rescale factor should fix overexposure when using zero terminal SNR.
            original_size (`Tuple[int]`, *optional*, defaults to (1024, 1024)):
                If `original_size` is not the same as `target_size` the image will appear to be down- or upsampled.
                `original_size` defaults to `(width, height)` if not specified. Part of SDXL's micro-conditioning as
                explained in section 2.2 of
                [https://huggingface.co/papers/2307.01952](https://huggingface.co/papers/2307.01952).
            crops_coords_top_left (`Tuple[int]`, *optional*, defaults to (0, 0)):
                `crops_coords_top_left` can be used to generate an image that appears to be "cropped" from the position
                `crops_coords_top_left` downwards. Favorable, well-centered images are usually achieved by setting
                `crops_coords_top_left` to (0, 0). Part of SDXL's micro-conditioning as explained in section 2.2 of
                [https://huggingface.co/papers/2307.01952](https://huggingface.co/papers/2307.01952).
            target_size (`Tuple[int]`, *optional*, defaults to (1024, 1024)):
                For most cases, `target_size` should be set to the desired height and width of the generated image. If
                not specified it will default to `(width, height)`. Part of SDXL's micro-conditioning as explained in
                section 2.2 of [https://huggingface.co/papers/2307.01952](https://huggingface.co/papers/2307.01952).
            t0 (`int`, *optional*, defaults to 44):
                Timestep t0. Should be in the range [0, num_inference_steps - 1]. See the
                [paper](https://arxiv.org/abs/2303.13439), Sect. 3.3.1.
            t1 (`int`, *optional*, defaults to 47):
                Timestep t0. Should be in the range [t0 + 1, num_inference_steps - 1]. See the
                [paper](https://arxiv.org/abs/2303.13439), Sect. 3.3.1.

        Returns:
            [`~pipelines.text_to_video_synthesis.pipeline_text_to_video_zero.TextToVideoSDXLPipelineOutput`] or
            `tuple`: [`~pipelines.text_to_video_synthesis.pipeline_text_to_video_zero.TextToVideoSDXLPipelineOutput`]
            if `return_dict` is True, otherwise a `tuple`. When returning a tuple, the first element is a list with the
            generated images.
        """
    assert video_length > 0
    if frame_ids is None:
        frame_ids = list(range(video_length))
    assert len(frame_ids) == video_length
    assert num_videos_per_prompt == 1
    original_attn_proc = self.unet.attn_processors
    processor = CrossFrameAttnProcessor2_0(batch_size=2) if hasattr(F,
        'scaled_dot_product_attention') else CrossFrameAttnProcessor(batch_size
        =2)
    self.unet.set_attn_processor(processor)
    if isinstance(prompt, str):
        prompt = [prompt]
    if isinstance(negative_prompt, str):
        negative_prompt = [negative_prompt]
    height = height or self.default_sample_size * self.vae_scale_factor
    width = width or self.default_sample_size * self.vae_scale_factor
    original_size = original_size or (height, width)
    target_size = target_size or (height, width)
    self.check_inputs(prompt, prompt_2, height, width, callback_steps,
        negative_prompt, negative_prompt_2, prompt_embeds,
        negative_prompt_embeds, pooled_prompt_embeds,
        negative_pooled_prompt_embeds)
    batch_size = 1 if isinstance(prompt, str) else len(prompt) if isinstance(
        prompt, list) else prompt_embeds.shape[0]
    device = self._execution_device
    do_classifier_free_guidance = guidance_scale > 1.0
    text_encoder_lora_scale = cross_attention_kwargs.get('scale', None
        ) if cross_attention_kwargs is not None else None
    (prompt_embeds, negative_prompt_embeds, pooled_prompt_embeds,
        negative_pooled_prompt_embeds) = (self.encode_prompt(prompt=prompt,
        prompt_2=prompt_2, device=device, num_images_per_prompt=
        num_videos_per_prompt, do_classifier_free_guidance=
        do_classifier_free_guidance, negative_prompt=negative_prompt,
        negative_prompt_2=negative_prompt_2, prompt_embeds=prompt_embeds,
        negative_prompt_embeds=negative_prompt_embeds, pooled_prompt_embeds
        =pooled_prompt_embeds, negative_pooled_prompt_embeds=
        negative_pooled_prompt_embeds, lora_scale=text_encoder_lora_scale))
    self.scheduler.set_timesteps(num_inference_steps, device=device)
    timesteps = self.scheduler.timesteps
    num_channels_latents = self.unet.config.in_channels
    latents = self.prepare_latents(batch_size * num_videos_per_prompt,
        num_channels_latents, height, width, prompt_embeds.dtype, device,
        generator, latents)
    extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)
    add_text_embeds = pooled_prompt_embeds
    if self.text_encoder_2 is None:
        text_encoder_projection_dim = int(pooled_prompt_embeds.shape[-1])
    else:
        text_encoder_projection_dim = self.text_encoder_2.config.projection_dim
    add_time_ids = self._get_add_time_ids(original_size,
        crops_coords_top_left, target_size, dtype=prompt_embeds.dtype,
        text_encoder_projection_dim=text_encoder_projection_dim)
    if do_classifier_free_guidance:
        prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds],
            dim=0)
        add_text_embeds = torch.cat([negative_pooled_prompt_embeds,
            add_text_embeds], dim=0)
        add_time_ids = torch.cat([add_time_ids, add_time_ids], dim=0)
    prompt_embeds = prompt_embeds.to(device)
    add_text_embeds = add_text_embeds.to(device)
    add_time_ids = add_time_ids.to(device).repeat(batch_size *
        num_videos_per_prompt, 1)
    num_warmup_steps = len(timesteps
        ) - num_inference_steps * self.scheduler.order
    x_1_t1 = self.backward_loop(timesteps=timesteps[:-t1 - 1],
        prompt_embeds=prompt_embeds, latents=latents, guidance_scale=
        guidance_scale, callback=callback, callback_steps=callback_steps,
        extra_step_kwargs=extra_step_kwargs, num_warmup_steps=
        num_warmup_steps, add_text_embeds=add_text_embeds, add_time_ids=
        add_time_ids)
    scheduler_copy = copy.deepcopy(self.scheduler)
    x_1_t0 = self.backward_loop(timesteps=timesteps[-t1 - 1:-t0 - 1],
        prompt_embeds=prompt_embeds, latents=x_1_t1, guidance_scale=
        guidance_scale, callback=callback, callback_steps=callback_steps,
        extra_step_kwargs=extra_step_kwargs, num_warmup_steps=0,
        add_text_embeds=add_text_embeds, add_time_ids=add_time_ids)
    x_2k_t0 = x_1_t0.repeat(video_length - 1, 1, 1, 1)
    x_2k_t0 = create_motion_field_and_warp_latents(motion_field_strength_x=
        motion_field_strength_x, motion_field_strength_y=
        motion_field_strength_y, latents=x_2k_t0, frame_ids=frame_ids[1:])
    x_2k_t1 = self.forward_loop(x_t0=x_2k_t0, t0=timesteps[-t0 - 1].to(
        torch.long), t1=timesteps[-t1 - 1].to(torch.long), generator=generator)
    latents = torch.cat([x_1_t1, x_2k_t1])
    self.scheduler = scheduler_copy
    timesteps = timesteps[-t1 - 1:]
    b, l, d = prompt_embeds.size()
    prompt_embeds = prompt_embeds[:, None].repeat(1, video_length, 1, 1
        ).reshape(b * video_length, l, d)
    b, k = add_text_embeds.size()
    add_text_embeds = add_text_embeds[:, None].repeat(1, video_length, 1
        ).reshape(b * video_length, k)
    b, k = add_time_ids.size()
    add_time_ids = add_time_ids[:, None].repeat(1, video_length, 1).reshape(
        b * video_length, k)
    if denoising_end is not None and isinstance(denoising_end, float
        ) and denoising_end > 0 and denoising_end < 1:
        discrete_timestep_cutoff = int(round(self.scheduler.config.
            num_train_timesteps - denoising_end * self.scheduler.config.
            num_train_timesteps))
        num_inference_steps = len(list(filter(lambda ts: ts >=
            discrete_timestep_cutoff, timesteps)))
        timesteps = timesteps[:num_inference_steps]
    x_1k_0 = self.backward_loop(timesteps=timesteps, prompt_embeds=
        prompt_embeds, latents=latents, guidance_scale=guidance_scale,
        callback=callback, callback_steps=callback_steps, extra_step_kwargs
        =extra_step_kwargs, num_warmup_steps=0, add_text_embeds=
        add_text_embeds, add_time_ids=add_time_ids)
    latents = x_1k_0
    if not output_type == 'latent':
        needs_upcasting = (self.vae.dtype == torch.float16 and self.vae.
            config.force_upcast)
        if needs_upcasting:
            self.upcast_vae()
            latents = latents.to(next(iter(self.vae.post_quant_conv.
                parameters())).dtype)
        image = self.vae.decode(latents / self.vae.config.scaling_factor,
            return_dict=False)[0]
        if needs_upcasting:
            self.vae.to(dtype=torch.float16)
    else:
        image = latents
        return TextToVideoSDXLPipelineOutput(images=image)
    if self.watermark is not None:
        image = self.watermark.apply_watermark(image)
    image = self.image_processor.postprocess(image, output_type=output_type)
    self.maybe_free_model_hooks()
    self.unet.set_attn_processor(original_attn_proc)
    if not return_dict:
        return image,
    return TextToVideoSDXLPipelineOutput(images=image)

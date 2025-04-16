@torch.no_grad()
@replace_example_docstring(EXAMPLE_DOC_STRING)
def __call__(self, prompt: Union[str, List[str]]=None, prompt_2: Optional[
    Union[str, List[str]]]=None, image: PipelineImageInput=None, height:
    Optional[int]=None, width: Optional[int]=None, num_inference_steps: int
    =50, guidance_scale: float=5.0, negative_prompt: Optional[Union[str,
    List[str]]]=None, negative_prompt_2: Optional[Union[str, List[str]]]=
    None, num_images_per_prompt: Optional[int]=1, eta: float=0.0, generator:
    Optional[Union[torch.Generator, List[torch.Generator]]]=None, latents:
    Optional[torch.Tensor]=None, prompt_embeds: Optional[torch.Tensor]=None,
    negative_prompt_embeds: Optional[torch.Tensor]=None,
    pooled_prompt_embeds: Optional[torch.Tensor]=None,
    negative_pooled_prompt_embeds: Optional[torch.Tensor]=None, output_type:
    Optional[str]='pil', return_dict: bool=True, cross_attention_kwargs:
    Optional[Dict[str, Any]]=None, controlnet_conditioning_scale: Union[
    float, List[float]]=1.0, control_guidance_start: float=0.0,
    control_guidance_end: float=1.0, original_size: Tuple[int, int]=None,
    crops_coords_top_left: Tuple[int, int]=(0, 0), target_size: Tuple[int,
    int]=None, negative_original_size: Optional[Tuple[int, int]]=None,
    negative_crops_coords_top_left: Tuple[int, int]=(0, 0),
    negative_target_size: Optional[Tuple[int, int]]=None, clip_skip:
    Optional[int]=None, callback_on_step_end: Optional[Union[Callable[[int,
    int, Dict], None], PipelineCallback, MultiPipelineCallbacks]]=None,
    callback_on_step_end_tensor_inputs: List[str]=['latents']):
    """
        The call function to the pipeline for generation.

        Args:
            prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts to guide image generation. If not defined, you need to pass `prompt_embeds`.
            prompt_2 (`str` or `List[str]`, *optional*):
                The prompt or prompts to be sent to `tokenizer_2` and `text_encoder_2`. If not defined, `prompt` is
                used in both text-encoders.
            image (`torch.Tensor`, `PIL.Image.Image`, `np.ndarray`, `List[torch.Tensor]`, `List[PIL.Image.Image]`, `List[np.ndarray]`,:
                    `List[List[torch.Tensor]]`, `List[List[np.ndarray]]` or `List[List[PIL.Image.Image]]`):
                The ControlNet input condition to provide guidance to the `unet` for generation. If the type is
                specified as `torch.Tensor`, it is passed to ControlNet as is. `PIL.Image.Image` can also be accepted
                as an image. The dimensions of the output image defaults to `image`'s dimensions. If height and/or
                width are passed, `image` is resized accordingly. If multiple ControlNets are specified in `init`,
                images must be passed as a list such that each element of the list can be correctly batched for input
                to a single ControlNet.
            height (`int`, *optional*, defaults to `self.unet.config.sample_size * self.vae_scale_factor`):
                The height in pixels of the generated image. Anything below 512 pixels won't work well for
                [stabilityai/stable-diffusion-xl-base-1.0](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0)
                and checkpoints that are not specifically fine-tuned on low resolutions.
            width (`int`, *optional*, defaults to `self.unet.config.sample_size * self.vae_scale_factor`):
                The width in pixels of the generated image. Anything below 512 pixels won't work well for
                [stabilityai/stable-diffusion-xl-base-1.0](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0)
                and checkpoints that are not specifically fine-tuned on low resolutions.
            num_inference_steps (`int`, *optional*, defaults to 50):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            guidance_scale (`float`, *optional*, defaults to 5.0):
                A higher guidance scale value encourages the model to generate images closely linked to the text
                `prompt` at the expense of lower image quality. Guidance scale is enabled when `guidance_scale > 1`.
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts to guide what to not include in image generation. If not defined, you need to
                pass `negative_prompt_embeds` instead. Ignored when not using guidance (`guidance_scale < 1`).
            negative_prompt_2 (`str` or `List[str]`, *optional*):
                The prompt or prompts to guide what to not include in image generation. This is sent to `tokenizer_2`
                and `text_encoder_2`. If not defined, `negative_prompt` is used in both text-encoders.
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
            pooled_prompt_embeds (`torch.Tensor`, *optional*):
                Pre-generated pooled text embeddings. Can be used to easily tweak text inputs (prompt weighting). If
                not provided, pooled text embeddings are generated from `prompt` input argument.
            negative_pooled_prompt_embeds (`torch.Tensor`, *optional*):
                Pre-generated negative pooled text embeddings. Can be used to easily tweak text inputs (prompt
                weighting). If not provided, pooled `negative_prompt_embeds` are generated from `negative_prompt` input
                argument.
            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generated image. Choose between `PIL.Image` or `np.array`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] instead of a
                plain tuple.
            cross_attention_kwargs (`dict`, *optional*):
                A kwargs dictionary that if specified is passed along to the [`AttentionProcessor`] as defined in
                [`self.processor`](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py).
            controlnet_conditioning_scale (`float` or `List[float]`, *optional*, defaults to 1.0):
                The outputs of the ControlNet are multiplied by `controlnet_conditioning_scale` before they are added
                to the residual in the original `unet`.
            control_guidance_start (`float`, *optional*, defaults to 0.0):
                The percentage of total steps at which the ControlNet starts applying.
            control_guidance_end (`float`, *optional*, defaults to 1.0):
                The percentage of total steps at which the ControlNet stops applying.
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
            negative_original_size (`Tuple[int]`, *optional*, defaults to (1024, 1024)):
                To negatively condition the generation process based on a specific image resolution. Part of SDXL's
                micro-conditioning as explained in section 2.2 of
                [https://huggingface.co/papers/2307.01952](https://huggingface.co/papers/2307.01952). For more
                information, refer to this issue thread: https://github.com/huggingface/diffusers/issues/4208.
            negative_crops_coords_top_left (`Tuple[int]`, *optional*, defaults to (0, 0)):
                To negatively condition the generation process based on a specific crop coordinates. Part of SDXL's
                micro-conditioning as explained in section 2.2 of
                [https://huggingface.co/papers/2307.01952](https://huggingface.co/papers/2307.01952). For more
                information, refer to this issue thread: https://github.com/huggingface/diffusers/issues/4208.
            negative_target_size (`Tuple[int]`, *optional*, defaults to (1024, 1024)):
                To negatively condition the generation process based on a target image resolution. It should be as same
                as the `target_size` for most cases. Part of SDXL's micro-conditioning as explained in section 2.2 of
                [https://huggingface.co/papers/2307.01952](https://huggingface.co/papers/2307.01952). For more
                information, refer to this issue thread: https://github.com/huggingface/diffusers/issues/4208.
            clip_skip (`int`, *optional*):
                Number of layers to be skipped from CLIP while computing the prompt embeddings. A value of 1 means that
                the output of the pre-final layer will be used for computing the prompt embeddings.
            callback_on_step_end (`Callable`, `PipelineCallback`, `MultiPipelineCallbacks`, *optional*):
                A function or a subclass of `PipelineCallback` or `MultiPipelineCallbacks` that is called at the end of
                each denoising step during the inference. with the following arguments: `callback_on_step_end(self:
                DiffusionPipeline, step: int, timestep: int, callback_kwargs: Dict)`. `callback_kwargs` will include a
                list of all tensors as specified by `callback_on_step_end_tensor_inputs`.
            callback_on_step_end_tensor_inputs (`List`, *optional*):
                The list of tensor inputs for the `callback_on_step_end` function. The tensors specified in the list
                will be passed as `callback_kwargs` argument. You will only be able to include variables listed in the
                `._callback_tensor_inputs` attribute of your pipeine class.

        Examples:

        Returns:
            [`~pipelines.stable_diffusion.StableDiffusionXLPipelineOutput`] or `tuple`:
                If `return_dict` is `True`, [`~pipelines.stable_diffusion.StableDiffusionXLPipelineOutput`] is
                returned, otherwise a `tuple` is returned containing the output images.
        """
    if isinstance(callback_on_step_end, (PipelineCallback,
        MultiPipelineCallbacks)):
        callback_on_step_end_tensor_inputs = callback_on_step_end.tensor_inputs
    unet = self.unet._orig_mod if is_compiled_module(self.unet) else self.unet
    self.check_inputs(prompt, prompt_2, image, negative_prompt,
        negative_prompt_2, prompt_embeds, negative_prompt_embeds,
        pooled_prompt_embeds, negative_pooled_prompt_embeds,
        controlnet_conditioning_scale, control_guidance_start,
        control_guidance_end, callback_on_step_end_tensor_inputs)
    self._guidance_scale = guidance_scale
    self._clip_skip = clip_skip
    self._cross_attention_kwargs = cross_attention_kwargs
    self._interrupt = False
    if prompt is not None and isinstance(prompt, str):
        batch_size = 1
    elif prompt is not None and isinstance(prompt, list):
        batch_size = len(prompt)
    else:
        batch_size = prompt_embeds.shape[0]
    device = self._execution_device
    do_classifier_free_guidance = guidance_scale > 1.0
    text_encoder_lora_scale = cross_attention_kwargs.get('scale', None
        ) if cross_attention_kwargs is not None else None
    (prompt_embeds, negative_prompt_embeds, pooled_prompt_embeds,
        negative_pooled_prompt_embeds) = (self.encode_prompt(prompt,
        prompt_2, device, num_images_per_prompt,
        do_classifier_free_guidance, negative_prompt, negative_prompt_2,
        prompt_embeds=prompt_embeds, negative_prompt_embeds=
        negative_prompt_embeds, pooled_prompt_embeds=pooled_prompt_embeds,
        negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
        lora_scale=text_encoder_lora_scale, clip_skip=clip_skip))
    if isinstance(unet, UNetControlNetXSModel):
        image = self.prepare_image(image=image, width=width, height=height,
            batch_size=batch_size * num_images_per_prompt,
            num_images_per_prompt=num_images_per_prompt, device=device,
            dtype=unet.dtype, do_classifier_free_guidance=
            do_classifier_free_guidance)
        height, width = image.shape[-2:]
    else:
        assert False
    self.scheduler.set_timesteps(num_inference_steps, device=device)
    timesteps = self.scheduler.timesteps
    num_channels_latents = self.unet.in_channels
    latents = self.prepare_latents(batch_size * num_images_per_prompt,
        num_channels_latents, height, width, prompt_embeds.dtype, device,
        generator, latents)
    extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)
    if isinstance(image, list):
        original_size = original_size or image[0].shape[-2:]
    else:
        original_size = original_size or image.shape[-2:]
    target_size = target_size or (height, width)
    add_text_embeds = pooled_prompt_embeds
    if self.text_encoder_2 is None:
        text_encoder_projection_dim = int(pooled_prompt_embeds.shape[-1])
    else:
        text_encoder_projection_dim = self.text_encoder_2.config.projection_dim
    add_time_ids = self._get_add_time_ids(original_size,
        crops_coords_top_left, target_size, dtype=prompt_embeds.dtype,
        text_encoder_projection_dim=text_encoder_projection_dim)
    if negative_original_size is not None and negative_target_size is not None:
        negative_add_time_ids = self._get_add_time_ids(negative_original_size,
            negative_crops_coords_top_left, negative_target_size, dtype=
            prompt_embeds.dtype, text_encoder_projection_dim=
            text_encoder_projection_dim)
    else:
        negative_add_time_ids = add_time_ids
    if do_classifier_free_guidance:
        prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds],
            dim=0)
        add_text_embeds = torch.cat([negative_pooled_prompt_embeds,
            add_text_embeds], dim=0)
        add_time_ids = torch.cat([negative_add_time_ids, add_time_ids], dim=0)
    prompt_embeds = prompt_embeds.to(device)
    add_text_embeds = add_text_embeds.to(device)
    add_time_ids = add_time_ids.to(device).repeat(batch_size *
        num_images_per_prompt, 1)
    num_warmup_steps = len(timesteps
        ) - num_inference_steps * self.scheduler.order
    self._num_timesteps = len(timesteps)
    is_controlnet_compiled = is_compiled_module(self.unet)
    is_torch_higher_equal_2_1 = is_torch_version('>=', '2.1')
    with self.progress_bar(total=num_inference_steps) as progress_bar:
        for i, t in enumerate(timesteps):
            if is_controlnet_compiled and is_torch_higher_equal_2_1:
                torch._inductor.cudagraph_mark_step_begin()
            latent_model_input = torch.cat([latents] * 2
                ) if do_classifier_free_guidance else latents
            latent_model_input = self.scheduler.scale_model_input(
                latent_model_input, t)
            added_cond_kwargs = {'text_embeds': add_text_embeds, 'time_ids':
                add_time_ids}
            apply_control = i / len(timesteps) >= control_guidance_start and (i
                 + 1) / len(timesteps) <= control_guidance_end
            noise_pred = self.unet(sample=latent_model_input, timestep=t,
                encoder_hidden_states=prompt_embeds, controlnet_cond=image,
                conditioning_scale=controlnet_conditioning_scale,
                cross_attention_kwargs=cross_attention_kwargs,
                added_cond_kwargs=added_cond_kwargs, return_dict=True,
                apply_control=apply_control).sample
            if do_classifier_free_guidance:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (
                    noise_pred_text - noise_pred_uncond)
            latents = self.scheduler.step(noise_pred, t, latents, **
                extra_step_kwargs, return_dict=False)[0]
            if callback_on_step_end is not None:
                callback_kwargs = {}
                for k in callback_on_step_end_tensor_inputs:
                    callback_kwargs[k] = locals()[k]
                callback_outputs = callback_on_step_end(self, i, t,
                    callback_kwargs)
                latents = callback_outputs.pop('latents', latents)
                prompt_embeds = callback_outputs.pop('prompt_embeds',
                    prompt_embeds)
                negative_prompt_embeds = callback_outputs.pop(
                    'negative_prompt_embeds', negative_prompt_embeds)
            if i == len(timesteps) - 1 or i + 1 > num_warmup_steps and (i + 1
                ) % self.scheduler.order == 0:
                progress_bar.update()
    if self.vae.dtype == torch.float16 and self.vae.config.force_upcast:
        self.upcast_vae()
        latents = latents.to(next(iter(self.vae.post_quant_conv.parameters(
            ))).dtype)
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
    if not output_type == 'latent':
        if self.watermark is not None:
            image = self.watermark.apply_watermark(image)
        image = self.image_processor.postprocess(image, output_type=output_type
            )
    self.maybe_free_model_hooks()
    if not return_dict:
        return image,
    return StableDiffusionXLPipelineOutput(images=image)

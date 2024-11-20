@torch.no_grad()
@replace_example_docstring(EXAMPLE_DOC_STRING)
def __call__(self, prompt: Optional[Union[str, List[str]]]=None, prompt_2:
    Optional[Union[str, List[str]]]=None, image: Optional[Union[torch.
    Tensor, PIL.Image.Image]]=None, mask_image: Optional[Union[torch.Tensor,
    PIL.Image.Image]]=None, adapter_image: PipelineImageInput=None,
    control_image: PipelineImageInput=None, height: Optional[int]=None,
    width: Optional[int]=None, strength: float=0.9999, num_inference_steps:
    int=50, denoising_start: Optional[float]=None, denoising_end: Optional[
    float]=None, guidance_scale: float=5.0, negative_prompt: Optional[Union
    [str, List[str]]]=None, negative_prompt_2: Optional[Union[str, List[str
    ]]]=None, num_images_per_prompt: Optional[int]=1, eta: float=0.0,
    generator: Optional[Union[torch.Generator, List[torch.Generator]]]=None,
    latents: Optional[Union[torch.Tensor]]=None, prompt_embeds: Optional[
    torch.Tensor]=None, negative_prompt_embeds: Optional[torch.Tensor]=None,
    pooled_prompt_embeds: Optional[torch.Tensor]=None,
    negative_pooled_prompt_embeds: Optional[torch.Tensor]=None, output_type:
    Optional[str]='pil', return_dict: bool=True, callback: Optional[
    Callable[[int, int, torch.Tensor], None]]=None, callback_steps: int=1,
    cross_attention_kwargs: Optional[Dict[str, Any]]=None, guidance_rescale:
    float=0.0, original_size: Optional[Tuple[int, int]]=None,
    crops_coords_top_left: Optional[Tuple[int, int]]=(0, 0), target_size:
    Optional[Tuple[int, int]]=None, adapter_conditioning_scale: Optional[
    Union[float, List[float]]]=1.0, cond_tau: float=1.0, aesthetic_score:
    float=6.0, negative_aesthetic_score: float=2.5,
    controlnet_conditioning_scale=1.0, guess_mode: bool=False,
    control_guidance_start=0.0, control_guidance_end=1.0):
    """
        Function invoked when calling the pipeline for generation.

        Args:
            prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts to guide the image generation. If not defined, one has to pass `prompt_embeds`.
                instead.
            prompt_2 (`str` or `List[str]`, *optional*):
                The prompt or prompts to be sent to the `tokenizer_2` and `text_encoder_2`. If not defined, `prompt` is
                used in both text-encoders
            image (`PIL.Image.Image`):
                `Image`, or tensor representing an image batch which will be inpainted, *i.e.* parts of the image will
                be masked out with `mask_image` and repainted according to `prompt`.
            mask_image (`PIL.Image.Image`):
                `Image`, or tensor representing an image batch, to mask `image`. White pixels in the mask will be
                repainted, while black pixels will be preserved. If `mask_image` is a PIL image, it will be converted
                to a single channel (luminance) before use. If it's a tensor, it should contain one color channel (L)
                instead of 3, so the expected shape would be `(B, H, W, 1)`.
            adapter_image (`torch.Tensor`, `PIL.Image.Image`, `List[torch.Tensor]` or `List[PIL.Image.Image]` or `List[List[PIL.Image.Image]]`):
                The Adapter input condition. Adapter uses this input condition to generate guidance to Unet. If the
                type is specified as `torch.Tensor`, it is passed to Adapter as is. PIL.Image.Image` can also be
                accepted as an image. The control image is automatically resized to fit the output image.
            control_image (`torch.Tensor`, `PIL.Image.Image`, `np.ndarray`, `List[torch.Tensor]`, `List[PIL.Image.Image]`, `List[np.ndarray]`,:
                    `List[List[torch.Tensor]]`, `List[List[np.ndarray]]` or `List[List[PIL.Image.Image]]`):
                The ControlNet input condition to provide guidance to the `unet` for generation. If the type is
                specified as `torch.Tensor`, it is passed to ControlNet as is. `PIL.Image.Image` can also be
                accepted as an image. The dimensions of the output image defaults to `image`'s dimensions. If height
                and/or width are passed, `image` is resized accordingly. If multiple ControlNets are specified in
                `init`, images must be passed as a list such that each element of the list can be correctly batched for
                input to a single ControlNet.
            height (`int`, *optional*, defaults to self.unet.config.sample_size * self.vae_scale_factor):
                The height in pixels of the generated image.
            width (`int`, *optional*, defaults to self.unet.config.sample_size * self.vae_scale_factor):
                The width in pixels of the generated image.
            strength (`float`, *optional*, defaults to 1.0):
                Indicates extent to transform the reference `image`. Must be between 0 and 1. `image` is used as a
                starting point and more noise is added the higher the `strength`. The number of denoising steps depends
                on the amount of noise initially added. When `strength` is 1, added noise is maximum and the denoising
                process runs for the full number of iterations specified in `num_inference_steps`. A value of 1
                essentially ignores `image`.
            num_inference_steps (`int`, *optional*, defaults to 50):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            denoising_start (`float`, *optional*):
                When specified, indicates the fraction (between 0.0 and 1.0) of the total denoising process to be
                bypassed before it is initiated. Consequently, the initial part of the denoising process is skipped and
                it is assumed that the passed `image` is a partly denoised image. Note that when this is specified,
                strength will be ignored. The `denoising_start` parameter is particularly beneficial when this pipeline
                is integrated into a "Mixture of Denoisers" multi-pipeline setup, as detailed in [**Refining the Image
                Output**](https://huggingface.co/docs/diffusers/api/pipelines/stable_diffusion/stable_diffusion_xl#refining-the-image-output).
            denoising_end (`float`, *optional*):
                When specified, determines the fraction (between 0.0 and 1.0) of the total denoising process to be
                completed before it is intentionally prematurely terminated. As a result, the returned sample will
                still retain a substantial amount of noise as determined by the discrete timesteps selected by the
                scheduler. The denoising_end parameter should ideally be utilized when this pipeline forms a part of a
                "Mixture of Denoisers" multi-pipeline setup, as elaborated in [**Refining the Image
                Output**](https://huggingface.co/docs/diffusers/api/pipelines/stable_diffusion/stable_diffusion_xl#refining-the-image-output)
            guidance_scale (`float`, *optional*, defaults to 5.0):
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
            pooled_prompt_embeds (`torch.Tensor`, *optional*):
                Pre-generated pooled text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting.
                If not provided, pooled text embeddings will be generated from `prompt` input argument.
            negative_pooled_prompt_embeds (`torch.Tensor`, *optional*):
                Pre-generated negative pooled text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt
                weighting. If not provided, pooled negative_prompt_embeds will be generated from `negative_prompt`
                input argument.
            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generate image. Choose between
                [PIL](https://pillow.readthedocs.io/en/stable/): `PIL.Image.Image` or `np.array`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~pipelines.stable_diffusion_xl.StableDiffusionAdapterPipelineOutput`]
                instead of a plain tuple.
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
            controlnet_conditioning_scale (`float` or `List[float]`, *optional*, defaults to 1.0):
                The outputs of the controlnet are multiplied by `controlnet_conditioning_scale` before they are added to the
                residual in the original unet. If multiple adapters are specified in init, you can set the
                corresponding scale as a list.
            adapter_conditioning_scale (`float` or `List[float]`, *optional*, defaults to 1.0):
                The outputs of the adapter are multiplied by `adapter_conditioning_scale` before they are added to the
                residual in the original unet. If multiple adapters are specified in init, you can set the
                corresponding scale as a list.
            aesthetic_score (`float`, *optional*, defaults to 6.0):
                Used to simulate an aesthetic score of the generated image by influencing the positive text condition.
                Part of SDXL's micro-conditioning as explained in section 2.2 of
                [https://huggingface.co/papers/2307.01952](https://huggingface.co/papers/2307.01952).
            negative_aesthetic_score (`float`, *optional*, defaults to 2.5):
                Part of SDXL's micro-conditioning as explained in section 2.2 of
                [https://huggingface.co/papers/2307.01952](https://huggingface.co/papers/2307.01952). Can be used to
                simulate an aesthetic score of the generated image by influencing the negative text condition.
        Examples:

        Returns:
            [`~pipelines.stable_diffusion.StableDiffusionAdapterPipelineOutput`] or `tuple`:
            [`~pipelines.stable_diffusion.StableDiffusionAdapterPipelineOutput`] if `return_dict` is True, otherwise a
            `tuple`. When returning a tuple, the first element is a list with the generated images.
        """
    controlnet = self.controlnet._orig_mod if is_compiled_module(self.
        controlnet) else self.controlnet
    adapter = self.adapter._orig_mod if is_compiled_module(self.adapter
        ) else self.adapter
    height, width = self._default_height_width(height, width, adapter_image)
    device = self._execution_device
    if isinstance(adapter, MultiAdapter):
        adapter_input = []
        for one_image in adapter_image:
            one_image = _preprocess_adapter_image(one_image, height, width)
            one_image = one_image.to(device=device, dtype=adapter.dtype)
            adapter_input.append(one_image)
    else:
        adapter_input = _preprocess_adapter_image(adapter_image, height, width)
        adapter_input = adapter_input.to(device=device, dtype=adapter.dtype)
    original_size = original_size or (height, width)
    target_size = target_size or (height, width)
    if not isinstance(control_guidance_start, list) and isinstance(
        control_guidance_end, list):
        control_guidance_start = len(control_guidance_end) * [
            control_guidance_start]
    elif not isinstance(control_guidance_end, list) and isinstance(
        control_guidance_start, list):
        control_guidance_end = len(control_guidance_start) * [
            control_guidance_end]
    elif not isinstance(control_guidance_start, list) and not isinstance(
        control_guidance_end, list):
        mult = len(controlnet.nets) if isinstance(controlnet,
            MultiControlNetModel) else 1
        control_guidance_start, control_guidance_end = mult * [
            control_guidance_start], mult * [control_guidance_end]
    if isinstance(controlnet, MultiControlNetModel) and isinstance(
        controlnet_conditioning_scale, float):
        controlnet_conditioning_scale = [controlnet_conditioning_scale] * len(
            controlnet.nets)
    if isinstance(adapter, MultiAdapter) and isinstance(
        adapter_conditioning_scale, float):
        adapter_conditioning_scale = [adapter_conditioning_scale] * len(adapter
            .nets)
    self.check_inputs(prompt, prompt_2, height, width, callback_steps,
        negative_prompt=negative_prompt, negative_prompt_2=
        negative_prompt_2, prompt_embeds=prompt_embeds,
        negative_prompt_embeds=negative_prompt_embeds, pooled_prompt_embeds
        =pooled_prompt_embeds, negative_pooled_prompt_embeds=
        negative_pooled_prompt_embeds)
    self.check_conditions(prompt, prompt_embeds, adapter_image,
        control_image, adapter_conditioning_scale,
        controlnet_conditioning_scale, control_guidance_start,
        control_guidance_end)
    if prompt is not None and isinstance(prompt, str):
        batch_size = 1
    elif prompt is not None and isinstance(prompt, list):
        batch_size = len(prompt)
    else:
        batch_size = prompt_embeds.shape[0]
    device = self._execution_device
    do_classifier_free_guidance = guidance_scale > 1.0
    (prompt_embeds, negative_prompt_embeds, pooled_prompt_embeds,
        negative_pooled_prompt_embeds) = (self.encode_prompt(prompt=prompt,
        prompt_2=prompt_2, device=device, num_images_per_prompt=
        num_images_per_prompt, do_classifier_free_guidance=
        do_classifier_free_guidance, negative_prompt=negative_prompt,
        negative_prompt_2=negative_prompt_2, prompt_embeds=prompt_embeds,
        negative_prompt_embeds=negative_prompt_embeds, pooled_prompt_embeds
        =pooled_prompt_embeds, negative_pooled_prompt_embeds=
        negative_pooled_prompt_embeds))

    def denoising_value_valid(dnv):
        return isinstance(dnv, float) and 0 < dnv < 1
    self.scheduler.set_timesteps(num_inference_steps, device=device)
    timesteps, num_inference_steps = self.get_timesteps(num_inference_steps,
        strength, device, denoising_start=denoising_start if
        denoising_value_valid(denoising_start) else None)
    if num_inference_steps < 1:
        raise ValueError(
            f'After adjusting the num_inference_steps by strength parameter: {strength}, the number of pipelinesteps is {num_inference_steps} which is < 1 and not appropriate for this pipeline.'
            )
    latent_timestep = timesteps[:1].repeat(batch_size * num_images_per_prompt)
    is_strength_max = strength == 1.0
    mask, masked_image, init_image = prepare_mask_and_masked_image(image,
        mask_image, height, width, return_image=True)
    num_channels_latents = self.vae.config.latent_channels
    num_channels_unet = self.unet.config.in_channels
    return_image_latents = num_channels_unet == 4
    add_noise = denoising_start is None
    latents_outputs = self.prepare_latents(batch_size *
        num_images_per_prompt, num_channels_latents, height, width,
        prompt_embeds.dtype, device, generator, latents, image=init_image,
        timestep=latent_timestep, is_strength_max=is_strength_max,
        add_noise=add_noise, return_noise=True, return_image_latents=
        return_image_latents)
    if return_image_latents:
        latents, noise, image_latents = latents_outputs
    else:
        latents, noise = latents_outputs
    mask, masked_image_latents = self.prepare_mask_latents(mask,
        masked_image, batch_size * num_images_per_prompt, height, width,
        prompt_embeds.dtype, device, generator, do_classifier_free_guidance)
    if num_channels_unet == 9:
        num_channels_mask = mask.shape[1]
        num_channels_masked_image = masked_image_latents.shape[1]
        if (num_channels_latents + num_channels_mask +
            num_channels_masked_image != self.unet.config.in_channels):
            raise ValueError(
                f'Incorrect configuration settings! The config of `pipeline.unet`: {self.unet.config} expects {self.unet.config.in_channels} but received `num_channels_latents`: {num_channels_latents} + `num_channels_mask`: {num_channels_mask} + `num_channels_masked_image`: {num_channels_masked_image} = {num_channels_latents + num_channels_masked_image + num_channels_mask}. Please verify the config of `pipeline.unet` or your `mask_image` or `image` input.'
                )
    elif num_channels_unet != 4:
        raise ValueError(
            f'The unet {self.unet.__class__} should have either 4 or 9 input channels, not {self.unet.config.in_channels}.'
            )
    extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)
    if isinstance(adapter, MultiAdapter):
        adapter_state = adapter(adapter_input, adapter_conditioning_scale)
        for k, v in enumerate(adapter_state):
            adapter_state[k] = v
    else:
        adapter_state = adapter(adapter_input)
        for k, v in enumerate(adapter_state):
            adapter_state[k] = v * adapter_conditioning_scale
    if num_images_per_prompt > 1:
        for k, v in enumerate(adapter_state):
            adapter_state[k] = v.repeat(num_images_per_prompt, 1, 1, 1)
    if do_classifier_free_guidance:
        for k, v in enumerate(adapter_state):
            adapter_state[k] = torch.cat([v] * 2, dim=0)
    if isinstance(controlnet, ControlNetModel):
        control_image = self.prepare_control_image(image=control_image,
            width=width, height=height, batch_size=batch_size *
            num_images_per_prompt, num_images_per_prompt=
            num_images_per_prompt, device=device, dtype=controlnet.dtype,
            do_classifier_free_guidance=do_classifier_free_guidance,
            guess_mode=guess_mode)
    elif isinstance(controlnet, MultiControlNetModel):
        control_images = []
        for control_image_ in control_image:
            control_image_ = self.prepare_control_image(image=
                control_image_, width=width, height=height, batch_size=
                batch_size * num_images_per_prompt, num_images_per_prompt=
                num_images_per_prompt, device=device, dtype=controlnet.
                dtype, do_classifier_free_guidance=
                do_classifier_free_guidance, guess_mode=guess_mode)
            control_images.append(control_image_)
        control_image = control_images
    else:
        raise ValueError(f'{controlnet.__class__} is not supported.')
    controlnet_keep = []
    for i in range(len(timesteps)):
        keeps = [(1.0 - float(i / len(timesteps) < s or (i + 1) / len(
            timesteps) > e)) for s, e in zip(control_guidance_start,
            control_guidance_end)]
        if isinstance(self.controlnet, MultiControlNetModel):
            controlnet_keep.append(keeps)
        else:
            controlnet_keep.append(keeps[0])
    add_text_embeds = pooled_prompt_embeds
    if self.text_encoder_2 is None:
        text_encoder_projection_dim = int(pooled_prompt_embeds.shape[-1])
    else:
        text_encoder_projection_dim = self.text_encoder_2.config.projection_dim
    add_time_ids, add_neg_time_ids = self._get_add_time_ids(original_size,
        crops_coords_top_left, target_size, aesthetic_score,
        negative_aesthetic_score, dtype=prompt_embeds.dtype,
        text_encoder_projection_dim=text_encoder_projection_dim)
    add_time_ids = add_time_ids.repeat(batch_size * num_images_per_prompt, 1)
    if do_classifier_free_guidance:
        prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds],
            dim=0)
        add_text_embeds = torch.cat([negative_pooled_prompt_embeds,
            add_text_embeds], dim=0)
        add_neg_time_ids = add_neg_time_ids.repeat(batch_size *
            num_images_per_prompt, 1)
        add_time_ids = torch.cat([add_neg_time_ids, add_time_ids], dim=0)
    prompt_embeds = prompt_embeds.to(device)
    add_text_embeds = add_text_embeds.to(device)
    add_time_ids = add_time_ids.to(device)
    num_warmup_steps = max(len(timesteps) - num_inference_steps * self.
        scheduler.order, 0)
    if (denoising_end is not None and denoising_start is not None and
        denoising_value_valid(denoising_end) and denoising_value_valid(
        denoising_start) and denoising_start >= denoising_end):
        raise ValueError(
            f'`denoising_start`: {denoising_start} cannot be larger than or equal to `denoising_end`: '
             + f' {denoising_end} when using type float.')
    elif denoising_end is not None and denoising_value_valid(denoising_end):
        discrete_timestep_cutoff = int(round(self.scheduler.config.
            num_train_timesteps - denoising_end * self.scheduler.config.
            num_train_timesteps))
        num_inference_steps = len(list(filter(lambda ts: ts >=
            discrete_timestep_cutoff, timesteps)))
        timesteps = timesteps[:num_inference_steps]
    with self.progress_bar(total=num_inference_steps) as progress_bar:
        for i, t in enumerate(timesteps):
            latent_model_input = torch.cat([latents] * 2
                ) if do_classifier_free_guidance else latents
            latent_model_input = self.scheduler.scale_model_input(
                latent_model_input, t)
            if num_channels_unet == 9:
                latent_model_input = torch.cat([latent_model_input, mask,
                    masked_image_latents], dim=1)
            added_cond_kwargs = {'text_embeds': add_text_embeds, 'time_ids':
                add_time_ids}
            if i < int(num_inference_steps * cond_tau):
                down_block_additional_residuals = [state.clone() for state in
                    adapter_state]
            else:
                down_block_additional_residuals = None
            latent_model_input_controlnet = torch.cat([latents] * 2
                ) if do_classifier_free_guidance else latents
            latent_model_input_controlnet = self.scheduler.scale_model_input(
                latent_model_input_controlnet, t)
            if guess_mode and do_classifier_free_guidance:
                control_model_input = latents
                control_model_input = self.scheduler.scale_model_input(
                    control_model_input, t)
                controlnet_prompt_embeds = prompt_embeds.chunk(2)[1]
                controlnet_added_cond_kwargs = {'text_embeds':
                    add_text_embeds.chunk(2)[1], 'time_ids': add_time_ids.
                    chunk(2)[1]}
            else:
                control_model_input = latent_model_input_controlnet
                controlnet_prompt_embeds = prompt_embeds
                controlnet_added_cond_kwargs = added_cond_kwargs
            if isinstance(controlnet_keep[i], list):
                cond_scale = [(c * s) for c, s in zip(
                    controlnet_conditioning_scale, controlnet_keep[i])]
            else:
                controlnet_cond_scale = controlnet_conditioning_scale
                if isinstance(controlnet_cond_scale, list):
                    controlnet_cond_scale = controlnet_cond_scale[0]
                cond_scale = controlnet_cond_scale * controlnet_keep[i]
            down_block_res_samples, mid_block_res_sample = self.controlnet(
                control_model_input, t, encoder_hidden_states=
                controlnet_prompt_embeds, controlnet_cond=control_image,
                conditioning_scale=cond_scale, guess_mode=guess_mode,
                added_cond_kwargs=controlnet_added_cond_kwargs, return_dict
                =False)
            noise_pred = self.unet(latent_model_input, t,
                encoder_hidden_states=prompt_embeds, cross_attention_kwargs
                =cross_attention_kwargs, added_cond_kwargs=
                added_cond_kwargs, return_dict=False,
                down_intrablock_additional_residuals=
                down_block_additional_residuals,
                down_block_additional_residuals=down_block_res_samples,
                mid_block_additional_residual=mid_block_res_sample)[0]
            if do_classifier_free_guidance:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (
                    noise_pred_text - noise_pred_uncond)
            if do_classifier_free_guidance and guidance_rescale > 0.0:
                noise_pred = rescale_noise_cfg(noise_pred, noise_pred_text,
                    guidance_rescale=guidance_rescale)
            latents = self.scheduler.step(noise_pred, t, latents, **
                extra_step_kwargs, return_dict=False)[0]
            if num_channels_unet == 4:
                init_latents_proper = image_latents
                if do_classifier_free_guidance:
                    init_mask, _ = mask.chunk(2)
                else:
                    init_mask = mask
                if i < len(timesteps) - 1:
                    noise_timestep = timesteps[i + 1]
                    init_latents_proper = self.scheduler.add_noise(
                        init_latents_proper, noise, torch.tensor([
                        noise_timestep]))
                latents = (1 - init_mask
                    ) * init_latents_proper + init_mask * latents
            if i == len(timesteps) - 1 or i + 1 > num_warmup_steps and (i + 1
                ) % self.scheduler.order == 0:
                progress_bar.update()
                if callback is not None and i % callback_steps == 0:
                    callback(i, t, latents)
    if self.vae.dtype == torch.float16 and self.vae.config.force_upcast:
        self.upcast_vae()
        latents = latents.to(next(iter(self.vae.post_quant_conv.parameters(
            ))).dtype)
    if output_type != 'latent':
        image = self.vae.decode(latents / self.vae.config.scaling_factor,
            return_dict=False)[0]
    else:
        image = latents
        return StableDiffusionXLPipelineOutput(images=image)
    image = self.image_processor.postprocess(image, output_type=output_type)
    if hasattr(self, 'final_offload_hook'
        ) and self.final_offload_hook is not None:
        self.final_offload_hook.offload()
    if not return_dict:
        return image,
    return StableDiffusionXLPipelineOutput(images=image)

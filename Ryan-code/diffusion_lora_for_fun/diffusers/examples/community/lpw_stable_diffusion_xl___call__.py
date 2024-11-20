@torch.no_grad()
@replace_example_docstring(EXAMPLE_DOC_STRING)
def __call__(self, prompt: str=None, prompt_2: Optional[str]=None, image:
    Optional[PipelineImageInput]=None, mask_image: Optional[
    PipelineImageInput]=None, masked_image_latents: Optional[torch.Tensor]=
    None, height: Optional[int]=None, width: Optional[int]=None, strength:
    float=0.8, num_inference_steps: int=50, timesteps: List[int]=None,
    denoising_start: Optional[float]=None, denoising_end: Optional[float]=
    None, guidance_scale: float=5.0, negative_prompt: Optional[str]=None,
    negative_prompt_2: Optional[str]=None, num_images_per_prompt: Optional[
    int]=1, eta: float=0.0, generator: Optional[Union[torch.Generator, List
    [torch.Generator]]]=None, latents: Optional[torch.Tensor]=None,
    ip_adapter_image: Optional[PipelineImageInput]=None, prompt_embeds:
    Optional[torch.Tensor]=None, negative_prompt_embeds: Optional[torch.
    Tensor]=None, pooled_prompt_embeds: Optional[torch.Tensor]=None,
    negative_pooled_prompt_embeds: Optional[torch.Tensor]=None, output_type:
    Optional[str]='pil', return_dict: bool=True, cross_attention_kwargs:
    Optional[Dict[str, Any]]=None, guidance_rescale: float=0.0,
    original_size: Optional[Tuple[int, int]]=None, crops_coords_top_left:
    Tuple[int, int]=(0, 0), target_size: Optional[Tuple[int, int]]=None,
    clip_skip: Optional[int]=None, callback_on_step_end: Optional[Callable[
    [int, int, Dict], None]]=None, callback_on_step_end_tensor_inputs: List
    [str]=['latents'], **kwargs):
    """
        Function invoked when calling the pipeline for generation.

        Args:
            prompt (`str`):
                The prompt  to guide the image generation. If not defined, one has to pass `prompt_embeds`.
                instead.
            prompt_2 (`str`):
                The prompt to be sent to the `tokenizer_2` and `text_encoder_2`. If not defined, `prompt` is
                used in both text-encoders
            image (`PipelineImageInput`, *optional*):
                `Image`, or tensor representing an image batch, that will be used as the starting point for the
                process.
            mask_image (`PipelineImageInput`, *optional*):
                `Image`, or tensor representing an image batch, to mask `image`. White pixels in the mask will be
                replaced by noise and therefore repainted, while black pixels will be preserved. If `mask_image` is a
                PIL image, it will be converted to a single channel (luminance) before use. If it's a tensor, it should
                contain one color channel (L) instead of 3, so the expected shape would be `(B, H, W, 1)`.
            height (`int`, *optional*, defaults to self.unet.config.sample_size * self.vae_scale_factor):
                The height in pixels of the generated image.
            width (`int`, *optional*, defaults to self.unet.config.sample_size * self.vae_scale_factor):
                The width in pixels of the generated image.
            strength (`float`, *optional*, defaults to 0.8):
                Conceptually, indicates how much to transform the reference `image`. Must be between 0 and 1.
                `image` will be used as a starting point, adding more noise to it the larger the `strength`. The
                number of denoising steps depends on the amount of noise initially added. When `strength` is 1, added
                noise will be maximum and the denoising process will run for the full number of iterations specified in
                `num_inference_steps`. A value of 1, therefore, essentially ignores `image`.
            num_inference_steps (`int`, *optional*, defaults to 50):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            timesteps (`List[int]`, *optional*):
                Custom timesteps to use for the denoising process with schedulers which support a `timesteps` argument
                in their `set_timesteps` method. If not defined, the default behavior when `num_inference_steps` is
                passed will be used. Must be in descending order.
            denoising_start (`float`, *optional*):
                When specified, indicates the fraction (between 0.0 and 1.0) of the total denoising process to be
                bypassed before it is initiated. Consequently, the initial part of the denoising process is skipped and
                it is assumed that the passed `image` is a partly denoised image. Note that when this is specified,
                strength will be ignored. The `denoising_start` parameter is particularly beneficial when this pipeline
                is integrated into a "Mixture of Denoisers" multi-pipeline setup, as detailed in [**Refine Image
                Quality**](https://huggingface.co/docs/diffusers/using-diffusers/sdxl#refine-image-quality).
            denoising_end (`float`, *optional*):
                When specified, determines the fraction (between 0.0 and 1.0) of the total denoising process to be
                completed before it is intentionally prematurely terminated. As a result, the returned sample will
                still retain a substantial amount of noise (ca. final 20% of timesteps still needed) and should be
                denoised by a successor pipeline that has `denoising_start` set to 0.8 so that it only denoises the
                final 20% of the scheduler. The denoising_end parameter should ideally be utilized when this pipeline
                forms a part of a "Mixture of Denoisers" multi-pipeline setup, as elaborated in [**Refine Image
                Quality**](https://huggingface.co/docs/diffusers/using-diffusers/sdxl#refine-image-quality).
            guidance_scale (`float`, *optional*, defaults to 5.0):
                Guidance scale as defined in [Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598).
                `guidance_scale` is defined as `w` of equation 2. of [Imagen
                Paper](https://arxiv.org/pdf/2205.11487.pdf). Guidance scale is enabled by setting `guidance_scale >
                1`. Higher guidance scale encourages to generate images that are closely linked to the text `prompt`,
                usually at the expense of lower image quality.
            negative_prompt (`str`):
                The prompt not to guide the image generation. If not defined, one has to pass
                `negative_prompt_embeds` instead. Ignored when not using guidance (i.e., ignored if `guidance_scale` is
                less than `1`).
            negative_prompt_2 (`str`):
                The prompt not to guide the image generation to be sent to `tokenizer_2` and
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
            ip_adapter_image: (`PipelineImageInput`, *optional*):
                Optional image input to work with IP Adapters.
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
                Whether or not to return a [`~pipelines.stable_diffusion_xl.StableDiffusionXLPipelineOutput`] instead
                of a plain tuple.
            cross_attention_kwargs (`dict`, *optional*):
                A kwargs dictionary that if specified is passed along to the `AttentionProcessor` as defined under
                `self.processor` in
                [diffusers.models.attention_processor](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py).
            guidance_rescale (`float`, *optional*, defaults to 0.0):
                Guidance rescale factor proposed by [Common Diffusion Noise Schedules and Sample Steps are
                Flawed](https://arxiv.org/pdf/2305.08891.pdf) `guidance_scale` is defined as `φ` in equation 16. of
                [Common Diffusion Noise Schedules and Sample Steps are Flawed](https://arxiv.org/pdf/2305.08891.pdf).
                Guidance rescale factor should fix overexposure when using zero terminal SNR.
            original_size (`Tuple[int]`, *optional*, defaults to (1024, 1024)):
                If `original_size` is not the same as `target_size` the image will appear to be down- or upsampled.
                `original_size` defaults to `(height, width)` if not specified. Part of SDXL's micro-conditioning as
                explained in section 2.2 of
                [https://huggingface.co/papers/2307.01952](https://huggingface.co/papers/2307.01952).
            crops_coords_top_left (`Tuple[int]`, *optional*, defaults to (0, 0)):
                `crops_coords_top_left` can be used to generate an image that appears to be "cropped" from the position
                `crops_coords_top_left` downwards. Favorable, well-centered images are usually achieved by setting
                `crops_coords_top_left` to (0, 0). Part of SDXL's micro-conditioning as explained in section 2.2 of
                [https://huggingface.co/papers/2307.01952](https://huggingface.co/papers/2307.01952).
            target_size (`Tuple[int]`, *optional*, defaults to (1024, 1024)):
                For most cases, `target_size` should be set to the desired height and width of the generated image. If
                not specified it will default to `(height, width)`. Part of SDXL's micro-conditioning as explained in
                section 2.2 of [https://huggingface.co/papers/2307.01952](https://huggingface.co/papers/2307.01952).
            clip_skip (`int`, *optional*):
                Number of layers to be skipped from CLIP while computing the prompt embeddings. A value of 1 means that
                the output of the pre-final layer will be used for computing the prompt embeddings.
            callback_on_step_end (`Callable`, *optional*):
                A function that calls at the end of each denoising steps during the inference. The function is called
                with the following arguments: `callback_on_step_end(self: DiffusionPipeline, step: int, timestep: int,
                callback_kwargs: Dict)`. `callback_kwargs` will include a list of all tensors as specified by
                `callback_on_step_end_tensor_inputs`.
            callback_on_step_end_tensor_inputs (`List`, *optional*):
                The list of tensor inputs for the `callback_on_step_end` function. The tensors specified in the list
                will be passed as `callback_kwargs` argument. You will only be able to include variables listed in the
                `._callback_tensor_inputs` attribute of your pipeline class.

        Examples:

        Returns:
            [`~pipelines.stable_diffusion_xl.StableDiffusionXLPipelineOutput`] or `tuple`:
            [`~pipelines.stable_diffusion_xl.StableDiffusionXLPipelineOutput`] if `return_dict` is True, otherwise a
            `tuple`. When returning a tuple, the first element is a list with the generated images.
        """
    callback = kwargs.pop('callback', None)
    callback_steps = kwargs.pop('callback_steps', None)
    if callback is not None:
        deprecate('callback', '1.0.0',
            'Passing `callback` as an input argument to `__call__` is deprecated, consider using `callback_on_step_end`'
            )
    if callback_steps is not None:
        deprecate('callback_steps', '1.0.0',
            'Passing `callback_steps` as an input argument to `__call__` is deprecated, consider using `callback_on_step_end`'
            )
    height = height or self.default_sample_size * self.vae_scale_factor
    width = width or self.default_sample_size * self.vae_scale_factor
    original_size = original_size or (height, width)
    target_size = target_size or (height, width)
    self.check_inputs(prompt, prompt_2, height, width, strength,
        callback_steps, negative_prompt, negative_prompt_2, prompt_embeds,
        negative_prompt_embeds, pooled_prompt_embeds,
        negative_pooled_prompt_embeds, callback_on_step_end_tensor_inputs)
    self._guidance_scale = guidance_scale
    self._guidance_rescale = guidance_rescale
    self._clip_skip = clip_skip
    self._cross_attention_kwargs = cross_attention_kwargs
    self._denoising_end = denoising_end
    self._denoising_start = denoising_start
    if prompt is not None and isinstance(prompt, str):
        batch_size = 1
    elif prompt is not None and isinstance(prompt, list):
        batch_size = len(prompt)
    else:
        batch_size = prompt_embeds.shape[0]
    device = self._execution_device
    if ip_adapter_image is not None:
        output_hidden_state = False if isinstance(self.unet.
            encoder_hid_proj, ImageProjection) else True
        image_embeds, negative_image_embeds = self.encode_image(
            ip_adapter_image, device, num_images_per_prompt,
            output_hidden_state)
        if self.do_classifier_free_guidance:
            image_embeds = torch.cat([negative_image_embeds, image_embeds])
    self.cross_attention_kwargs.get('scale', None
        ) if self.cross_attention_kwargs is not None else None
    negative_prompt = negative_prompt if negative_prompt is not None else ''
    (prompt_embeds, negative_prompt_embeds, pooled_prompt_embeds,
        negative_pooled_prompt_embeds) = (get_weighted_text_embeddings_sdxl
        (pipe=self, prompt=prompt, neg_prompt=negative_prompt,
        num_images_per_prompt=num_images_per_prompt, clip_skip=clip_skip))
    dtype = prompt_embeds.dtype
    if isinstance(image, Image.Image):
        image = self.image_processor.preprocess(image, height=height, width
            =width)
    if image is not None:
        image = image.to(device=self.device, dtype=dtype)
    if isinstance(mask_image, Image.Image):
        mask = self.mask_processor.preprocess(mask_image, height=height,
            width=width)
    else:
        mask = mask_image
    if mask_image is not None:
        mask = mask.to(device=self.device, dtype=dtype)
        if masked_image_latents is not None:
            masked_image = masked_image_latents
        elif image.shape[1] == 4:
            masked_image = None
        else:
            masked_image = image * (mask < 0.5)
    else:
        mask = None

    def denoising_value_valid(dnv):
        return isinstance(dnv, float) and 0 < dnv < 1
    timesteps, num_inference_steps = retrieve_timesteps(self.scheduler,
        num_inference_steps, device, timesteps)
    if image is not None:
        timesteps, num_inference_steps = self.get_timesteps(num_inference_steps
            , strength, device, denoising_start=self.denoising_start if
            denoising_value_valid(self.denoising_start) else None)
        if num_inference_steps < 1:
            raise ValueError(
                f'After adjusting the num_inference_steps by strength parameter: {strength}, the number of pipelinesteps is {num_inference_steps} which is < 1 and not appropriate for this pipeline.'
                )
    latent_timestep = timesteps[:1].repeat(batch_size * num_images_per_prompt)
    is_strength_max = strength == 1.0
    add_noise = True if self.denoising_start is None else False
    num_channels_latents = self.vae.config.latent_channels
    num_channels_unet = self.unet.config.in_channels
    return_image_latents = num_channels_unet == 4
    latents = self.prepare_latents(image=image, mask=mask, width=width,
        height=height, num_channels_latents=num_channels_unet, timestep=
        latent_timestep, batch_size=batch_size, num_images_per_prompt=
        num_images_per_prompt, dtype=prompt_embeds.dtype, device=device,
        generator=generator, add_noise=add_noise, latents=latents,
        is_strength_max=is_strength_max, return_noise=True,
        return_image_latents=return_image_latents)
    if mask is not None:
        if return_image_latents:
            latents, noise, image_latents = latents
        else:
            latents, noise = latents
    if mask is not None:
        mask, masked_image_latents = self.prepare_mask_latents(mask=mask,
            masked_image=masked_image, batch_size=batch_size *
            num_images_per_prompt, height=height, width=width, dtype=
            prompt_embeds.dtype, device=device, generator=generator,
            do_classifier_free_guidance=self.do_classifier_free_guidance)
        if num_channels_unet == 9:
            num_channels_mask = mask.shape[1]
            num_channels_masked_image = masked_image_latents.shape[1]
            if (num_channels_latents + num_channels_mask +
                num_channels_masked_image != num_channels_unet):
                raise ValueError(
                    f'Incorrect configuration settings! The config of `pipeline.unet`: {self.unet.config} expects {self.unet.config.in_channels} but received `num_channels_latents`: {num_channels_latents} + `num_channels_mask`: {num_channels_mask} + `num_channels_masked_image`: {num_channels_masked_image} = {num_channels_latents + num_channels_masked_image + num_channels_mask}. Please verify the config of `pipeline.unet` or your `mask_image` or `image` input.'
                    )
        elif num_channels_unet != 4:
            raise ValueError(
                f'The unet {self.unet.__class__} should have either 4 or 9 input channels, not {self.unet.config.in_channels}.'
                )
    extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)
    added_cond_kwargs = {'image_embeds': image_embeds
        } if ip_adapter_image is not None else {}
    height, width = latents.shape[-2:]
    height = height * self.vae_scale_factor
    width = width * self.vae_scale_factor
    original_size = original_size or (height, width)
    target_size = target_size or (height, width)
    add_text_embeds = pooled_prompt_embeds
    add_time_ids = self._get_add_time_ids(original_size,
        crops_coords_top_left, target_size, dtype=prompt_embeds.dtype)
    if self.do_classifier_free_guidance:
        prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds],
            dim=0)
        add_text_embeds = torch.cat([negative_pooled_prompt_embeds,
            add_text_embeds], dim=0)
        add_time_ids = torch.cat([add_time_ids, add_time_ids], dim=0)
    prompt_embeds = prompt_embeds.to(device)
    add_text_embeds = add_text_embeds.to(device)
    add_time_ids = add_time_ids.to(device).repeat(batch_size *
        num_images_per_prompt, 1)
    num_warmup_steps = max(len(timesteps) - num_inference_steps * self.
        scheduler.order, 0)
    if (self.denoising_end is not None and self.denoising_start is not None and
        denoising_value_valid(self.denoising_end) and denoising_value_valid
        (self.denoising_start) and self.denoising_start >= self.denoising_end):
        raise ValueError(
            f'`denoising_start`: {self.denoising_start} cannot be larger than or equal to `denoising_end`: '
             + f' {self.denoising_end} when using type float.')
    elif self.denoising_end is not None and denoising_value_valid(self.
        denoising_end):
        discrete_timestep_cutoff = int(round(self.scheduler.config.
            num_train_timesteps - self.denoising_end * self.scheduler.
            config.num_train_timesteps))
        num_inference_steps = len(list(filter(lambda ts: ts >=
            discrete_timestep_cutoff, timesteps)))
        timesteps = timesteps[:num_inference_steps]
    timestep_cond = None
    if self.unet.config.time_cond_proj_dim is not None:
        guidance_scale_tensor = torch.tensor(self.guidance_scale - 1).repeat(
            batch_size * num_images_per_prompt)
        timestep_cond = self.get_guidance_scale_embedding(guidance_scale_tensor
            , embedding_dim=self.unet.config.time_cond_proj_dim).to(device=
            device, dtype=latents.dtype)
    self._num_timesteps = len(timesteps)
    with self.progress_bar(total=num_inference_steps) as progress_bar:
        for i, t in enumerate(timesteps):
            latent_model_input = torch.cat([latents] * 2
                ) if self.do_classifier_free_guidance else latents
            latent_model_input = self.scheduler.scale_model_input(
                latent_model_input, t)
            if mask is not None and num_channels_unet == 9:
                latent_model_input = torch.cat([latent_model_input, mask,
                    masked_image_latents], dim=1)
            added_cond_kwargs.update({'text_embeds': add_text_embeds,
                'time_ids': add_time_ids})
            noise_pred = self.unet(latent_model_input, t,
                encoder_hidden_states=prompt_embeds, timestep_cond=
                timestep_cond, cross_attention_kwargs=self.
                cross_attention_kwargs, added_cond_kwargs=added_cond_kwargs,
                return_dict=False)[0]
            if self.do_classifier_free_guidance:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + self.guidance_scale * (
                    noise_pred_text - noise_pred_uncond)
            if self.do_classifier_free_guidance and guidance_rescale > 0.0:
                noise_pred = rescale_noise_cfg(noise_pred, noise_pred_text,
                    guidance_rescale=guidance_rescale)
            latents = self.scheduler.step(noise_pred, t, latents, **
                extra_step_kwargs, return_dict=False)[0]
            if mask is not None and num_channels_unet == 4:
                init_latents_proper = image_latents
                if self.do_classifier_free_guidance:
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
                add_text_embeds = callback_outputs.pop('add_text_embeds',
                    add_text_embeds)
                negative_pooled_prompt_embeds = callback_outputs.pop(
                    'negative_pooled_prompt_embeds',
                    negative_pooled_prompt_embeds)
                add_time_ids = callback_outputs.pop('add_time_ids',
                    add_time_ids)
            if i == len(timesteps) - 1 or i + 1 > num_warmup_steps and (i + 1
                ) % self.scheduler.order == 0:
                progress_bar.update()
                if callback is not None and i % callback_steps == 0:
                    step_idx = i // getattr(self.scheduler, 'order', 1)
                    callback(step_idx, t, latents)
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
        return StableDiffusionXLPipelineOutput(images=image)
    if self.watermark is not None:
        image = self.watermark.apply_watermark(image)
    image = self.image_processor.postprocess(image, output_type=output_type)
    if hasattr(self, 'final_offload_hook'
        ) and self.final_offload_hook is not None:
        self.final_offload_hook.offload()
    if not return_dict:
        return image,
    return StableDiffusionXLPipelineOutput(images=image)

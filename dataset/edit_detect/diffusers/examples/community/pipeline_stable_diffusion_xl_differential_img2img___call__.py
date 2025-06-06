@torch.no_grad()
@replace_example_docstring(EXAMPLE_DOC_STRING)
def __call__(self, prompt: Union[str, List[str]]=None, prompt_2: Optional[
    Union[str, List[str]]]=None, image: Union[torch.Tensor, PIL.Image.Image,
    np.ndarray, List[torch.Tensor], List[PIL.Image.Image], List[np.ndarray]
    ]=None, strength: float=0.3, num_inference_steps: int=50, timesteps:
    List[int]=None, denoising_start: Optional[float]=None, denoising_end:
    Optional[float]=None, guidance_scale: float=5.0, negative_prompt:
    Optional[Union[str, List[str]]]=None, negative_prompt_2: Optional[Union
    [str, List[str]]]=None, num_images_per_prompt: Optional[int]=1, eta:
    float=0.0, generator: Optional[Union[torch.Generator, List[torch.
    Generator]]]=None, latents: Optional[torch.Tensor]=None, prompt_embeds:
    Optional[torch.Tensor]=None, negative_prompt_embeds: Optional[torch.
    Tensor]=None, pooled_prompt_embeds: Optional[torch.Tensor]=None,
    negative_pooled_prompt_embeds: Optional[torch.Tensor]=None,
    ip_adapter_image: Optional[PipelineImageInput]=None,
    ip_adapter_image_embeds: Optional[List[torch.Tensor]]=None, output_type:
    Optional[str]='pil', return_dict: bool=True, cross_attention_kwargs:
    Optional[Dict[str, Any]]=None, guidance_rescale: float=0.0,
    original_size: Tuple[int, int]=None, crops_coords_top_left: Tuple[int,
    int]=(0, 0), target_size: Tuple[int, int]=None, negative_original_size:
    Optional[Tuple[int, int]]=None, negative_crops_coords_top_left: Tuple[
    int, int]=(0, 0), negative_target_size: Optional[Tuple[int, int]]=None,
    aesthetic_score: float=6.0, negative_aesthetic_score: float=2.5,
    clip_skip: Optional[int]=None, callback_on_step_end: Optional[Callable[
    [int, int, Dict], None]]=None, callback_on_step_end_tensor_inputs: List
    [str]=['latents'], map: torch.Tensor=None, original_image: Union[torch.
    Tensor, PIL.Image.Image, np.ndarray, List[torch.Tensor], List[PIL.Image
    .Image], List[np.ndarray]]=None, **kwargs):
    """
        Function invoked when calling the pipeline for generation.

        Args:
            prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts to guide the image generation. If not defined, one has to pass `prompt_embeds`.
                instead.
            prompt_2 (`str` or `List[str]`, *optional*):
                The prompt or prompts to be sent to the `tokenizer_2` and `text_encoder_2`. If not defined, `prompt` is
                used in both text-encoders
            image (`torch.Tensor` or `PIL.Image.Image` or `np.ndarray` or `List[torch.Tensor]` or `List[PIL.Image.Image]` or `List[np.ndarray]`):
                The image(s) to modify with the pipeline.
            strength (`float`, *optional*, defaults to 0.3):
                Conceptually, indicates how much to transform the reference `image`. Must be between 0 and 1. `image`
                will be used as a starting point, adding more noise to it the larger the `strength`. The number of
                denoising steps depends on the amount of noise initially added. When `strength` is 1, added noise will
                be maximum and the denoising process will run for the full number of iterations specified in
                `num_inference_steps`. A value of 1, therefore, essentially ignores `image`. Note that in the case of
                `denoising_start` being declared as an integer, the value of `strength` will be ignored.
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
                still retain a substantial amount of noise (ca. final 20% of timesteps still needed) and should be
                denoised by a successor pipeline that has `denoising_start` set to 0.8 so that it only denoises the
                final 20% of the scheduler. The denoising_end parameter should ideally be utilized when this pipeline
                forms a part of a "Mixture of Denoisers" multi-pipeline setup, as elaborated in [**Refining the Image
                Output**](https://huggingface.co/docs/diffusers/api/pipelines/stable_diffusion/stable_diffusion_xl#refining-the-image-output).
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
            ip_adapter_image: (`PipelineImageInput`, *optional*): Optional image input to work with IP Adapters.
            ip_adapter_image_embeds (`List[torch.Tensor]`, *optional*):
                Pre-generated image embeddings for IP-Adapter. It should be a list of length same as number of IP-adapters.
                Each element should be a tensor of shape `(batch_size, num_images, emb_dim)`. It should contain the negative image embedding
                if `do_classifier_free_guidance` is set to `True`.
                If not provided, embeddings are computed from the `ip_adapter_image` input argument.
            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generate image. Choose between
                [PIL](https://pillow.readthedocs.io/en/stable/): `PIL.Image.Image` or `np.array`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~pipelines.stable_diffusion.StableDiffusionXLPipelineOutput`] instead of a
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
            [`~pipelines.stable_diffusion.StableDiffusionXLPipelineOutput`] or `tuple`:
            [`~pipelines.stable_diffusion.StableDiffusionXLPipelineOutput`] if `return_dict` is True, otherwise a
            `tuple. When returning a tuple, the first element is a list with the generated images.
        """
    callback = kwargs.pop('callback', None)
    callback_steps = kwargs.pop('callback_steps', None)
    if callback is not None:
        deprecate('callback', '1.0.0',
            'Passing `callback` as an input argument to `__call__` is deprecated, consider use `callback_on_step_end`'
            )
    if callback_steps is not None:
        deprecate('callback_steps', '1.0.0',
            'Passing `callback_steps` as an input argument to `__call__` is deprecated, consider use `callback_on_step_end`'
            )
    self.check_inputs(prompt, prompt_2, strength, num_inference_steps,
        callback_steps, negative_prompt, negative_prompt_2, prompt_embeds,
        negative_prompt_embeds, ip_adapter_image, ip_adapter_image_embeds,
        callback_on_step_end_tensor_inputs)
    self._guidance_scale = guidance_scale
    self._guidance_rescale = guidance_rescale
    self._clip_skip = clip_skip
    self._cross_attention_kwargs = cross_attention_kwargs
    self._denoising_end = denoising_end
    self._denoising_start = denoising_start
    self._interrupt = False
    if prompt is not None and isinstance(prompt, str):
        batch_size = 1
    elif prompt is not None and isinstance(prompt, list):
        batch_size = len(prompt)
    else:
        batch_size = prompt_embeds.shape[0]
    device = self._execution_device
    text_encoder_lora_scale = cross_attention_kwargs.get('scale', None
        ) if cross_attention_kwargs is not None else None
    (prompt_embeds, negative_prompt_embeds, pooled_prompt_embeds,
        negative_pooled_prompt_embeds) = (self.encode_prompt(prompt=prompt,
        prompt_2=prompt_2, device=device, num_images_per_prompt=
        num_images_per_prompt, do_classifier_free_guidance=self.
        do_classifier_free_guidance, negative_prompt=negative_prompt,
        negative_prompt_2=negative_prompt_2, prompt_embeds=prompt_embeds,
        negative_prompt_embeds=negative_prompt_embeds, pooled_prompt_embeds
        =pooled_prompt_embeds, negative_pooled_prompt_embeds=
        negative_pooled_prompt_embeds, lora_scale=text_encoder_lora_scale))
    map = torchvision.transforms.Resize(tuple(s // self.vae_scale_factor for
        s in original_image.shape[2:]), antialias=None)(map)

    def denoising_value_valid(dnv):
        return isinstance(dnv, float) and 0 < dnv < 1
    timesteps, num_inference_steps = retrieve_timesteps(self.scheduler,
        num_inference_steps, device, timesteps)
    total_time_steps = num_inference_steps
    timesteps, num_inference_steps = self.get_timesteps(num_inference_steps,
        strength, device, denoising_start=self.denoising_start if
        denoising_value_valid(self.denoising_start) else None)
    latent_timestep = timesteps[:1].repeat(batch_size * num_images_per_prompt)
    add_noise = True if denoising_start is None else False
    latents = self.prepare_latents(image, latent_timestep, batch_size,
        num_images_per_prompt, prompt_embeds.dtype, device, generator,
        add_noise)
    extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)
    height, width = latents.shape[-2:]
    height = height * self.vae_scale_factor
    width = width * self.vae_scale_factor
    original_size = original_size or (height, width)
    target_size = target_size or (height, width)
    if negative_original_size is None:
        negative_original_size = original_size
    if negative_target_size is None:
        negative_target_size = target_size
    add_text_embeds = pooled_prompt_embeds
    if self.text_encoder_2 is None:
        text_encoder_projection_dim = int(pooled_prompt_embeds.shape[-1])
    else:
        text_encoder_projection_dim = self.text_encoder_2.config.projection_dim
    add_time_ids, add_neg_time_ids = self._get_add_time_ids(original_size,
        crops_coords_top_left, target_size, aesthetic_score,
        negative_aesthetic_score, negative_original_size,
        negative_crops_coords_top_left, negative_target_size, dtype=
        prompt_embeds.dtype, text_encoder_projection_dim=
        text_encoder_projection_dim)
    add_time_ids = add_time_ids.repeat(batch_size * num_images_per_prompt, 1)
    if self.do_classifier_free_guidance:
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
    if ip_adapter_image is not None or ip_adapter_image_embeds is not None:
        image_embeds = self.prepare_ip_adapter_image_embeds(ip_adapter_image,
            ip_adapter_image_embeds, device, batch_size *
            num_images_per_prompt, self.do_classifier_free_guidance)
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
    original_with_noise = self.prepare_latents(original_image, timesteps,
        batch_size, num_images_per_prompt, prompt_embeds.dtype, device,
        generator)
    thresholds = torch.arange(total_time_steps, dtype=map.dtype
        ) / total_time_steps
    thresholds = thresholds.unsqueeze(1).unsqueeze(1).to(device)
    masks = map > thresholds + (denoising_start or 0)
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
            if self.interrupt:
                continue
            if i == 0 and denoising_start is None:
                latents = original_with_noise[:1]
            else:
                mask = masks[i].unsqueeze(0)
                mask = mask.to(latents.dtype)
                mask = mask.unsqueeze(1)
                latents = original_with_noise[i] * mask + latents * (1 - mask)
            latent_model_input = torch.cat([latents] * 2
                ) if self.do_classifier_free_guidance else latents
            latent_model_input = self.scheduler.scale_model_input(
                latent_model_input, t)
            added_cond_kwargs = {'text_embeds': add_text_embeds, 'time_ids':
                add_time_ids}
            if (ip_adapter_image is not None or ip_adapter_image_embeds is not
                None):
                added_cond_kwargs['image_embeds'] = image_embeds
            noise_pred = self.unet(latent_model_input, t,
                encoder_hidden_states=prompt_embeds, timestep_cond=
                timestep_cond, cross_attention_kwargs=
                cross_attention_kwargs, added_cond_kwargs=added_cond_kwargs,
                return_dict=False)[0]
            if self.do_classifier_free_guidance:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (
                    noise_pred_text - noise_pred_uncond)
            if self.do_classifier_free_guidance and guidance_rescale > 0.0:
                noise_pred = rescale_noise_cfg(noise_pred, noise_pred_text,
                    guidance_rescale=guidance_rescale)
            latents_dtype = latents.dtype
            latents = self.scheduler.step(noise_pred, t, latents, **
                extra_step_kwargs, return_dict=False)[0]
            if latents.dtype != latents_dtype:
                if torch.backends.mps.is_available():
                    latents = latents.to(latents_dtype)
                else:
                    raise ValueError(
                        'For the given accelerator, there seems to be an unexpected problem in type-casting. Please file an issue on the PyTorch GitHub repository. See also: https://github.com/huggingface/diffusers/pull/7446/.'
                        )
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
                add_neg_time_ids = callback_outputs.pop('add_neg_time_ids',
                    add_neg_time_ids)
            if i == len(timesteps) - 1 or i + 1 > num_warmup_steps and (i + 1
                ) % self.scheduler.order == 0:
                progress_bar.update()
                if callback is not None and i % callback_steps == 0:
                    step_idx = i // getattr(self.scheduler, 'order', 1)
                    callback(step_idx, t, latents)
            if XLA_AVAILABLE:
                xm.mark_step()
    if not output_type == 'latent':
        needs_upcasting = (self.vae.dtype == torch.float16 and self.vae.
            config.force_upcast)
        if needs_upcasting:
            self.upcast_vae()
            latents = latents.to(next(iter(self.vae.post_quant_conv.
                parameters())).dtype)
        elif latents.dtype != self.vae.dtype:
            if torch.backends.mps.is_available():
                self.vae = self.vae.to(latents.dtype)
            else:
                raise ValueError(
                    'For the given accelerator, there seems to be an unexpected problem in type-casting. Please file an issue on the PyTorch GitHub repository. See also: https://github.com/huggingface/diffusers/pull/7446/.'
                    )
        has_latents_mean = hasattr(self.vae.config, 'latents_mean'
            ) and self.vae.config.latents_mean is not None
        has_latents_std = hasattr(self.vae.config, 'latents_std'
            ) and self.vae.config.latents_std is not None
        if has_latents_mean and has_latents_std:
            latents_mean = torch.tensor(self.vae.config.latents_mean).view(
                1, 4, 1, 1).to(latents.device, latents.dtype)
            latents_std = torch.tensor(self.vae.config.latents_std).view(1,
                4, 1, 1).to(latents.device, latents.dtype)
            latents = (latents * latents_std / self.vae.config.
                scaling_factor + latents_mean)
        else:
            latents = latents / self.vae.config.scaling_factor
        image = self.vae.decode(latents, return_dict=False)[0]
        if needs_upcasting:
            self.vae.to(dtype=torch.float16)
    else:
        image = latents
    if self.watermark is not None:
        image = self.watermark.apply_watermark(image)
    image = self.image_processor.postprocess(image, output_type=output_type)
    self.maybe_free_model_hooks()
    if not return_dict:
        return image,
    return StableDiffusionXLPipelineOutput(images=image)

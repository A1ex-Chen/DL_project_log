@torch.no_grad()
@replace_example_docstring(EXAMPLE_DOC_STRING)
def __call__(self, prompt: Union[str, List[str]]=None, height: Optional[int
    ]=None, width: Optional[int]=None, num_inference_steps: int=50,
    parallel: int=10, tolerance: float=0.1, guidance_scale: float=7.5,
    negative_prompt: Optional[Union[str, List[str]]]=None,
    num_images_per_prompt: Optional[int]=1, eta: float=0.0, generator:
    Optional[Union[torch.Generator, List[torch.Generator]]]=None, latents:
    Optional[torch.Tensor]=None, prompt_embeds: Optional[torch.Tensor]=None,
    negative_prompt_embeds: Optional[torch.Tensor]=None, output_type:
    Optional[str]='pil', return_dict: bool=True, callback: Optional[
    Callable[[int, int, torch.Tensor], None]]=None, callback_steps: int=1,
    cross_attention_kwargs: Optional[Dict[str, Any]]=None, debug: bool=
    False, clip_skip: int=None):
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
            parallel (`int`, *optional*, defaults to 10):
                The batch size to use when doing parallel sampling. More parallelism may lead to faster inference but
                requires higher memory usage and can also require more total FLOPs.
            tolerance (`float`, *optional*, defaults to 0.1):
                The error tolerance for determining when to slide the batch window forward for parallel sampling. Lower
                tolerance usually leads to less or no degradation. Higher tolerance is faster but can risk degradation
                of sample quality. The tolerance is specified as a ratio of the scheduler's noise magnitude.
            guidance_scale (`float`, *optional*, defaults to 7.5):
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
            debug (`bool`, *optional*, defaults to `False`):
                Whether or not to run in debug mode. In debug mode, `torch.cumsum` is evaluated using the CPU.
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
    num_channels_latents = self.unet.config.in_channels
    latents = self.prepare_latents(batch_size * num_images_per_prompt,
        num_channels_latents, height, width, prompt_embeds.dtype, device,
        generator, latents)
    extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)
    extra_step_kwargs.pop('generator', None)
    scheduler = self.scheduler
    parallel = min(parallel, len(scheduler.timesteps))
    begin_idx = 0
    end_idx = parallel
    latents_time_evolution_buffer = torch.stack([latents] * (len(scheduler.
        timesteps) + 1))
    noise_array = torch.zeros_like(latents_time_evolution_buffer)
    for j in range(len(scheduler.timesteps)):
        base_noise = randn_tensor(shape=latents.shape, generator=generator,
            device=latents.device, dtype=prompt_embeds.dtype)
        noise = self.scheduler._get_variance(scheduler.timesteps[j]
            ) ** 0.5 * base_noise
        noise_array[j] = noise.clone()
    inverse_variance_norm = 1.0 / torch.tensor([scheduler._get_variance(
        scheduler.timesteps[j]) for j in range(len(scheduler.timesteps))] + [0]
        ).to(noise_array.device)
    latent_dim = noise_array[0, 0].numel()
    inverse_variance_norm = inverse_variance_norm[:, None] / latent_dim
    scaled_tolerance = tolerance ** 2
    with self.progress_bar(total=num_inference_steps) as progress_bar:
        steps = 0
        while begin_idx < len(scheduler.timesteps):
            parallel_len = end_idx - begin_idx
            block_prompt_embeds = torch.stack([prompt_embeds] * parallel_len)
            block_latents = latents_time_evolution_buffer[begin_idx:end_idx]
            block_t = scheduler.timesteps[begin_idx:end_idx, None].repeat(1,
                batch_size * num_images_per_prompt)
            t_vec = block_t
            if do_classifier_free_guidance:
                t_vec = t_vec.repeat(1, 2)
            latent_model_input = torch.cat([block_latents] * 2, dim=1
                ) if do_classifier_free_guidance else block_latents
            latent_model_input = self.scheduler.scale_model_input(
                latent_model_input, t_vec)
            net = self.wrapped_unet if parallel_len > 3 else self.unet
            model_output = net(latent_model_input.flatten(0, 1), t_vec.
                flatten(0, 1), encoder_hidden_states=block_prompt_embeds.
                flatten(0, 1), cross_attention_kwargs=
                cross_attention_kwargs, return_dict=False)[0]
            per_latent_shape = model_output.shape[1:]
            if do_classifier_free_guidance:
                model_output = model_output.reshape(parallel_len, 2, 
                    batch_size * num_images_per_prompt, *per_latent_shape)
                noise_pred_uncond, noise_pred_text = model_output[:, 0
                    ], model_output[:, 1]
                model_output = noise_pred_uncond + guidance_scale * (
                    noise_pred_text - noise_pred_uncond)
            model_output = model_output.reshape(parallel_len * batch_size *
                num_images_per_prompt, *per_latent_shape)
            block_latents_denoise = scheduler.batch_step_no_noise(model_output
                =model_output, timesteps=block_t.flatten(0, 1), sample=
                block_latents.flatten(0, 1), **extra_step_kwargs).reshape(
                block_latents.shape)
            delta = block_latents_denoise - block_latents
            cumulative_delta = self._cumsum(delta, dim=0, debug=debug)
            cumulative_noise = self._cumsum(noise_array[begin_idx:end_idx],
                dim=0, debug=debug)
            if scheduler._is_ode_scheduler:
                cumulative_noise = 0
            block_latents_new = latents_time_evolution_buffer[begin_idx][None,
                ] + cumulative_delta + cumulative_noise
            cur_error = torch.linalg.norm((block_latents_new -
                latents_time_evolution_buffer[begin_idx + 1:end_idx + 1]).
                reshape(parallel_len, batch_size * num_images_per_prompt, -
                1), dim=-1).pow(2)
            error_ratio = cur_error * inverse_variance_norm[begin_idx + 1:
                end_idx + 1]
            error_ratio = torch.nn.functional.pad(error_ratio, (0, 0, 0, 1),
                value=1000000000.0)
            any_error_at_time = torch.max(error_ratio > scaled_tolerance, dim=1
                ).values.int()
            ind = torch.argmax(any_error_at_time).item()
            new_begin_idx = begin_idx + min(1 + ind, parallel)
            new_end_idx = min(new_begin_idx + parallel, len(scheduler.
                timesteps))
            latents_time_evolution_buffer[begin_idx + 1:end_idx + 1
                ] = block_latents_new
            latents_time_evolution_buffer[end_idx:new_end_idx + 1
                ] = latents_time_evolution_buffer[end_idx][None,]
            steps += 1
            progress_bar.update(new_begin_idx - begin_idx)
            if callback is not None and steps % callback_steps == 0:
                callback(begin_idx, block_t[begin_idx],
                    latents_time_evolution_buffer[begin_idx])
            begin_idx = new_begin_idx
            end_idx = new_end_idx
    latents = latents_time_evolution_buffer[-1]
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

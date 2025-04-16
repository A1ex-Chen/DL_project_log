@torch.no_grad()
def __call__(self, prompt: Union[str, List[str]]=None, image:
    PipelineImageInput=None, mask_image: PipelineImageInput=None,
    masked_image_latents: torch.Tensor=None, height: Optional[int]=None,
    width: Optional[int]=None, padding_mask_crop: Optional[int]=None,
    strength: float=1.0, num_inference_steps: int=50, timesteps: List[int]=
    None, guidance_scale: float=7.5, positive_prompt: Optional[str]='',
    negative_prompt: Optional[Union[str, List[str]]]=None,
    num_images_per_prompt: Optional[int]=1, eta: float=0.01, generator:
    Optional[Union[torch.Generator, List[torch.Generator]]]=None, latents:
    Optional[torch.Tensor]=None, prompt_embeds: Optional[torch.Tensor]=None,
    negative_prompt_embeds: Optional[torch.Tensor]=None, ip_adapter_image:
    Optional[PipelineImageInput]=None, output_type: Optional[str]='pil',
    return_dict: bool=True, cross_attention_kwargs: Optional[Dict[str, Any]
    ]=None, clip_skip: int=None, callback_on_step_end: Optional[Callable[[
    int, int, Dict], None]]=None, callback_on_step_end_tensor_inputs: List[
    str]=['latents'], use_painta=True, use_rasg=True,
    self_attention_layer_name='.attn1', cross_attention_layer_name='.attn2',
    painta_scale_factors=[2, 4], rasg_scale_factor=4,
    list_of_painta_layer_names=None, list_of_rasg_layer_names=None, **kwargs):
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
    height = height or self.unet.config.sample_size * self.vae_scale_factor
    width = width or self.unet.config.sample_size * self.vae_scale_factor
    prompt_no_positives = prompt
    if isinstance(prompt, list):
        prompt = [(x + positive_prompt) for x in prompt]
    else:
        prompt = prompt + positive_prompt
    self.check_inputs(prompt, image, mask_image, height, width, strength,
        callback_steps, negative_prompt, prompt_embeds,
        negative_prompt_embeds, callback_on_step_end_tensor_inputs,
        padding_mask_crop)
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
    text_encoder_lora_scale = cross_attention_kwargs.get('scale', None
        ) if cross_attention_kwargs is not None else None
    prompt_embeds, negative_prompt_embeds = self.encode_prompt(prompt,
        device, num_images_per_prompt, self.do_classifier_free_guidance,
        negative_prompt, prompt_embeds=prompt_embeds,
        negative_prompt_embeds=negative_prompt_embeds, lora_scale=
        text_encoder_lora_scale, clip_skip=self.clip_skip)
    if self.do_classifier_free_guidance:
        prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])
    if ip_adapter_image is not None:
        output_hidden_state = False if isinstance(self.unet.
            encoder_hid_proj, ImageProjection) else True
        image_embeds, negative_image_embeds = self.encode_image(
            ip_adapter_image, device, num_images_per_prompt,
            output_hidden_state)
        if self.do_classifier_free_guidance:
            image_embeds = torch.cat([negative_image_embeds, image_embeds])
    timesteps, num_inference_steps = retrieve_timesteps(self.scheduler,
        num_inference_steps, device, timesteps)
    timesteps, num_inference_steps = self.get_timesteps(num_inference_steps
        =num_inference_steps, strength=strength, device=device)
    if num_inference_steps < 1:
        raise ValueError(
            f'After adjusting the num_inference_steps by strength parameter: {strength}, the number of pipelinesteps is {num_inference_steps} which is < 1 and not appropriate for this pipeline.'
            )
    latent_timestep = timesteps[:1].repeat(batch_size * num_images_per_prompt)
    is_strength_max = strength == 1.0
    if padding_mask_crop is not None:
        crops_coords = self.mask_processor.get_crop_region(mask_image,
            width, height, pad=padding_mask_crop)
        resize_mode = 'fill'
    else:
        crops_coords = None
        resize_mode = 'default'
    original_image = image
    init_image = self.image_processor.preprocess(image, height=height,
        width=width, crops_coords=crops_coords, resize_mode=resize_mode)
    init_image = init_image.to(dtype=torch.float32)
    num_channels_latents = self.vae.config.latent_channels
    num_channels_unet = self.unet.config.in_channels
    return_image_latents = num_channels_unet == 4
    latents_outputs = self.prepare_latents(batch_size *
        num_images_per_prompt, num_channels_latents, height, width,
        prompt_embeds.dtype, device, generator, latents, image=init_image,
        timestep=latent_timestep, is_strength_max=is_strength_max,
        return_noise=True, return_image_latents=return_image_latents)
    if return_image_latents:
        latents, noise, image_latents = latents_outputs
    else:
        latents, noise = latents_outputs
    mask_condition = self.mask_processor.preprocess(mask_image, height=
        height, width=width, resize_mode=resize_mode, crops_coords=crops_coords
        )
    if masked_image_latents is None:
        masked_image = init_image * (mask_condition < 0.5)
    else:
        masked_image = masked_image_latents
    mask, masked_image_latents = self.prepare_mask_latents(mask_condition,
        masked_image, batch_size * num_images_per_prompt, height, width,
        prompt_embeds.dtype, device, generator, self.
        do_classifier_free_guidance)
    token_idx = list(range(1, self.get_tokenized_prompt(prompt_no_positives
        ).index('<|endoftext|>'))) + [self.get_tokenized_prompt(prompt).
        index('<|endoftext|>')]
    self.init_attn_processors(mask_condition, token_idx, use_painta,
        use_rasg, painta_scale_factors=painta_scale_factors,
        rasg_scale_factor=rasg_scale_factor, self_attention_layer_name=
        self_attention_layer_name, cross_attention_layer_name=
        cross_attention_layer_name, list_of_painta_layer_names=
        list_of_painta_layer_names, list_of_rasg_layer_names=
        list_of_rasg_layer_names)
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
    if use_rasg:
        extra_step_kwargs['generator'] = None
    added_cond_kwargs = {'image_embeds': image_embeds
        } if ip_adapter_image is not None else None
    timestep_cond = None
    if self.unet.config.time_cond_proj_dim is not None:
        guidance_scale_tensor = torch.tensor(self.guidance_scale - 1).repeat(
            batch_size * num_images_per_prompt)
        timestep_cond = self.get_guidance_scale_embedding(guidance_scale_tensor
            , embedding_dim=self.unet.config.time_cond_proj_dim).to(device=
            device, dtype=latents.dtype)
    num_warmup_steps = len(timesteps
        ) - num_inference_steps * self.scheduler.order
    self._num_timesteps = len(timesteps)
    painta_active = True
    with self.progress_bar(total=num_inference_steps) as progress_bar:
        for i, t in enumerate(timesteps):
            if self.interrupt:
                continue
            if t < 500 and painta_active:
                self.init_attn_processors(mask_condition, token_idx, False,
                    use_rasg, painta_scale_factors=painta_scale_factors,
                    rasg_scale_factor=rasg_scale_factor,
                    self_attention_layer_name=self_attention_layer_name,
                    cross_attention_layer_name=cross_attention_layer_name,
                    list_of_painta_layer_names=list_of_painta_layer_names,
                    list_of_rasg_layer_names=list_of_rasg_layer_names)
                painta_active = False
            with torch.enable_grad():
                self.unet.zero_grad()
                latents = latents.detach()
                latents.requires_grad = True
                latent_model_input = torch.cat([latents] * 2
                    ) if self.do_classifier_free_guidance else latents
                latent_model_input = self.scheduler.scale_model_input(
                    latent_model_input, t)
                if num_channels_unet == 9:
                    latent_model_input = torch.cat([latent_model_input,
                        mask, masked_image_latents], dim=1)
                self.scheduler.latents = latents
                self.encoder_hidden_states = prompt_embeds
                for attn_processor in self.unet.attn_processors.values():
                    attn_processor.encoder_hidden_states = prompt_embeds
                noise_pred = self.unet(latent_model_input, t,
                    encoder_hidden_states=prompt_embeds, timestep_cond=
                    timestep_cond, cross_attention_kwargs=self.
                    cross_attention_kwargs, added_cond_kwargs=
                    added_cond_kwargs, return_dict=False)[0]
                if self.do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + self.guidance_scale * (
                        noise_pred_text - noise_pred_uncond)
                if use_rasg:
                    _, _, height, width = mask_condition.shape
                    scale_factor = self.vae_scale_factor * rasg_scale_factor
                    rasg_mask = F.interpolate(mask_condition, (height //
                        scale_factor, width // scale_factor), mode='bicubic')[
                        0, 0]
                    attn_map = []
                    for processor in self.unet.attn_processors.values():
                        if hasattr(processor, 'attention_scores'
                            ) and processor.attention_scores is not None:
                            if self.do_classifier_free_guidance:
                                attn_map.append(processor.attention_scores.
                                    chunk(2)[1])
                            else:
                                attn_map.append(processor.attention_scores)
                    attn_map = torch.cat(attn_map).mean(0).permute(1, 0
                        ).reshape((-1, height // scale_factor, width //
                        scale_factor))
                    attn_score = -sum([F.binary_cross_entropy_with_logits(x -
                        1.0, rasg_mask.to(device)) for x in attn_map[
                        token_idx]])
                    attn_score.backward()
                    variance_noise = latents.grad.detach()
                    variance_noise -= torch.mean(variance_noise, [1, 2, 3],
                        keepdim=True)
                    variance_noise /= torch.std(variance_noise, [1, 2, 3],
                        keepdim=True)
                else:
                    variance_noise = None
            latents = self.scheduler.step(noise_pred, t, latents, **
                extra_step_kwargs, return_dict=False, variance_noise=
                variance_noise)[0]
            if num_channels_unet == 4:
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
                mask = callback_outputs.pop('mask', mask)
                masked_image_latents = callback_outputs.pop(
                    'masked_image_latents', masked_image_latents)
            if i == len(timesteps) - 1 or i + 1 > num_warmup_steps and (i + 1
                ) % self.scheduler.order == 0:
                progress_bar.update()
                if callback is not None and i % callback_steps == 0:
                    step_idx = i // getattr(self.scheduler, 'order', 1)
                    callback(step_idx, t, latents)
    if not output_type == 'latent':
        condition_kwargs = {}
        if isinstance(self.vae, AsymmetricAutoencoderKL):
            init_image = init_image.to(device=device, dtype=
                masked_image_latents.dtype)
            init_image_condition = init_image.clone()
            init_image = self._encode_vae_image(init_image, generator=generator
                )
            mask_condition = mask_condition.to(device=device, dtype=
                masked_image_latents.dtype)
            condition_kwargs = {'image': init_image_condition, 'mask':
                mask_condition}
        image = self.vae.decode(latents / self.vae.config.scaling_factor,
            return_dict=False, generator=generator, **condition_kwargs)[0]
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
    if padding_mask_crop is not None:
        image = [self.image_processor.apply_overlay(mask_image,
            original_image, i, crops_coords) for i in image]
    self.maybe_free_model_hooks()
    if not return_dict:
        return image, has_nsfw_concept
    return StableDiffusionPipelineOutput(images=image,
        nsfw_content_detected=has_nsfw_concept)

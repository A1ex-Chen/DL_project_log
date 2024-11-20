@torch.no_grad()
@replace_example_docstring(EXAMPLE_DOC_STRING)
def __call__(self, negative_prompt: Optional[Union[str, List[str]]]=None,
    generator: Optional[Union[torch.Generator, List[torch.Generator]]]=None,
    output_type: Optional[str]='pil', return_dict: bool=True,
    editing_prompt: Optional[Union[str, List[str]]]=None,
    editing_prompt_embeds: Optional[torch.Tensor]=None,
    negative_prompt_embeds: Optional[torch.Tensor]=None,
    reverse_editing_direction: Optional[Union[bool, List[bool]]]=False,
    edit_guidance_scale: Optional[Union[float, List[float]]]=5,
    edit_warmup_steps: Optional[Union[int, List[int]]]=0,
    edit_cooldown_steps: Optional[Union[int, List[int]]]=None,
    edit_threshold: Optional[Union[float, List[float]]]=0.9, user_mask:
    Optional[torch.Tensor]=None, sem_guidance: Optional[List[torch.Tensor]]
    =None, use_cross_attn_mask: bool=False, use_intersect_mask: bool=True,
    attn_store_steps: Optional[List[int]]=[], store_averaged_over_steps:
    bool=True, cross_attention_kwargs: Optional[Dict[str, Any]]=None,
    guidance_rescale: float=0.0, clip_skip: Optional[int]=None,
    callback_on_step_end: Optional[Callable[[int, int, Dict], None]]=None,
    callback_on_step_end_tensor_inputs: List[str]=['latents'], **kwargs):
    """
        The call function to the pipeline for editing. The
        [`~pipelines.ledits_pp.LEditsPPPipelineStableDiffusion.invert`] method has to be called beforehand. Edits will
        always be performed for the last inverted image(s).

        Args:
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the image generation. Ignored when not using guidance (i.e., ignored
                if `guidance_scale` is less than `1`).
            generator (`torch.Generator`, *optional*):
                One or a list of [torch generator(s)](https://pytorch.org/docs/stable/generated/torch.Generator.html)
                to make generation deterministic.
            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generate image. Choose between
                [PIL](https://pillow.readthedocs.io/en/stable/): `PIL.Image.Image` or `np.array`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~pipelines.ledits_pp.LEditsPPDiffusionPipelineOutput`] instead of a plain
                tuple.
            editing_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts to guide the image generation. The image is reconstructed by setting
                `editing_prompt = None`. Guidance direction of prompt should be specified via
                `reverse_editing_direction`.
            editing_prompt_embeds (`torch.Tensor>`, *optional*):
                Pre-computed embeddings to use for guiding the image generation. Guidance direction of embedding should
                be specified via `reverse_editing_direction`.
            negative_prompt_embeds (`torch.Tensor`, *optional*):
                Pre-generated negative text embeddings. Can be used to easily tweak text inputs (prompt weighting). If
                not provided, `negative_prompt_embeds` are generated from the `negative_prompt` input argument.
            reverse_editing_direction (`bool` or `List[bool]`, *optional*, defaults to `False`):
                Whether the corresponding prompt in `editing_prompt` should be increased or decreased.
            edit_guidance_scale (`float` or `List[float]`, *optional*, defaults to 5):
                Guidance scale for guiding the image generation. If provided as list values should correspond to
                `editing_prompt`. `edit_guidance_scale` is defined as `s_e` of equation 12 of [LEDITS++
                Paper](https://arxiv.org/abs/2301.12247).
            edit_warmup_steps (`float` or `List[float]`, *optional*, defaults to 10):
                Number of diffusion steps (for each prompt) for which guidance will not be applied.
            edit_cooldown_steps (`float` or `List[float]`, *optional*, defaults to `None`):
                Number of diffusion steps (for each prompt) after which guidance will no longer be applied.
            edit_threshold (`float` or `List[float]`, *optional*, defaults to 0.9):
                Masking threshold of guidance. Threshold should be proportional to the image region that is modified.
                'edit_threshold' is defined as 'λ' of equation 12 of [LEDITS++
                Paper](https://arxiv.org/abs/2301.12247).
            user_mask (`torch.Tensor`, *optional*):
                User-provided mask for even better control over the editing process. This is helpful when LEDITS++'s
                implicit masks do not meet user preferences.
            sem_guidance (`List[torch.Tensor]`, *optional*):
                List of pre-generated guidance vectors to be applied at generation. Length of the list has to
                correspond to `num_inference_steps`.
            use_cross_attn_mask (`bool`, defaults to `False`):
                Whether cross-attention masks are used. Cross-attention masks are always used when use_intersect_mask
                is set to true. Cross-attention masks are defined as 'M^1' of equation 12 of [LEDITS++
                paper](https://arxiv.org/pdf/2311.16711.pdf).
            use_intersect_mask (`bool`, defaults to `True`):
                Whether the masking term is calculated as intersection of cross-attention masks and masks derived from
                the noise estimate. Cross-attention mask are defined as 'M^1' and masks derived from the noise estimate
                are defined as 'M^2' of equation 12 of [LEDITS++ paper](https://arxiv.org/pdf/2311.16711.pdf).
            attn_store_steps (`List[int]`, *optional*):
                Steps for which the attention maps are stored in the AttentionStore. Just for visualization purposes.
            store_averaged_over_steps (`bool`, defaults to `True`):
                Whether the attention maps for the 'attn_store_steps' are stored averaged over the diffusion steps. If
                False, attention maps for each step are stores separately. Just for visualization purposes.
            cross_attention_kwargs (`dict`, *optional*):
                A kwargs dictionary that if specified is passed along to the [`AttentionProcessor`] as defined in
                [`self.processor`](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py).
            guidance_rescale (`float`, *optional*, defaults to 0.0):
                Guidance rescale factor from [Common Diffusion Noise Schedules and Sample Steps are
                Flawed](https://arxiv.org/pdf/2305.08891.pdf). Guidance rescale factor should fix overexposure when
                using zero terminal SNR.
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
            [`~pipelines.ledits_pp.LEditsPPDiffusionPipelineOutput`] or `tuple`:
            [`~pipelines.ledits_pp.LEditsPPDiffusionPipelineOutput`] if `return_dict` is True, otherwise a `tuple. When
            returning a tuple, the first element is a list with the generated images, and the second element is a list
            of `bool`s denoting whether the corresponding generated image likely represents "not-safe-for-work" (nsfw)
            content, according to the `safety_checker`.
        """
    if self.inversion_steps is None:
        raise ValueError(
            'You need to invert an input image first before calling the pipeline. The `invert` method has to be called beforehand. Edits will always be performed for the last inverted image(s).'
            )
    eta = self.eta
    num_images_per_prompt = 1
    latents = self.init_latents
    zs = self.zs
    self.scheduler.set_timesteps(len(self.scheduler.timesteps))
    if use_intersect_mask:
        use_cross_attn_mask = True
    if use_cross_attn_mask:
        self.smoothing = LeditsGaussianSmoothing(self.device)
    if user_mask is not None:
        user_mask = user_mask.to(self.device)
    org_prompt = ''
    self.check_inputs(negative_prompt, editing_prompt_embeds,
        negative_prompt_embeds, callback_on_step_end_tensor_inputs)
    self._guidance_rescale = guidance_rescale
    self._clip_skip = clip_skip
    self._cross_attention_kwargs = cross_attention_kwargs
    batch_size = self.batch_size
    if editing_prompt:
        enable_edit_guidance = True
        if isinstance(editing_prompt, str):
            editing_prompt = [editing_prompt]
        self.enabled_editing_prompts = len(editing_prompt)
    elif editing_prompt_embeds is not None:
        enable_edit_guidance = True
        self.enabled_editing_prompts = editing_prompt_embeds.shape[0]
    else:
        self.enabled_editing_prompts = 0
        enable_edit_guidance = False
    lora_scale = self.cross_attention_kwargs.get('scale', None
        ) if self.cross_attention_kwargs is not None else None
    edit_concepts, uncond_embeddings, num_edit_tokens = self.encode_prompt(
        editing_prompt=editing_prompt, device=self.device,
        num_images_per_prompt=num_images_per_prompt, enable_edit_guidance=
        enable_edit_guidance, negative_prompt=negative_prompt,
        editing_prompt_embeds=editing_prompt_embeds, negative_prompt_embeds
        =negative_prompt_embeds, lora_scale=lora_scale, clip_skip=self.
        clip_skip)
    if enable_edit_guidance:
        text_embeddings = torch.cat([uncond_embeddings, edit_concepts])
        self.text_cross_attention_maps = [editing_prompt] if isinstance(
            editing_prompt, str) else editing_prompt
    else:
        text_embeddings = torch.cat([uncond_embeddings])
    timesteps = self.inversion_steps
    t_to_idx = {int(v): k for k, v in enumerate(timesteps[-zs.shape[0]:])}
    if use_cross_attn_mask:
        self.attention_store = LeditsAttentionStore(average=
            store_averaged_over_steps, batch_size=batch_size, max_size=
            latents.shape[-2] / 4.0 * (latents.shape[-1] / 4.0),
            max_resolution=None)
        self.prepare_unet(self.attention_store, PnP=False)
        resolution = latents.shape[-2:]
        att_res = int(resolution[0] / 4), int(resolution[1] / 4)
    num_channels_latents = self.unet.config.in_channels
    latents = self.prepare_latents(batch_size * num_images_per_prompt,
        num_channels_latents, None, None, text_embeddings.dtype, self.
        device, latents)
    extra_step_kwargs = self.prepare_extra_step_kwargs(eta)
    self.sem_guidance = None
    self.activation_mask = None
    num_warmup_steps = 0
    with self.progress_bar(total=len(timesteps)) as progress_bar:
        for i, t in enumerate(timesteps):
            if enable_edit_guidance:
                latent_model_input = torch.cat([latents] * (1 + self.
                    enabled_editing_prompts))
            else:
                latent_model_input = latents
            latent_model_input = self.scheduler.scale_model_input(
                latent_model_input, t)
            text_embed_input = text_embeddings
            noise_pred = self.unet(latent_model_input, t,
                encoder_hidden_states=text_embed_input).sample
            noise_pred_out = noise_pred.chunk(1 + self.enabled_editing_prompts)
            noise_pred_uncond = noise_pred_out[0]
            noise_pred_edit_concepts = noise_pred_out[1:]
            noise_guidance_edit = torch.zeros(noise_pred_uncond.shape,
                device=self.device, dtype=noise_pred_uncond.dtype)
            if sem_guidance is not None and len(sem_guidance) > i:
                noise_guidance_edit += sem_guidance[i].to(self.device)
            elif enable_edit_guidance:
                if self.activation_mask is None:
                    self.activation_mask = torch.zeros((len(timesteps), len
                        (noise_pred_edit_concepts), *
                        noise_pred_edit_concepts[0].shape))
                if self.sem_guidance is None:
                    self.sem_guidance = torch.zeros((len(timesteps), *
                        noise_pred_uncond.shape))
                for c, noise_pred_edit_concept in enumerate(
                    noise_pred_edit_concepts):
                    if isinstance(edit_warmup_steps, list):
                        edit_warmup_steps_c = edit_warmup_steps[c]
                    else:
                        edit_warmup_steps_c = edit_warmup_steps
                    if i < edit_warmup_steps_c:
                        continue
                    if isinstance(edit_guidance_scale, list):
                        edit_guidance_scale_c = edit_guidance_scale[c]
                    else:
                        edit_guidance_scale_c = edit_guidance_scale
                    if isinstance(edit_threshold, list):
                        edit_threshold_c = edit_threshold[c]
                    else:
                        edit_threshold_c = edit_threshold
                    if isinstance(reverse_editing_direction, list):
                        reverse_editing_direction_c = (
                            reverse_editing_direction[c])
                    else:
                        reverse_editing_direction_c = reverse_editing_direction
                    if isinstance(edit_cooldown_steps, list):
                        edit_cooldown_steps_c = edit_cooldown_steps[c]
                    elif edit_cooldown_steps is None:
                        edit_cooldown_steps_c = i + 1
                    else:
                        edit_cooldown_steps_c = edit_cooldown_steps
                    if i >= edit_cooldown_steps_c:
                        continue
                    noise_guidance_edit_tmp = (noise_pred_edit_concept -
                        noise_pred_uncond)
                    if reverse_editing_direction_c:
                        noise_guidance_edit_tmp = noise_guidance_edit_tmp * -1
                    noise_guidance_edit_tmp = (noise_guidance_edit_tmp *
                        edit_guidance_scale_c)
                    if user_mask is not None:
                        noise_guidance_edit_tmp = (noise_guidance_edit_tmp *
                            user_mask)
                    if use_cross_attn_mask:
                        out = self.attention_store.aggregate_attention(
                            attention_maps=self.attention_store.step_store,
                            prompts=self.text_cross_attention_maps, res=
                            att_res, from_where=['up', 'down'], is_cross=
                            True, select=self.text_cross_attention_maps.
                            index(editing_prompt[c]))
                        attn_map = out[:, :, :, 1:1 + num_edit_tokens[c]]
                        if attn_map.shape[3] != num_edit_tokens[c]:
                            raise ValueError(
                                f'Incorrect shape of attention_map. Expected size {num_edit_tokens[c]}, but found {attn_map.shape[3]}!'
                                )
                        attn_map = torch.sum(attn_map, dim=3)
                        attn_map = F.pad(attn_map.unsqueeze(1), (1, 1, 1, 1
                            ), mode='reflect')
                        attn_map = self.smoothing(attn_map).squeeze(1)
                        if attn_map.dtype == torch.float32:
                            tmp = torch.quantile(attn_map.flatten(start_dim
                                =1), edit_threshold_c, dim=1)
                        else:
                            tmp = torch.quantile(attn_map.flatten(start_dim
                                =1).to(torch.float32), edit_threshold_c, dim=1
                                ).to(attn_map.dtype)
                        attn_mask = torch.where(attn_map >= tmp.unsqueeze(1
                            ).unsqueeze(1).repeat(1, *att_res), 1.0, 0.0)
                        attn_mask = F.interpolate(attn_mask.unsqueeze(1),
                            noise_guidance_edit_tmp.shape[-2:]).repeat(1, 4,
                            1, 1)
                        self.activation_mask[i, c] = attn_mask.detach().cpu()
                        if not use_intersect_mask:
                            noise_guidance_edit_tmp = (
                                noise_guidance_edit_tmp * attn_mask)
                    if use_intersect_mask:
                        if t <= 800:
                            noise_guidance_edit_tmp_quantile = torch.abs(
                                noise_guidance_edit_tmp)
                            noise_guidance_edit_tmp_quantile = torch.sum(
                                noise_guidance_edit_tmp_quantile, dim=1,
                                keepdim=True)
                            noise_guidance_edit_tmp_quantile = (
                                noise_guidance_edit_tmp_quantile.repeat(1,
                                self.unet.config.in_channels, 1, 1))
                            if (noise_guidance_edit_tmp_quantile.dtype ==
                                torch.float32):
                                tmp = torch.quantile(
                                    noise_guidance_edit_tmp_quantile.
                                    flatten(start_dim=2), edit_threshold_c,
                                    dim=2, keepdim=False)
                            else:
                                tmp = torch.quantile(
                                    noise_guidance_edit_tmp_quantile.
                                    flatten(start_dim=2).to(torch.float32),
                                    edit_threshold_c, dim=2, keepdim=False).to(
                                    noise_guidance_edit_tmp_quantile.dtype)
                            intersect_mask = torch.where(
                                noise_guidance_edit_tmp_quantile >= tmp[:,
                                :, None, None], torch.ones_like(
                                noise_guidance_edit_tmp), torch.zeros_like(
                                noise_guidance_edit_tmp)) * attn_mask
                            self.activation_mask[i, c] = intersect_mask.detach(
                                ).cpu()
                            noise_guidance_edit_tmp = (
                                noise_guidance_edit_tmp * intersect_mask)
                        else:
                            noise_guidance_edit_tmp = (
                                noise_guidance_edit_tmp * attn_mask)
                    elif not use_cross_attn_mask:
                        noise_guidance_edit_tmp_quantile = torch.abs(
                            noise_guidance_edit_tmp)
                        noise_guidance_edit_tmp_quantile = torch.sum(
                            noise_guidance_edit_tmp_quantile, dim=1,
                            keepdim=True)
                        noise_guidance_edit_tmp_quantile = (
                            noise_guidance_edit_tmp_quantile.repeat(1, 4, 1, 1)
                            )
                        if (noise_guidance_edit_tmp_quantile.dtype == torch
                            .float32):
                            tmp = torch.quantile(
                                noise_guidance_edit_tmp_quantile.flatten(
                                start_dim=2), edit_threshold_c, dim=2,
                                keepdim=False)
                        else:
                            tmp = torch.quantile(
                                noise_guidance_edit_tmp_quantile.flatten(
                                start_dim=2).to(torch.float32),
                                edit_threshold_c, dim=2, keepdim=False).to(
                                noise_guidance_edit_tmp_quantile.dtype)
                        self.activation_mask[i, c] = torch.where(
                            noise_guidance_edit_tmp_quantile >= tmp[:, :,
                            None, None], torch.ones_like(
                            noise_guidance_edit_tmp), torch.zeros_like(
                            noise_guidance_edit_tmp)).detach().cpu()
                        noise_guidance_edit_tmp = torch.where(
                            noise_guidance_edit_tmp_quantile >= tmp[:, :,
                            None, None], noise_guidance_edit_tmp, torch.
                            zeros_like(noise_guidance_edit_tmp))
                    noise_guidance_edit += noise_guidance_edit_tmp
                self.sem_guidance[i] = noise_guidance_edit.detach().cpu()
            noise_pred = noise_pred_uncond + noise_guidance_edit
            if enable_edit_guidance and self.guidance_rescale > 0.0:
                noise_pred = rescale_noise_cfg(noise_pred,
                    noise_pred_edit_concepts.mean(dim=0, keepdim=False),
                    guidance_rescale=self.guidance_rescale)
            idx = t_to_idx[int(t)]
            latents = self.scheduler.step(noise_pred, t, latents,
                variance_noise=zs[idx], **extra_step_kwargs).prev_sample
            if use_cross_attn_mask:
                store_step = i in attn_store_steps
                self.attention_store.between_steps(store_step)
            if callback_on_step_end is not None:
                callback_kwargs = {}
                for k in callback_on_step_end_tensor_inputs:
                    callback_kwargs[k] = locals()[k]
                callback_outputs = callback_on_step_end(self, i, t,
                    callback_kwargs)
                latents = callback_outputs.pop('latents', latents)
                negative_prompt_embeds = callback_outputs.pop(
                    'negative_prompt_embeds', negative_prompt_embeds)
            if i == len(timesteps) - 1 or i + 1 > num_warmup_steps and (i + 1
                ) % self.scheduler.order == 0:
                progress_bar.update()
    if not output_type == 'latent':
        image = self.vae.decode(latents / self.vae.config.scaling_factor,
            return_dict=False, generator=generator)[0]
        image, has_nsfw_concept = self.run_safety_checker(image, self.
            device, text_embeddings.dtype)
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
    return LEditsPPDiffusionPipelineOutput(images=image,
        nsfw_content_detected=has_nsfw_concept)

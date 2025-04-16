@torch.no_grad()
@replace_example_docstring(EXAMPLE_DOC_STRING)
def __call__(self, denoising_end: Optional[float]=None, negative_prompt:
    Optional[Union[str, List[str]]]=None, negative_prompt_2: Optional[Union
    [str, List[str]]]=None, negative_prompt_embeds: Optional[torch.Tensor]=
    None, negative_pooled_prompt_embeds: Optional[torch.Tensor]=None,
    ip_adapter_image: Optional[PipelineImageInput]=None, output_type:
    Optional[str]='pil', return_dict: bool=True, cross_attention_kwargs:
    Optional[Dict[str, Any]]=None, guidance_rescale: float=0.0,
    crops_coords_top_left: Tuple[int, int]=(0, 0), target_size: Optional[
    Tuple[int, int]]=None, editing_prompt: Optional[Union[str, List[str]]]=
    None, editing_prompt_embeddings: Optional[torch.Tensor]=None,
    editing_pooled_prompt_embeds: Optional[torch.Tensor]=None,
    reverse_editing_direction: Optional[Union[bool, List[bool]]]=False,
    edit_guidance_scale: Optional[Union[float, List[float]]]=5,
    edit_warmup_steps: Optional[Union[int, List[int]]]=0,
    edit_cooldown_steps: Optional[Union[int, List[int]]]=None,
    edit_threshold: Optional[Union[float, List[float]]]=0.9, sem_guidance:
    Optional[List[torch.Tensor]]=None, use_cross_attn_mask: bool=False,
    use_intersect_mask: bool=False, user_mask: Optional[torch.Tensor]=None,
    attn_store_steps: Optional[List[int]]=[], store_averaged_over_steps:
    bool=True, clip_skip: Optional[int]=None, callback_on_step_end:
    Optional[Callable[[int, int, Dict], None]]=None,
    callback_on_step_end_tensor_inputs: List[str]=['latents'], **kwargs):
    """
        The call function to the pipeline for editing. The
        [`~pipelines.ledits_pp.LEditsPPPipelineStableDiffusionXL.invert`] method has to be called beforehand. Edits
        will always be performed for the last inverted image(s).

        Args:
            denoising_end (`float`, *optional*):
                When specified, determines the fraction (between 0.0 and 1.0) of the total denoising process to be
                completed before it is intentionally prematurely terminated. As a result, the returned sample will
                still retain a substantial amount of noise as determined by the discrete timesteps selected by the
                scheduler. The denoising_end parameter should ideally be utilized when this pipeline forms a part of a
                "Mixture of Denoisers" multi-pipeline setup, as elaborated in [**Refining the Image
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the image generation. If not defined, one has to pass
                `negative_prompt_embeds` instead. Ignored when not using guidance (i.e., ignored if `guidance_scale` is
                less than `1`).
            negative_prompt_2 (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the image generation to be sent to `tokenizer_2` and
                `text_encoder_2`. If not defined, `negative_prompt` is used in both text-encoders
            negative_prompt_embeds (`torch.Tensor`, *optional*):
                Pre-generated negative text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt
                weighting. If not provided, negative_prompt_embeds will be generated from `negative_prompt` input
                argument.
            negative_pooled_prompt_embeds (`torch.Tensor`, *optional*):
                Pre-generated negative pooled text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt
                weighting. If not provided, pooled negative_prompt_embeds will be generated from `negative_prompt`
                input argument.
            ip_adapter_image: (`PipelineImageInput`, *optional*):
                Optional image input to work with IP Adapters.
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
                [diffusers.models.attention_processor](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py).
            guidance_rescale (`float`, *optional*, defaults to 0.7):
                Guidance rescale factor proposed by [Common Diffusion Noise Schedules and Sample Steps are
                Flawed](https://arxiv.org/pdf/2305.08891.pdf) `guidance_scale` is defined as `φ` in equation 16. of
                [Common Diffusion Noise Schedules and Sample Steps are Flawed](https://arxiv.org/pdf/2305.08891.pdf).
                Guidance rescale factor should fix overexposure when using zero terminal SNR.
            crops_coords_top_left (`Tuple[int]`, *optional*, defaults to (0, 0)):
                `crops_coords_top_left` can be used to generate an image that appears to be "cropped" from the position
                `crops_coords_top_left` downwards. Favorable, well-centered images are usually achieved by setting
                `crops_coords_top_left` to (0, 0). Part of SDXL's micro-conditioning as explained in section 2.2 of
                [https://huggingface.co/papers/2307.01952](https://huggingface.co/papers/2307.01952).
            target_size (`Tuple[int]`, *optional*, defaults to (1024, 1024)):
                For most cases, `target_size` should be set to the desired height and width of the generated image. If
                not specified it will default to `(width, height)`. Part of SDXL's micro-conditioning as explained in
                section 2.2 of [https://huggingface.co/papers/2307.01952](https://huggingface.co/papers/2307.01952).
            editing_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts to guide the image generation. The image is reconstructed by setting
                `editing_prompt = None`. Guidance direction of prompt should be specified via
                `reverse_editing_direction`.
            editing_prompt_embeddings (`torch.Tensor`, *optional*):
                Pre-generated edit text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting.
                If not provided, editing_prompt_embeddings will be generated from `editing_prompt` input argument.
            editing_pooled_prompt_embeddings (`torch.Tensor`, *optional*):
                Pre-generated pooled edit text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt
                weighting. If not provided, editing_prompt_embeddings will be generated from `editing_prompt` input
                argument.
            reverse_editing_direction (`bool` or `List[bool]`, *optional*, defaults to `False`):
                Whether the corresponding prompt in `editing_prompt` should be increased or decreased.
            edit_guidance_scale (`float` or `List[float]`, *optional*, defaults to 5):
                Guidance scale for guiding the image generation. If provided as list values should correspond to
                `editing_prompt`. `edit_guidance_scale` is defined as `s_e` of equation 12 of [LEDITS++
                Paper](https://arxiv.org/abs/2301.12247).
            edit_warmup_steps (`float` or `List[float]`, *optional*, defaults to 10):
                Number of diffusion steps (for each prompt) for which guidance is not applied.
            edit_cooldown_steps (`float` or `List[float]`, *optional*, defaults to `None`):
                Number of diffusion steps (for each prompt) after which guidance is no longer applied.
            edit_threshold (`float` or `List[float]`, *optional*, defaults to 0.9):
                Masking threshold of guidance. Threshold should be proportional to the image region that is modified.
                'edit_threshold' is defined as 'λ' of equation 12 of [LEDITS++
                Paper](https://arxiv.org/abs/2301.12247).
            sem_guidance (`List[torch.Tensor]`, *optional*):
                List of pre-generated guidance vectors to be applied at generation. Length of the list has to
                correspond to `num_inference_steps`.
            use_cross_attn_mask:
                Whether cross-attention masks are used. Cross-attention masks are always used when use_intersect_mask
                is set to true. Cross-attention masks are defined as 'M^1' of equation 12 of [LEDITS++
                paper](https://arxiv.org/pdf/2311.16711.pdf).
            use_intersect_mask:
                Whether the masking term is calculated as intersection of cross-attention masks and masks derived from
                the noise estimate. Cross-attention mask are defined as 'M^1' and masks derived from the noise estimate
                are defined as 'M^2' of equation 12 of [LEDITS++ paper](https://arxiv.org/pdf/2311.16711.pdf).
            user_mask:
                User-provided mask for even better control over the editing process. This is helpful when LEDITS++'s
                implicit masks do not meet user preferences.
            attn_store_steps:
                Steps for which the attention maps are stored in the AttentionStore. Just for visualization purposes.
            store_averaged_over_steps:
                Whether the attention maps for the 'attn_store_steps' are stored averaged over the diffusion steps. If
                False, attention maps for each step are stores separately. Just for visualization purposes.
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
            returning a tuple, the first element is a list with the generated images.
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
    self._guidance_rescale = guidance_rescale
    self._clip_skip = clip_skip
    self._cross_attention_kwargs = cross_attention_kwargs
    self._denoising_end = denoising_end
    batch_size = self.batch_size
    device = self._execution_device
    if editing_prompt:
        enable_edit_guidance = True
        if isinstance(editing_prompt, str):
            editing_prompt = [editing_prompt]
        self.enabled_editing_prompts = len(editing_prompt)
    elif editing_prompt_embeddings is not None:
        enable_edit_guidance = True
        self.enabled_editing_prompts = editing_prompt_embeddings.shape[0]
    else:
        self.enabled_editing_prompts = 0
        enable_edit_guidance = False
    text_encoder_lora_scale = cross_attention_kwargs.get('scale', None
        ) if cross_attention_kwargs is not None else None
    (prompt_embeds, edit_prompt_embeds, negative_pooled_prompt_embeds,
        pooled_edit_embeds, num_edit_tokens) = (self.encode_prompt(device=
        device, num_images_per_prompt=num_images_per_prompt,
        negative_prompt=negative_prompt, negative_prompt_2=
        negative_prompt_2, negative_prompt_embeds=negative_prompt_embeds,
        negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
        lora_scale=text_encoder_lora_scale, clip_skip=self.clip_skip,
        enable_edit_guidance=enable_edit_guidance, editing_prompt=
        editing_prompt, editing_prompt_embeds=editing_prompt_embeddings,
        editing_pooled_prompt_embeds=editing_pooled_prompt_embeds))
    timesteps = self.inversion_steps
    t_to_idx = {int(v): k for k, v in enumerate(timesteps)}
    if use_cross_attn_mask:
        self.attention_store = LeditsAttentionStore(average=
            store_averaged_over_steps, batch_size=batch_size, max_size=
            latents.shape[-2] / 4.0 * (latents.shape[-1] / 4.0),
            max_resolution=None)
        self.prepare_unet(self.attention_store)
        resolution = latents.shape[-2:]
        att_res = int(resolution[0] / 4), int(resolution[1] / 4)
    latents = self.prepare_latents(device=device, latents=latents)
    extra_step_kwargs = self.prepare_extra_step_kwargs(eta)
    if self.text_encoder_2 is None:
        text_encoder_projection_dim = int(negative_pooled_prompt_embeds.
            shape[-1])
    else:
        text_encoder_projection_dim = self.text_encoder_2.config.projection_dim
    add_text_embeds = negative_pooled_prompt_embeds
    add_time_ids = self._get_add_time_ids(self.size, crops_coords_top_left,
        self.size, dtype=negative_pooled_prompt_embeds.dtype,
        text_encoder_projection_dim=text_encoder_projection_dim)
    if enable_edit_guidance:
        prompt_embeds = torch.cat([prompt_embeds, edit_prompt_embeds], dim=0)
        add_text_embeds = torch.cat([add_text_embeds, pooled_edit_embeds],
            dim=0)
        edit_concepts_time_ids = add_time_ids.repeat(edit_prompt_embeds.
            shape[0], 1)
        add_time_ids = torch.cat([add_time_ids, edit_concepts_time_ids], dim=0)
        self.text_cross_attention_maps = [editing_prompt] if isinstance(
            editing_prompt, str) else editing_prompt
    prompt_embeds = prompt_embeds.to(device)
    add_text_embeds = add_text_embeds.to(device)
    add_time_ids = add_time_ids.to(device).repeat(batch_size *
        num_images_per_prompt, 1)
    if ip_adapter_image is not None:
        image_embeds, negative_image_embeds = self.encode_image(
            ip_adapter_image, device, num_images_per_prompt)
        if self.do_classifier_free_guidance:
            image_embeds = torch.cat([negative_image_embeds, image_embeds])
            image_embeds = image_embeds.to(device)
    self.sem_guidance = None
    self.activation_mask = None
    if self.denoising_end is not None and isinstance(self.denoising_end, float
        ) and self.denoising_end > 0 and self.denoising_end < 1:
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
    with self.progress_bar(total=self._num_timesteps) as progress_bar:
        for i, t in enumerate(timesteps):
            latent_model_input = torch.cat([latents] * (1 + self.
                enabled_editing_prompts))
            latent_model_input = self.scheduler.scale_model_input(
                latent_model_input, t)
            added_cond_kwargs = {'text_embeds': add_text_embeds, 'time_ids':
                add_time_ids}
            if ip_adapter_image is not None:
                added_cond_kwargs['image_embeds'] = image_embeds
            noise_pred = self.unet(latent_model_input, t,
                encoder_hidden_states=prompt_embeds, cross_attention_kwargs
                =cross_attention_kwargs, added_cond_kwargs=
                added_cond_kwargs, return_dict=False)[0]
            noise_pred_out = noise_pred.chunk(1 + self.enabled_editing_prompts)
            noise_pred_uncond = noise_pred_out[0]
            noise_pred_edit_concepts = noise_pred_out[1:]
            noise_guidance_edit = torch.zeros(noise_pred_uncond.shape,
                device=self.device, dtype=noise_pred_uncond.dtype)
            if sem_guidance is not None and len(sem_guidance) > i:
                noise_guidance_edit += sem_guidance[i].to(self.device)
            elif enable_edit_guidance:
                if self.activation_mask is None:
                    self.activation_mask = torch.zeros((len(timesteps),
                        self.enabled_editing_prompts, *
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
                        noise_guidance_edit_tmp_quantile = torch.abs(
                            noise_guidance_edit_tmp)
                        noise_guidance_edit_tmp_quantile = torch.sum(
                            noise_guidance_edit_tmp_quantile, dim=1,
                            keepdim=True)
                        noise_guidance_edit_tmp_quantile = (
                            noise_guidance_edit_tmp_quantile.repeat(1, self
                            .unet.config.in_channels, 1, 1))
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
                        intersect_mask = torch.where(
                            noise_guidance_edit_tmp_quantile >= tmp[:, :,
                            None, None], torch.ones_like(
                            noise_guidance_edit_tmp), torch.zeros_like(
                            noise_guidance_edit_tmp)) * attn_mask
                        self.activation_mask[i, c] = intersect_mask.detach(
                            ).cpu()
                        noise_guidance_edit_tmp = (noise_guidance_edit_tmp *
                            intersect_mask)
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
                variance_noise=zs[idx], **extra_step_kwargs, return_dict=False
                )[0]
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
            if i == len(timesteps) - 1 or i + 1 > 0 and (i + 1
                ) % self.scheduler.order == 0:
                progress_bar.update()
            if XLA_AVAILABLE:
                xm.mark_step()
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
    return LEditsPPDiffusionPipelineOutput(images=image,
        nsfw_content_detected=None)

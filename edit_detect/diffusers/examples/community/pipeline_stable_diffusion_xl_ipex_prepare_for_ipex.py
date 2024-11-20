@torch.no_grad()
def prepare_for_ipex(self, dtype=torch.float32, prompt: Union[str, List[str
    ]]=None, prompt_2: Optional[Union[str, List[str]]]=None, height:
    Optional[int]=None, width: Optional[int]=None, num_inference_steps: int
    =50, timesteps: List[int]=None, denoising_end: Optional[float]=None,
    guidance_scale: float=5.0, negative_prompt: Optional[Union[str, List[
    str]]]=None, negative_prompt_2: Optional[Union[str, List[str]]]=None,
    num_images_per_prompt: Optional[int]=1, eta: float=0.0, generator:
    Optional[Union[torch.Generator, List[torch.Generator]]]=None, latents:
    Optional[torch.Tensor]=None, prompt_embeds: Optional[torch.Tensor]=None,
    negative_prompt_embeds: Optional[torch.Tensor]=None,
    pooled_prompt_embeds: Optional[torch.Tensor]=None,
    negative_pooled_prompt_embeds: Optional[torch.Tensor]=None,
    ip_adapter_image: Optional[PipelineImageInput]=None, output_type:
    Optional[str]='pil', return_dict: bool=True, cross_attention_kwargs:
    Optional[Dict[str, Any]]=None, guidance_rescale: float=0.0,
    original_size: Optional[Tuple[int, int]]=None, crops_coords_top_left:
    Tuple[int, int]=(0, 0), target_size: Optional[Tuple[int, int]]=None,
    negative_original_size: Optional[Tuple[int, int]]=None,
    negative_crops_coords_top_left: Tuple[int, int]=(0, 0),
    negative_target_size: Optional[Tuple[int, int]]=None, clip_skip:
    Optional[int]=None, callback_on_step_end: Optional[Callable[[int, int,
    Dict], None]]=None, callback_on_step_end_tensor_inputs: List[str]=[
    'latents'], **kwargs):
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
    height = height or self.default_sample_size * self.vae_scale_factor
    width = width or self.default_sample_size * self.vae_scale_factor
    original_size = original_size or (height, width)
    target_size = target_size or (height, width)
    self.check_inputs(prompt, prompt_2, height, width, callback_steps,
        negative_prompt, negative_prompt_2, prompt_embeds,
        negative_prompt_embeds, pooled_prompt_embeds,
        negative_pooled_prompt_embeds, callback_on_step_end_tensor_inputs)
    self._guidance_scale = guidance_scale
    self._guidance_rescale = guidance_rescale
    self._clip_skip = clip_skip
    self._cross_attention_kwargs = cross_attention_kwargs
    self._denoising_end = denoising_end
    if prompt is not None and isinstance(prompt, str):
        batch_size = 1
    elif prompt is not None and isinstance(prompt, list):
        batch_size = len(prompt)
    else:
        batch_size = prompt_embeds.shape[0]
    device = 'cpu'
    do_classifier_free_guidance = self.do_classifier_free_guidance
    lora_scale = self.cross_attention_kwargs.get('scale', None
        ) if self.cross_attention_kwargs is not None else None
    (prompt_embeds, negative_prompt_embeds, pooled_prompt_embeds,
        negative_pooled_prompt_embeds) = (self.encode_prompt(prompt=prompt,
        prompt_2=prompt_2, device=device, num_images_per_prompt=
        num_images_per_prompt, do_classifier_free_guidance=self.
        do_classifier_free_guidance, negative_prompt=negative_prompt,
        negative_prompt_2=negative_prompt_2, prompt_embeds=prompt_embeds,
        negative_prompt_embeds=negative_prompt_embeds, pooled_prompt_embeds
        =pooled_prompt_embeds, negative_pooled_prompt_embeds=
        negative_pooled_prompt_embeds, lora_scale=lora_scale, clip_skip=
        self.clip_skip))
    num_channels_latents = self.unet.config.in_channels
    latents = self.prepare_latents(batch_size * num_images_per_prompt,
        num_channels_latents, height, width, prompt_embeds.dtype, device,
        generator, latents)
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
    if self.do_classifier_free_guidance:
        prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds],
            dim=0)
        add_text_embeds = torch.cat([negative_pooled_prompt_embeds,
            add_text_embeds], dim=0)
        add_time_ids = torch.cat([negative_add_time_ids, add_time_ids], dim=0)
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
    dummy = torch.ones(1, dtype=torch.int32)
    latent_model_input = torch.cat([latents] * 2
        ) if do_classifier_free_guidance else latents
    latent_model_input = self.scheduler.scale_model_input(latent_model_input,
        dummy)
    added_cond_kwargs = {'text_embeds': add_text_embeds, 'time_ids':
        add_time_ids}
    if ip_adapter_image is not None:
        added_cond_kwargs['image_embeds'] = image_embeds
    if not output_type == 'latent':
        needs_upcasting = (self.vae.dtype == torch.float16 and self.vae.
            config.force_upcast)
        if needs_upcasting:
            self.upcast_vae()
            latents = latents.to(next(iter(self.vae.post_quant_conv.
                parameters())).dtype)
        if needs_upcasting:
            self.vae.to(dtype=torch.float16)
    self.unet = self.unet.to(memory_format=torch.channels_last)
    self.vae.decoder = self.vae.decoder.to(memory_format=torch.channels_last)
    self.text_encoder = self.text_encoder.to(memory_format=torch.channels_last)
    unet_input_example = {'sample': latent_model_input, 'timestep': dummy,
        'encoder_hidden_states': prompt_embeds, 'added_cond_kwargs':
        added_cond_kwargs}
    vae_decoder_input_example = latents
    if dtype == torch.bfloat16:
        self.unet = ipex.optimize(self.unet.eval(), dtype=torch.bfloat16,
            inplace=True)
        self.vae.decoder = ipex.optimize(self.vae.decoder.eval(), dtype=
            torch.bfloat16, inplace=True)
        self.text_encoder = ipex.optimize(self.text_encoder.eval(), dtype=
            torch.bfloat16, inplace=True)
    elif dtype == torch.float32:
        self.unet = ipex.optimize(self.unet.eval(), dtype=torch.float32,
            inplace=True, level='O1', weights_prepack=True,
            auto_kernel_selection=False)
        self.vae.decoder = ipex.optimize(self.vae.decoder.eval(), dtype=
            torch.float32, inplace=True, level='O1', weights_prepack=True,
            auto_kernel_selection=False)
        self.text_encoder = ipex.optimize(self.text_encoder.eval(), dtype=
            torch.float32, inplace=True, level='O1', weights_prepack=True,
            auto_kernel_selection=False)
    else:
        raise ValueError(
            " The value of 'dtype' should be 'torch.bfloat16' or 'torch.float32' !"
            )
    with torch.cpu.amp.autocast(enabled=dtype == torch.bfloat16
        ), torch.no_grad():
        unet_trace_model = torch.jit.trace(self.unet, example_kwarg_inputs=
            unet_input_example, check_trace=False, strict=False)
        unet_trace_model = torch.jit.freeze(unet_trace_model)
        self.unet.forward = unet_trace_model.forward
    with torch.cpu.amp.autocast(enabled=dtype == torch.bfloat16
        ), torch.no_grad():
        vae_decoder_trace_model = torch.jit.trace(self.vae.decoder,
            vae_decoder_input_example, check_trace=False, strict=False)
        vae_decoder_trace_model = torch.jit.freeze(vae_decoder_trace_model)
        self.vae.decoder.forward = vae_decoder_trace_model.forward

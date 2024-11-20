@torch.no_grad()
def invert(self, image: PipelineImageInput, source_prompt: str='',
    source_guidance_scale=3.5, negative_prompt: str=None, negative_prompt_2:
    str=None, num_inversion_steps: int=50, skip: float=0.15, generator:
    Optional[torch.Generator]=None, crops_coords_top_left: Tuple[int, int]=
    (0, 0), num_zero_noise_steps: int=3, cross_attention_kwargs: Optional[
    Dict[str, Any]]=None):
    """
        The function to the pipeline for image inversion as described by the [LEDITS++
        Paper](https://arxiv.org/abs/2301.12247). If the scheduler is set to [`~schedulers.DDIMScheduler`] the
        inversion proposed by [edit-friendly DPDM](https://arxiv.org/abs/2304.06140) will be performed instead.

         Args:
            image (`PipelineImageInput`):
                Input for the image(s) that are to be edited. Multiple input images have to default to the same aspect
                ratio.
            source_prompt (`str`, defaults to `""`):
                Prompt describing the input image that will be used for guidance during inversion. Guidance is disabled
                if the `source_prompt` is `""`.
            source_guidance_scale (`float`, defaults to `3.5`):
                Strength of guidance during inversion.
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the image generation. If not defined, one has to pass
                `negative_prompt_embeds` instead. Ignored when not using guidance (i.e., ignored if `guidance_scale` is
                less than `1`).
            negative_prompt_2 (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the image generation to be sent to `tokenizer_2` and
                `text_encoder_2`. If not defined, `negative_prompt` is used in both text-encoders
            num_inversion_steps (`int`, defaults to `50`):
                Number of total performed inversion steps after discarding the initial `skip` steps.
            skip (`float`, defaults to `0.15`):
                Portion of initial steps that will be ignored for inversion and subsequent generation. Lower values
                will lead to stronger changes to the input image. `skip` has to be between `0` and `1`.
            generator (`torch.Generator`, *optional*):
                A [`torch.Generator`](https://pytorch.org/docs/stable/generated/torch.Generator.html) to make inversion
                deterministic.
            crops_coords_top_left (`Tuple[int]`, *optional*, defaults to (0, 0)):
                `crops_coords_top_left` can be used to generate an image that appears to be "cropped" from the position
                `crops_coords_top_left` downwards. Favorable, well-centered images are usually achieved by setting
                `crops_coords_top_left` to (0, 0). Part of SDXL's micro-conditioning as explained in section 2.2 of
                [https://huggingface.co/papers/2307.01952](https://huggingface.co/papers/2307.01952).
            num_zero_noise_steps (`int`, defaults to `3`):
                Number of final diffusion steps that will not renoise the current image. If no steps are set to zero
                SD-XL in combination with [`DPMSolverMultistepScheduler`] will produce noise artifacts.
            cross_attention_kwargs (`dict`, *optional*):
                A kwargs dictionary that if specified is passed along to the `AttentionProcessor` as defined under
                `self.processor` in
                [diffusers.models.attention_processor](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py).

        Returns:
            [`~pipelines.ledits_pp.LEditsPPInversionPipelineOutput`]: Output will contain the resized input image(s)
            and respective VAE reconstruction(s).
        """
    self.unet.set_attn_processor(AttnProcessor())
    self.eta = 1.0
    self.scheduler.config.timestep_spacing = 'leading'
    self.scheduler.set_timesteps(int(num_inversion_steps * (1 + skip)))
    self.inversion_steps = self.scheduler.timesteps[-num_inversion_steps:]
    timesteps = self.inversion_steps
    num_images_per_prompt = 1
    device = self._execution_device
    if source_prompt == '':
        source_guidance_scale = 0.0
        do_classifier_free_guidance = False
    else:
        do_classifier_free_guidance = source_guidance_scale > 1.0
    x0, resized = self.encode_image(image, dtype=self.text_encoder_2.dtype)
    width = x0.shape[2] * self.vae_scale_factor
    height = x0.shape[3] * self.vae_scale_factor
    self.size = height, width
    self.batch_size = x0.shape[0]
    text_encoder_lora_scale = cross_attention_kwargs.get('scale', None
        ) if cross_attention_kwargs is not None else None
    if isinstance(source_prompt, str):
        source_prompt = [source_prompt] * self.batch_size
    (negative_prompt_embeds, prompt_embeds, negative_pooled_prompt_embeds,
        edit_pooled_prompt_embeds, _) = (self.encode_prompt(device=device,
        num_images_per_prompt=num_images_per_prompt, negative_prompt=
        negative_prompt, negative_prompt_2=negative_prompt_2,
        editing_prompt=source_prompt, lora_scale=text_encoder_lora_scale,
        enable_edit_guidance=do_classifier_free_guidance))
    if self.text_encoder_2 is None:
        text_encoder_projection_dim = int(negative_pooled_prompt_embeds.
            shape[-1])
    else:
        text_encoder_projection_dim = self.text_encoder_2.config.projection_dim
    add_text_embeds = negative_pooled_prompt_embeds
    add_time_ids = self._get_add_time_ids(self.size, crops_coords_top_left,
        self.size, dtype=negative_prompt_embeds.dtype,
        text_encoder_projection_dim=text_encoder_projection_dim)
    if do_classifier_free_guidance:
        negative_prompt_embeds = torch.cat([negative_prompt_embeds,
            prompt_embeds], dim=0)
        add_text_embeds = torch.cat([add_text_embeds,
            edit_pooled_prompt_embeds], dim=0)
        add_time_ids = torch.cat([add_time_ids, add_time_ids], dim=0)
    negative_prompt_embeds = negative_prompt_embeds.to(device)
    add_text_embeds = add_text_embeds.to(device)
    add_time_ids = add_time_ids.to(device).repeat(self.batch_size *
        num_images_per_prompt, 1)
    if self.vae.dtype == torch.float16 and self.vae.config.force_upcast:
        self.upcast_vae()
        x0_tmp = x0.to(next(iter(self.vae.post_quant_conv.parameters())).dtype)
        image_rec = self.vae.decode(x0_tmp / self.vae.config.scaling_factor,
            return_dict=False, generator=generator)[0]
    elif self.vae.config.force_upcast:
        x0_tmp = x0.to(next(iter(self.vae.post_quant_conv.parameters())).dtype)
        image_rec = self.vae.decode(x0_tmp / self.vae.config.scaling_factor,
            return_dict=False, generator=generator)[0]
    else:
        image_rec = self.vae.decode(x0 / self.vae.config.scaling_factor,
            return_dict=False, generator=generator)[0]
    image_rec = self.image_processor.postprocess(image_rec, output_type='pil')
    variance_noise_shape = num_inversion_steps, *x0.shape
    t_to_idx = {int(v): k for k, v in enumerate(timesteps)}
    xts = torch.zeros(size=variance_noise_shape, device=self.device, dtype=
        negative_prompt_embeds.dtype)
    for t in reversed(timesteps):
        idx = num_inversion_steps - t_to_idx[int(t)] - 1
        noise = randn_tensor(shape=x0.shape, generator=generator, device=
            self.device, dtype=x0.dtype)
        xts[idx] = self.scheduler.add_noise(x0, noise, t.unsqueeze(0))
    xts = torch.cat([x0.unsqueeze(0), xts], dim=0)
    zs = torch.zeros(size=variance_noise_shape, device=self.device, dtype=
        negative_prompt_embeds.dtype)
    self.scheduler.set_timesteps(len(self.scheduler.timesteps))
    for t in self.progress_bar(timesteps):
        idx = num_inversion_steps - t_to_idx[int(t)] - 1
        xt = xts[idx + 1]
        latent_model_input = torch.cat([xt] * 2
            ) if do_classifier_free_guidance else xt
        latent_model_input = self.scheduler.scale_model_input(
            latent_model_input, t)
        added_cond_kwargs = {'text_embeds': add_text_embeds, 'time_ids':
            add_time_ids}
        noise_pred = self.unet(latent_model_input, t, encoder_hidden_states
            =negative_prompt_embeds, cross_attention_kwargs=
            cross_attention_kwargs, added_cond_kwargs=added_cond_kwargs,
            return_dict=False)[0]
        if do_classifier_free_guidance:
            noise_pred_out = noise_pred.chunk(2)
            noise_pred_uncond, noise_pred_text = noise_pred_out[0
                ], noise_pred_out[1]
            noise_pred = noise_pred_uncond + source_guidance_scale * (
                noise_pred_text - noise_pred_uncond)
        xtm1 = xts[idx]
        z, xtm1_corrected = compute_noise(self.scheduler, xtm1, xt, t,
            noise_pred, self.eta)
        zs[idx] = z
        xts[idx] = xtm1_corrected
    self.init_latents = xts[-1]
    zs = zs.flip(0)
    if num_zero_noise_steps > 0:
        zs[-num_zero_noise_steps:] = torch.zeros_like(zs[-
            num_zero_noise_steps:])
    self.zs = zs
    return LEditsPPInversionPipelineOutput(images=resized,
        vae_reconstruction_images=image_rec)

@torch.no_grad()
def invert(self, image: PipelineImageInput, source_prompt: str='',
    source_guidance_scale: float=3.5, num_inversion_steps: int=30, skip:
    float=0.15, generator: Optional[torch.Generator]=None,
    cross_attention_kwargs: Optional[Dict[str, Any]]=None, clip_skip:
    Optional[int]=None, height: Optional[int]=None, width: Optional[int]=
    None, resize_mode: Optional[str]='default', crops_coords: Optional[
    Tuple[int, int, int, int]]=None):
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
            num_inversion_steps (`int`, defaults to `30`):
                Number of total performed inversion steps after discarding the initial `skip` steps.
            skip (`float`, defaults to `0.15`):
                Portion of initial steps that will be ignored for inversion and subsequent generation. Lower values
                will lead to stronger changes to the input image. `skip` has to be between `0` and `1`.
            generator (`torch.Generator`, *optional*):
                A [`torch.Generator`](https://pytorch.org/docs/stable/generated/torch.Generator.html) to make inversion
                deterministic.
            cross_attention_kwargs (`dict`, *optional*):
                A kwargs dictionary that if specified is passed along to the [`AttentionProcessor`] as defined in
                [`self.processor`](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py).
            clip_skip (`int`, *optional*):
                Number of layers to be skipped from CLIP while computing the prompt embeddings. A value of 1 means that
                the output of the pre-final layer will be used for computing the prompt embeddings.
            height (`int`, *optional*, defaults to `None`):
                The height in preprocessed image. If `None`, will use the `get_default_height_width()` to get default
                height.
            width (`int`, *optional*`, defaults to `None`):
                The width in preprocessed. If `None`, will use get_default_height_width()` to get the default width.
            resize_mode (`str`, *optional*, defaults to `default`):
                The resize mode, can be one of `default` or `fill`. If `default`, will resize the image to fit within
                the specified width and height, and it may not maintaining the original aspect ratio. If `fill`, will
                resize the image to fit within the specified width and height, maintaining the aspect ratio, and then
                center the image within the dimensions, filling empty with data from image. If `crop`, will resize the
                image to fit within the specified width and height, maintaining the aspect ratio, and then center the
                image within the dimensions, cropping the excess. Note that resize_mode `fill` and `crop` are only
                supported for PIL image input.
            crops_coords (`List[Tuple[int, int, int, int]]`, *optional*, defaults to `None`):
                The crop coordinates for each image in the batch. If `None`, will not crop the image.

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
    x0, resized = self.encode_image(image, dtype=self.text_encoder.dtype,
        height=height, width=width, resize_mode=resize_mode, crops_coords=
        crops_coords)
    self.batch_size = x0.shape[0]
    image_rec = self.vae.decode(x0 / self.vae.config.scaling_factor,
        return_dict=False, generator=generator)[0]
    image_rec = self.image_processor.postprocess(image_rec, output_type='pil')
    do_classifier_free_guidance = source_guidance_scale > 1.0
    lora_scale = cross_attention_kwargs.get('scale', None
        ) if cross_attention_kwargs is not None else None
    uncond_embedding, text_embeddings, _ = self.encode_prompt(
        num_images_per_prompt=1, device=self.device, negative_prompt=None,
        enable_edit_guidance=do_classifier_free_guidance, editing_prompt=
        source_prompt, lora_scale=lora_scale, clip_skip=clip_skip)
    variance_noise_shape = num_inversion_steps, *x0.shape
    t_to_idx = {int(v): k for k, v in enumerate(timesteps)}
    xts = torch.zeros(size=variance_noise_shape, device=self.device, dtype=
        uncond_embedding.dtype)
    for t in reversed(timesteps):
        idx = num_inversion_steps - t_to_idx[int(t)] - 1
        noise = randn_tensor(shape=x0.shape, generator=generator, device=
            self.device, dtype=x0.dtype)
        xts[idx] = self.scheduler.add_noise(x0, noise, torch.Tensor([t]))
    xts = torch.cat([x0.unsqueeze(0), xts], dim=0)
    self.scheduler.set_timesteps(len(self.scheduler.timesteps))
    zs = torch.zeros(size=variance_noise_shape, device=self.device, dtype=
        uncond_embedding.dtype)
    with self.progress_bar(total=len(timesteps)) as progress_bar:
        for t in timesteps:
            idx = num_inversion_steps - t_to_idx[int(t)] - 1
            xt = xts[idx + 1]
            noise_pred = self.unet(xt, timestep=t, encoder_hidden_states=
                uncond_embedding).sample
            if not source_prompt == '':
                noise_pred_cond = self.unet(xt, timestep=t,
                    encoder_hidden_states=text_embeddings).sample
                noise_pred = noise_pred + source_guidance_scale * (
                    noise_pred_cond - noise_pred)
            xtm1 = xts[idx]
            z, xtm1_corrected = compute_noise(self.scheduler, xtm1, xt, t,
                noise_pred, self.eta)
            zs[idx] = z
            xts[idx] = xtm1_corrected
            progress_bar.update()
    self.init_latents = xts[-1].expand(self.batch_size, -1, -1, -1)
    zs = zs.flip(0)
    self.zs = zs
    return LEditsPPInversionPipelineOutput(images=resized,
        vae_reconstruction_images=image_rec)

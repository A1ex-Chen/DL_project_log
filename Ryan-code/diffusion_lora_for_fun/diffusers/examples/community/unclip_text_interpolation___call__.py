@torch.no_grad()
def __call__(self, start_prompt: str, end_prompt: str, steps: int=5,
    prior_num_inference_steps: int=25, decoder_num_inference_steps: int=25,
    super_res_num_inference_steps: int=7, generator: Optional[Union[torch.
    Generator, List[torch.Generator]]]=None, prior_guidance_scale: float=
    4.0, decoder_guidance_scale: float=8.0, enable_sequential_cpu_offload=
    True, gpu_id=0, output_type: Optional[str]='pil', return_dict: bool=True):
    """
        Function invoked when calling the pipeline for generation.

        Args:
            start_prompt (`str`):
                The prompt to start the image generation interpolation from.
            end_prompt (`str`):
                The prompt to end the image generation interpolation at.
            steps (`int`, *optional*, defaults to 5):
                The number of steps over which to interpolate from start_prompt to end_prompt. The pipeline returns
                the same number of images as this value.
            prior_num_inference_steps (`int`, *optional*, defaults to 25):
                The number of denoising steps for the prior. More denoising steps usually lead to a higher quality
                image at the expense of slower inference.
            decoder_num_inference_steps (`int`, *optional*, defaults to 25):
                The number of denoising steps for the decoder. More denoising steps usually lead to a higher quality
                image at the expense of slower inference.
            super_res_num_inference_steps (`int`, *optional*, defaults to 7):
                The number of denoising steps for super resolution. More denoising steps usually lead to a higher
                quality image at the expense of slower inference.
            generator (`torch.Generator` or `List[torch.Generator]`, *optional*):
                One or a list of [torch generator(s)](https://pytorch.org/docs/stable/generated/torch.Generator.html)
                to make generation deterministic.
            prior_guidance_scale (`float`, *optional*, defaults to 4.0):
                Guidance scale as defined in [Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598).
                `guidance_scale` is defined as `w` of equation 2. of [Imagen
                Paper](https://arxiv.org/pdf/2205.11487.pdf). Guidance scale is enabled by setting `guidance_scale >
                1`. Higher guidance scale encourages to generate images that are closely linked to the text `prompt`,
                usually at the expense of lower image quality.
            decoder_guidance_scale (`float`, *optional*, defaults to 4.0):
                Guidance scale as defined in [Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598).
                `guidance_scale` is defined as `w` of equation 2. of [Imagen
                Paper](https://arxiv.org/pdf/2205.11487.pdf). Guidance scale is enabled by setting `guidance_scale >
                1`. Higher guidance scale encourages to generate images that are closely linked to the text `prompt`,
                usually at the expense of lower image quality.
            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generated image. Choose between
                [PIL](https://pillow.readthedocs.io/en/stable/): `PIL.Image.Image` or `np.array`.
            enable_sequential_cpu_offload (`bool`, *optional*, defaults to `True`):
                If True, offloads all models to CPU using accelerate, significantly reducing memory usage. When called, the pipeline's
                models have their state dicts saved to CPU and then are moved to a `torch.device('meta') and loaded to GPU only
                when their specific submodule has its `forward` method called.
            gpu_id (`int`, *optional*, defaults to `0`):
                The gpu_id to be passed to enable_sequential_cpu_offload. Only works when enable_sequential_cpu_offload is set to True.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~pipelines.ImagePipelineOutput`] instead of a plain tuple.
        """
    if not isinstance(start_prompt, str) or not isinstance(end_prompt, str):
        raise ValueError(
            f'`start_prompt` and `end_prompt` should be of type `str` but got {type(start_prompt)} and {type(end_prompt)} instead'
            )
    if enable_sequential_cpu_offload:
        self.enable_sequential_cpu_offload(gpu_id=gpu_id)
    device = self._execution_device
    inputs = self.tokenizer([start_prompt, end_prompt], padding=
        'max_length', truncation=True, max_length=self.tokenizer.
        model_max_length, return_tensors='pt')
    inputs.to(device)
    text_model_output = self.text_encoder(**inputs)
    text_attention_mask = torch.max(inputs.attention_mask[0], inputs.
        attention_mask[1])
    text_attention_mask = torch.cat([text_attention_mask.unsqueeze(0)] * steps
        ).to(device)
    batch_text_embeds = []
    batch_last_hidden_state = []
    for interp_val in torch.linspace(0, 1, steps):
        text_embeds = slerp(interp_val, text_model_output.text_embeds[0],
            text_model_output.text_embeds[1])
        last_hidden_state = slerp(interp_val, text_model_output.
            last_hidden_state[0], text_model_output.last_hidden_state[1])
        batch_text_embeds.append(text_embeds.unsqueeze(0))
        batch_last_hidden_state.append(last_hidden_state.unsqueeze(0))
    batch_text_embeds = torch.cat(batch_text_embeds)
    batch_last_hidden_state = torch.cat(batch_last_hidden_state)
    text_model_output = CLIPTextModelOutput(text_embeds=batch_text_embeds,
        last_hidden_state=batch_last_hidden_state)
    batch_size = text_model_output[0].shape[0]
    do_classifier_free_guidance = (prior_guidance_scale > 1.0 or 
        decoder_guidance_scale > 1.0)
    prompt_embeds, text_encoder_hidden_states, text_mask = self._encode_prompt(
        prompt=None, device=device, num_images_per_prompt=1,
        do_classifier_free_guidance=do_classifier_free_guidance,
        text_model_output=text_model_output, text_attention_mask=
        text_attention_mask)
    self.prior_scheduler.set_timesteps(prior_num_inference_steps, device=device
        )
    prior_timesteps_tensor = self.prior_scheduler.timesteps
    embedding_dim = self.prior.config.embedding_dim
    prior_latents = self.prepare_latents((batch_size, embedding_dim),
        prompt_embeds.dtype, device, generator, None, self.prior_scheduler)
    for i, t in enumerate(self.progress_bar(prior_timesteps_tensor)):
        latent_model_input = torch.cat([prior_latents] * 2
            ) if do_classifier_free_guidance else prior_latents
        predicted_image_embedding = self.prior(latent_model_input, timestep
            =t, proj_embedding=prompt_embeds, encoder_hidden_states=
            text_encoder_hidden_states, attention_mask=text_mask
            ).predicted_image_embedding
        if do_classifier_free_guidance:
            (predicted_image_embedding_uncond, predicted_image_embedding_text
                ) = predicted_image_embedding.chunk(2)
            predicted_image_embedding = (predicted_image_embedding_uncond +
                prior_guidance_scale * (predicted_image_embedding_text -
                predicted_image_embedding_uncond))
        if i + 1 == prior_timesteps_tensor.shape[0]:
            prev_timestep = None
        else:
            prev_timestep = prior_timesteps_tensor[i + 1]
        prior_latents = self.prior_scheduler.step(predicted_image_embedding,
            timestep=t, sample=prior_latents, generator=generator,
            prev_timestep=prev_timestep).prev_sample
    prior_latents = self.prior.post_process_latents(prior_latents)
    image_embeddings = prior_latents
    text_encoder_hidden_states, additive_clip_time_embeddings = self.text_proj(
        image_embeddings=image_embeddings, prompt_embeds=prompt_embeds,
        text_encoder_hidden_states=text_encoder_hidden_states,
        do_classifier_free_guidance=do_classifier_free_guidance)
    if device.type == 'mps':
        text_mask = text_mask.type(torch.int)
        decoder_text_mask = F.pad(text_mask, (self.text_proj.
            clip_extra_context_tokens, 0), value=1)
        decoder_text_mask = decoder_text_mask.type(torch.bool)
    else:
        decoder_text_mask = F.pad(text_mask, (self.text_proj.
            clip_extra_context_tokens, 0), value=True)
    self.decoder_scheduler.set_timesteps(decoder_num_inference_steps,
        device=device)
    decoder_timesteps_tensor = self.decoder_scheduler.timesteps
    num_channels_latents = self.decoder.config.in_channels
    height = self.decoder.config.sample_size
    width = self.decoder.config.sample_size
    decoder_latents = self.prepare_latents((batch_size,
        num_channels_latents, height, width), text_encoder_hidden_states.
        dtype, device, generator, None, self.decoder_scheduler)
    for i, t in enumerate(self.progress_bar(decoder_timesteps_tensor)):
        latent_model_input = torch.cat([decoder_latents] * 2
            ) if do_classifier_free_guidance else decoder_latents
        noise_pred = self.decoder(sample=latent_model_input, timestep=t,
            encoder_hidden_states=text_encoder_hidden_states, class_labels=
            additive_clip_time_embeddings, attention_mask=decoder_text_mask
            ).sample
        if do_classifier_free_guidance:
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred_uncond, _ = noise_pred_uncond.split(latent_model_input
                .shape[1], dim=1)
            noise_pred_text, predicted_variance = noise_pred_text.split(
                latent_model_input.shape[1], dim=1)
            noise_pred = noise_pred_uncond + decoder_guidance_scale * (
                noise_pred_text - noise_pred_uncond)
            noise_pred = torch.cat([noise_pred, predicted_variance], dim=1)
        if i + 1 == decoder_timesteps_tensor.shape[0]:
            prev_timestep = None
        else:
            prev_timestep = decoder_timesteps_tensor[i + 1]
        decoder_latents = self.decoder_scheduler.step(noise_pred, t,
            decoder_latents, prev_timestep=prev_timestep, generator=generator
            ).prev_sample
    decoder_latents = decoder_latents.clamp(-1, 1)
    image_small = decoder_latents
    self.super_res_scheduler.set_timesteps(super_res_num_inference_steps,
        device=device)
    super_res_timesteps_tensor = self.super_res_scheduler.timesteps
    channels = self.super_res_first.config.in_channels // 2
    height = self.super_res_first.config.sample_size
    width = self.super_res_first.config.sample_size
    super_res_latents = self.prepare_latents((batch_size, channels, height,
        width), image_small.dtype, device, generator, None, self.
        super_res_scheduler)
    if device.type == 'mps':
        image_upscaled = F.interpolate(image_small, size=[height, width])
    else:
        interpolate_antialias = {}
        if 'antialias' in inspect.signature(F.interpolate).parameters:
            interpolate_antialias['antialias'] = True
        image_upscaled = F.interpolate(image_small, size=[height, width],
            mode='bicubic', align_corners=False, **interpolate_antialias)
    for i, t in enumerate(self.progress_bar(super_res_timesteps_tensor)):
        if i == super_res_timesteps_tensor.shape[0] - 1:
            unet = self.super_res_last
        else:
            unet = self.super_res_first
        latent_model_input = torch.cat([super_res_latents, image_upscaled],
            dim=1)
        noise_pred = unet(sample=latent_model_input, timestep=t).sample
        if i + 1 == super_res_timesteps_tensor.shape[0]:
            prev_timestep = None
        else:
            prev_timestep = super_res_timesteps_tensor[i + 1]
        super_res_latents = self.super_res_scheduler.step(noise_pred, t,
            super_res_latents, prev_timestep=prev_timestep, generator=generator
            ).prev_sample
    image = super_res_latents
    image = image * 0.5 + 0.5
    image = image.clamp(0, 1)
    image = image.cpu().permute(0, 2, 3, 1).float().numpy()
    if output_type == 'pil':
        image = self.numpy_to_pil(image)
    if not return_dict:
        return image,
    return ImagePipelineOutput(images=image)

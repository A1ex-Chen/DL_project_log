@torch.no_grad()
def __call__(self, prompt: Optional[Union[str, List[str]]]=None,
    num_images_per_prompt: int=1, prior_num_inference_steps: int=25,
    decoder_num_inference_steps: int=25, super_res_num_inference_steps: int
    =7, generator: Optional[Union[torch.Generator, List[torch.Generator]]]=
    None, prior_latents: Optional[torch.Tensor]=None, decoder_latents:
    Optional[torch.Tensor]=None, super_res_latents: Optional[torch.Tensor]=
    None, text_model_output: Optional[Union[CLIPTextModelOutput, Tuple]]=
    None, text_attention_mask: Optional[torch.Tensor]=None,
    prior_guidance_scale: float=4.0, decoder_guidance_scale: float=8.0,
    output_type: Optional[str]='pil', return_dict: bool=True):
    """
        The call function to the pipeline for generation.

        Args:
            prompt (`str` or `List[str]`):
                The prompt or prompts to guide image generation. This can only be left undefined if `text_model_output`
                and `text_attention_mask` is passed.
            num_images_per_prompt (`int`, *optional*, defaults to 1):
                The number of images to generate per prompt.
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
                A [`torch.Generator`](https://pytorch.org/docs/stable/generated/torch.Generator.html) to make
                generation deterministic.
            prior_latents (`torch.Tensor` of shape (batch size, embeddings dimension), *optional*):
                Pre-generated noisy latents to be used as inputs for the prior.
            decoder_latents (`torch.Tensor` of shape (batch size, channels, height, width), *optional*):
                Pre-generated noisy latents to be used as inputs for the decoder.
            super_res_latents (`torch.Tensor` of shape (batch size, channels, super res height, super res width), *optional*):
                Pre-generated noisy latents to be used as inputs for the decoder.
            prior_guidance_scale (`float`, *optional*, defaults to 4.0):
                A higher guidance scale value encourages the model to generate images closely linked to the text
                `prompt` at the expense of lower image quality. Guidance scale is enabled when `guidance_scale > 1`.
            decoder_guidance_scale (`float`, *optional*, defaults to 4.0):
                A higher guidance scale value encourages the model to generate images closely linked to the text
                `prompt` at the expense of lower image quality. Guidance scale is enabled when `guidance_scale > 1`.
            text_model_output (`CLIPTextModelOutput`, *optional*):
                Pre-defined [`CLIPTextModel`] outputs that can be derived from the text encoder. Pre-defined text
                outputs can be passed for tasks like text embedding interpolations. Make sure to also pass
                `text_attention_mask` in this case. `prompt` can the be left `None`.
            text_attention_mask (`torch.Tensor`, *optional*):
                Pre-defined CLIP text attention mask that can be derived from the tokenizer. Pre-defined text attention
                masks are necessary when passing `text_model_output`.
            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generated image. Choose between `PIL.Image` or `np.array`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~pipelines.ImagePipelineOutput`] instead of a plain tuple.

        Returns:
            [`~pipelines.ImagePipelineOutput`] or `tuple`:
                If `return_dict` is `True`, [`~pipelines.ImagePipelineOutput`] is returned, otherwise a `tuple` is
                returned where the first element is a list with the generated images.
        """
    if prompt is not None:
        if isinstance(prompt, str):
            batch_size = 1
        elif isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            raise ValueError(
                f'`prompt` has to be of type `str` or `list` but is {type(prompt)}'
                )
    else:
        batch_size = text_model_output[0].shape[0]
    device = self._execution_device
    batch_size = batch_size * num_images_per_prompt
    do_classifier_free_guidance = (prior_guidance_scale > 1.0 or 
        decoder_guidance_scale > 1.0)
    prompt_embeds, text_enc_hid_states, text_mask = self._encode_prompt(prompt,
        device, num_images_per_prompt, do_classifier_free_guidance,
        text_model_output, text_attention_mask)
    self.prior_scheduler.set_timesteps(prior_num_inference_steps, device=device
        )
    prior_timesteps_tensor = self.prior_scheduler.timesteps
    embedding_dim = self.prior.config.embedding_dim
    prior_latents = self.prepare_latents((batch_size, embedding_dim),
        prompt_embeds.dtype, device, generator, prior_latents, self.
        prior_scheduler)
    for i, t in enumerate(self.progress_bar(prior_timesteps_tensor)):
        latent_model_input = torch.cat([prior_latents] * 2
            ) if do_classifier_free_guidance else prior_latents
        predicted_image_embedding = self.prior(latent_model_input, timestep
            =t, proj_embedding=prompt_embeds, encoder_hidden_states=
            text_enc_hid_states, attention_mask=text_mask
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
    text_enc_hid_states, additive_clip_time_embeddings = self.text_proj(
        image_embeddings=image_embeddings, prompt_embeds=prompt_embeds,
        text_encoder_hidden_states=text_enc_hid_states,
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
        num_channels_latents, height, width), text_enc_hid_states.dtype,
        device, generator, decoder_latents, self.decoder_scheduler)
    for i, t in enumerate(self.progress_bar(decoder_timesteps_tensor)):
        latent_model_input = torch.cat([decoder_latents] * 2
            ) if do_classifier_free_guidance else decoder_latents
        noise_pred = self.decoder(sample=latent_model_input, timestep=t,
            encoder_hidden_states=text_enc_hid_states, class_labels=
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
        width), image_small.dtype, device, generator, super_res_latents,
        self.super_res_scheduler)
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
    self.maybe_free_model_hooks()
    image = image * 0.5 + 0.5
    image = image.clamp(0, 1)
    image = image.cpu().permute(0, 2, 3, 1).float().numpy()
    if output_type == 'pil':
        image = self.numpy_to_pil(image)
    if not return_dict:
        return image,
    return ImagePipelineOutput(images=image)

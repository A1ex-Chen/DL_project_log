@torch.no_grad()
def __call__(self, image: Optional[Union[PIL.Image.Image, List[PIL.Image.
    Image], torch.FloatTensor]]=None, num_images_per_prompt: int=1,
    decoder_num_inference_steps: int=25, super_res_num_inference_steps: int
    =7, generator: Optional[torch.Generator]=None, decoder_latents:
    Optional[torch.FloatTensor]=None, super_res_latents: Optional[torch.
    FloatTensor]=None, image_embeddings: Optional[torch.Tensor]=None,
    decoder_guidance_scale: float=8.0, output_type: Optional[str]='pil',
    return_dict: bool=True):
    """
        Function invoked when calling the pipeline for generation.

        Args:
            image (`PIL.Image.Image` or `List[PIL.Image.Image]` or `torch.FloatTensor`):
                The image or images to guide the image generation. If you provide a tensor, it needs to comply with the
                configuration of
                [this](https://huggingface.co/fusing/karlo-image-variations-diffusers/blob/main/feature_extractor/preprocessor_config.json)
                `CLIPImageProcessor`. Can be left to `None` only when `image_embeddings` are passed.
            num_images_per_prompt (`int`, *optional*, defaults to 1):
                The number of images to generate per prompt.
            decoder_num_inference_steps (`int`, *optional*, defaults to 25):
                The number of denoising steps for the decoder. More denoising steps usually lead to a higher quality
                image at the expense of slower inference.
            super_res_num_inference_steps (`int`, *optional*, defaults to 7):
                The number of denoising steps for super resolution. More denoising steps usually lead to a higher
                quality image at the expense of slower inference.
            generator (`torch.Generator`, *optional*):
                One or a list of [torch generator(s)](https://pytorch.org/docs/stable/generated/torch.Generator.html)
                to make generation deterministic.
            decoder_latents (`torch.FloatTensor` of shape (batch size, channels, height, width), *optional*):
                Pre-generated noisy latents to be used as inputs for the decoder.
            super_res_latents (`torch.FloatTensor` of shape (batch size, channels, super res height, super res width), *optional*):
                Pre-generated noisy latents to be used as inputs for the decoder.
            decoder_guidance_scale (`float`, *optional*, defaults to 4.0):
                Guidance scale as defined in [Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598).
                `guidance_scale` is defined as `w` of equation 2. of [Imagen
                Paper](https://arxiv.org/pdf/2205.11487.pdf). Guidance scale is enabled by setting `guidance_scale >
                1`. Higher guidance scale encourages to generate images that are closely linked to the text `prompt`,
                usually at the expense of lower image quality.
            image_embeddings (`torch.Tensor`, *optional*):
                Pre-defined image embeddings that can be derived from the image encoder. Pre-defined image embeddings
                can be passed for tasks like image interpolations. `image` can the be left to `None`.
            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generated image. Choose between
                [PIL](https://pillow.readthedocs.io/en/stable/): `PIL.Image.Image` or `np.array`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~pipelines.ImagePipelineOutput`] instead of a plain tuple.
        """
    if image is not None:
        if isinstance(image, PIL.Image.Image):
            batch_size = 1
        elif isinstance(image, list):
            batch_size = len(image)
        else:
            batch_size = image.shape[0]
    else:
        batch_size = image_embeddings.shape[0]
    prompt = [''] * batch_size
    device = self._execution_device
    batch_size = batch_size * num_images_per_prompt
    do_classifier_free_guidance = decoder_guidance_scale > 1.0
    prompt_embeds, text_encoder_hidden_states, text_mask = self._encode_prompt(
        prompt, device, num_images_per_prompt, do_classifier_free_guidance)
    image_embeddings = self._encode_image(image, device,
        num_images_per_prompt, image_embeddings)
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
    num_channels_latents = self.decoder.in_channels
    height = self.decoder.sample_size
    width = self.decoder.sample_size
    if decoder_latents is None:
        decoder_latents = self.prepare_latents((batch_size,
            num_channels_latents, height, width),
            text_encoder_hidden_states.dtype, device, generator,
            decoder_latents, self.decoder_scheduler)
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
    channels = self.super_res_first.in_channels // 2
    height = self.super_res_first.sample_size
    width = self.super_res_first.sample_size
    if super_res_latents is None:
        super_res_latents = self.prepare_latents((batch_size, channels,
            height, width), image_small.dtype, device, generator,
            super_res_latents, self.super_res_scheduler)
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

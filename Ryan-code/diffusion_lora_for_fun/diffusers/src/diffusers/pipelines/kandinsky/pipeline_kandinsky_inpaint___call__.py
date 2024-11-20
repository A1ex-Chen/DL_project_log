@torch.no_grad()
@replace_example_docstring(EXAMPLE_DOC_STRING)
def __call__(self, prompt: Union[str, List[str]], image: Union[torch.Tensor,
    PIL.Image.Image], mask_image: Union[torch.Tensor, PIL.Image.Image, np.
    ndarray], image_embeds: torch.Tensor, negative_image_embeds: torch.
    Tensor, negative_prompt: Optional[Union[str, List[str]]]=None, height:
    int=512, width: int=512, num_inference_steps: int=100, guidance_scale:
    float=4.0, num_images_per_prompt: int=1, generator: Optional[Union[
    torch.Generator, List[torch.Generator]]]=None, latents: Optional[torch.
    Tensor]=None, output_type: Optional[str]='pil', callback: Optional[
    Callable[[int, int, torch.Tensor], None]]=None, callback_steps: int=1,
    return_dict: bool=True):
    """
        Function invoked when calling the pipeline for generation.

        Args:
            prompt (`str` or `List[str]`):
                The prompt or prompts to guide the image generation.
            image (`torch.Tensor`, `PIL.Image.Image` or `np.ndarray`):
                `Image`, or tensor representing an image batch, that will be used as the starting point for the
                process.
            mask_image (`PIL.Image.Image`,`torch.Tensor` or `np.ndarray`):
                `Image`, or a tensor representing an image batch, to mask `image`. White pixels in the mask will be
                repainted, while black pixels will be preserved. You can pass a pytorch tensor as mask only if the
                image you passed is a pytorch tensor, and it should contain one color channel (L) instead of 3, so the
                expected shape would be either `(B, 1, H, W,)`, `(B, H, W)`, `(1, H, W)` or `(H, W)` If image is an PIL
                image or numpy array, mask should also be a either PIL image or numpy array. If it is a PIL image, it
                will be converted to a single channel (luminance) before use. If it is a nummpy array, the expected
                shape is `(H, W)`.
            image_embeds (`torch.Tensor` or `List[torch.Tensor]`):
                The clip image embeddings for text prompt, that will be used to condition the image generation.
            negative_image_embeds (`torch.Tensor` or `List[torch.Tensor]`):
                The clip image embeddings for negative text prompt, will be used to condition the image generation.
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the image generation. Ignored when not using guidance (i.e., ignored
                if `guidance_scale` is less than `1`).
            height (`int`, *optional*, defaults to 512):
                The height in pixels of the generated image.
            width (`int`, *optional*, defaults to 512):
                The width in pixels of the generated image.
            num_inference_steps (`int`, *optional*, defaults to 100):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            guidance_scale (`float`, *optional*, defaults to 4.0):
                Guidance scale as defined in [Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598).
                `guidance_scale` is defined as `w` of equation 2. of [Imagen
                Paper](https://arxiv.org/pdf/2205.11487.pdf). Guidance scale is enabled by setting `guidance_scale >
                1`. Higher guidance scale encourages to generate images that are closely linked to the text `prompt`,
                usually at the expense of lower image quality.
            num_images_per_prompt (`int`, *optional*, defaults to 1):
                The number of images to generate per prompt.
            generator (`torch.Generator` or `List[torch.Generator]`, *optional*):
                One or a list of [torch generator(s)](https://pytorch.org/docs/stable/generated/torch.Generator.html)
                to make generation deterministic.
            latents (`torch.Tensor`, *optional*):
                Pre-generated noisy latents, sampled from a Gaussian distribution, to be used as inputs for image
                generation. Can be used to tweak the same generation with different prompts. If not provided, a latents
                tensor will ge generated by sampling using the supplied random `generator`.
            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generate image. Choose between: `"pil"` (`PIL.Image.Image`), `"np"`
                (`np.array`) or `"pt"` (`torch.Tensor`).
            callback (`Callable`, *optional*):
                A function that calls every `callback_steps` steps during inference. The function is called with the
                following arguments: `callback(step: int, timestep: int, latents: torch.Tensor)`.
            callback_steps (`int`, *optional*, defaults to 1):
                The frequency at which the `callback` function is called. If not specified, the callback is called at
                every step.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~pipelines.ImagePipelineOutput`] instead of a plain tuple.

        Examples:

        Returns:
            [`~pipelines.ImagePipelineOutput`] or `tuple`
        """
    if not self._warn_has_been_called and version.parse(version.parse(
        __version__).base_version) < version.parse('0.23.0.dev0'):
        logger.warning(
            "Please note that the expected format of `mask_image` has recently been changed. Before diffusers == 0.19.0, Kandinsky Inpainting pipelines repainted black pixels and preserved black pixels. As of diffusers==0.19.0 this behavior has been inverted. Now white pixels are repainted and black pixels are preserved. This way, Kandinsky's masking behavior is aligned with Stable Diffusion. THIS means that you HAVE to invert the input mask to have the same behavior as before as explained in https://github.com/huggingface/diffusers/pull/4207. This warning will be surpressed after the first inference call and will be removed in diffusers>0.23.0"
            )
        self._warn_has_been_called = True
    if isinstance(prompt, str):
        batch_size = 1
    elif isinstance(prompt, list):
        batch_size = len(prompt)
    else:
        raise ValueError(
            f'`prompt` has to be of type `str` or `list` but is {type(prompt)}'
            )
    device = self._execution_device
    batch_size = batch_size * num_images_per_prompt
    do_classifier_free_guidance = guidance_scale > 1.0
    prompt_embeds, text_encoder_hidden_states, _ = self._encode_prompt(prompt,
        device, num_images_per_prompt, do_classifier_free_guidance,
        negative_prompt)
    if isinstance(image_embeds, list):
        image_embeds = torch.cat(image_embeds, dim=0)
    if isinstance(negative_image_embeds, list):
        negative_image_embeds = torch.cat(negative_image_embeds, dim=0)
    if do_classifier_free_guidance:
        image_embeds = image_embeds.repeat_interleave(num_images_per_prompt,
            dim=0)
        negative_image_embeds = negative_image_embeds.repeat_interleave(
            num_images_per_prompt, dim=0)
        image_embeds = torch.cat([negative_image_embeds, image_embeds], dim=0
            ).to(dtype=prompt_embeds.dtype, device=device)
    mask_image, image = prepare_mask_and_masked_image(image, mask_image,
        height, width)
    image = image.to(dtype=prompt_embeds.dtype, device=device)
    image = self.movq.encode(image)['latents']
    mask_image = mask_image.to(dtype=prompt_embeds.dtype, device=device)
    image_shape = tuple(image.shape[-2:])
    mask_image = F.interpolate(mask_image, image_shape, mode='nearest')
    mask_image = prepare_mask(mask_image)
    masked_image = image * mask_image
    mask_image = mask_image.repeat_interleave(num_images_per_prompt, dim=0)
    masked_image = masked_image.repeat_interleave(num_images_per_prompt, dim=0)
    if do_classifier_free_guidance:
        mask_image = mask_image.repeat(2, 1, 1, 1)
        masked_image = masked_image.repeat(2, 1, 1, 1)
    self.scheduler.set_timesteps(num_inference_steps, device=device)
    timesteps_tensor = self.scheduler.timesteps
    num_channels_latents = self.movq.config.latent_channels
    sample_height, sample_width = get_new_h_w(height, width, self.
        movq_scale_factor)
    latents = self.prepare_latents((batch_size, num_channels_latents,
        sample_height, sample_width), text_encoder_hidden_states.dtype,
        device, generator, latents, self.scheduler)
    num_channels_mask = mask_image.shape[1]
    num_channels_masked_image = masked_image.shape[1]
    if (num_channels_latents + num_channels_mask +
        num_channels_masked_image != self.unet.config.in_channels):
        raise ValueError(
            f'Incorrect configuration settings! The config of `pipeline.unet`: {self.unet.config} expects {self.unet.config.in_channels} but received `num_channels_latents`: {num_channels_latents} + `num_channels_mask`: {num_channels_mask} + `num_channels_masked_image`: {num_channels_masked_image} = {num_channels_latents + num_channels_masked_image + num_channels_mask}. Please verify the config of `pipeline.unet` or your `mask_image` or `image` input.'
            )
    for i, t in enumerate(self.progress_bar(timesteps_tensor)):
        latent_model_input = torch.cat([latents] * 2
            ) if do_classifier_free_guidance else latents
        latent_model_input = torch.cat([latent_model_input, masked_image,
            mask_image], dim=1)
        added_cond_kwargs = {'text_embeds': prompt_embeds, 'image_embeds':
            image_embeds}
        noise_pred = self.unet(sample=latent_model_input, timestep=t,
            encoder_hidden_states=text_encoder_hidden_states,
            added_cond_kwargs=added_cond_kwargs, return_dict=False)[0]
        if do_classifier_free_guidance:
            noise_pred, variance_pred = noise_pred.split(latents.shape[1],
                dim=1)
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            _, variance_pred_text = variance_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text
                 - noise_pred_uncond)
            noise_pred = torch.cat([noise_pred, variance_pred_text], dim=1)
        if not (hasattr(self.scheduler.config, 'variance_type') and self.
            scheduler.config.variance_type in ['learned', 'learned_range']):
            noise_pred, _ = noise_pred.split(latents.shape[1], dim=1)
        latents = self.scheduler.step(noise_pred, t, latents, generator=
            generator).prev_sample
        if callback is not None and i % callback_steps == 0:
            step_idx = i // getattr(self.scheduler, 'order', 1)
            callback(step_idx, t, latents)
    image = self.movq.decode(latents, force_not_quantize=True)['sample']
    self.maybe_free_model_hooks()
    if output_type not in ['pt', 'np', 'pil']:
        raise ValueError(
            f'Only the output types `pt`, `pil` and `np` are supported not output_type={output_type}'
            )
    if output_type in ['np', 'pil']:
        image = image * 0.5 + 0.5
        image = image.clamp(0, 1)
        image = image.cpu().permute(0, 2, 3, 1).float().numpy()
    if output_type == 'pil':
        image = self.numpy_to_pil(image)
    if not return_dict:
        return image,
    return ImagePipelineOutput(images=image)
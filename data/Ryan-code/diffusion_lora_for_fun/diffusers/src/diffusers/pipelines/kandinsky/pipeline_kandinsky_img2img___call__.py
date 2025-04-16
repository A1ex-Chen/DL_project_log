@torch.no_grad()
@replace_example_docstring(EXAMPLE_DOC_STRING)
def __call__(self, prompt: Union[str, List[str]], image: Union[torch.Tensor,
    PIL.Image.Image, List[torch.Tensor], List[PIL.Image.Image]],
    image_embeds: torch.Tensor, negative_image_embeds: torch.Tensor,
    negative_prompt: Optional[Union[str, List[str]]]=None, height: int=512,
    width: int=512, num_inference_steps: int=100, strength: float=0.3,
    guidance_scale: float=7.0, num_images_per_prompt: int=1, generator:
    Optional[Union[torch.Generator, List[torch.Generator]]]=None,
    output_type: Optional[str]='pil', callback: Optional[Callable[[int, int,
    torch.Tensor], None]]=None, callback_steps: int=1, return_dict: bool=True):
    """
        Function invoked when calling the pipeline for generation.

        Args:
            prompt (`str` or `List[str]`):
                The prompt or prompts to guide the image generation.
            image (`torch.Tensor`, `PIL.Image.Image`):
                `Image`, or tensor representing an image batch, that will be used as the starting point for the
                process.
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
            strength (`float`, *optional*, defaults to 0.3):
                Conceptually, indicates how much to transform the reference `image`. Must be between 0 and 1. `image`
                will be used as a starting point, adding more noise to it the larger the `strength`. The number of
                denoising steps depends on the amount of noise initially added. When `strength` is 1, added noise will
                be maximum and the denoising process will run for the full number of iterations specified in
                `num_inference_steps`. A value of 1, therefore, essentially ignores `image`.
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
    if not isinstance(image, list):
        image = [image]
    if not all(isinstance(i, (PIL.Image.Image, torch.Tensor)) for i in image):
        raise ValueError(
            f'Input is in incorrect format: {[type(i) for i in image]}. Currently, we only support  PIL image and pytorch tensor'
            )
    image = torch.cat([prepare_image(i, width, height) for i in image], dim=0)
    image = image.to(dtype=prompt_embeds.dtype, device=device)
    latents = self.movq.encode(image)['latents']
    latents = latents.repeat_interleave(num_images_per_prompt, dim=0)
    self.scheduler.set_timesteps(num_inference_steps, device=device)
    timesteps_tensor, num_inference_steps = self.get_timesteps(
        num_inference_steps, strength, device)
    latent_timestep = int(self.scheduler.config.num_train_timesteps * strength
        ) - 2
    latent_timestep = torch.tensor([latent_timestep] * batch_size, dtype=
        timesteps_tensor.dtype, device=device)
    num_channels_latents = self.unet.config.in_channels
    height, width = get_new_h_w(height, width, self.movq_scale_factor)
    latents = self.prepare_latents(latents, latent_timestep, (batch_size,
        num_channels_latents, height, width), text_encoder_hidden_states.
        dtype, device, generator, self.scheduler)
    for i, t in enumerate(self.progress_bar(timesteps_tensor)):
        latent_model_input = torch.cat([latents] * 2
            ) if do_classifier_free_guidance else latents
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

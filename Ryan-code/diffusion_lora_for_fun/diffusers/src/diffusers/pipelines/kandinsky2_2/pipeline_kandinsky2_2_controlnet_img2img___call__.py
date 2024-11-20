@torch.no_grad()
def __call__(self, image_embeds: Union[torch.Tensor, List[torch.Tensor]],
    image: Union[torch.Tensor, PIL.Image.Image, List[torch.Tensor], List[
    PIL.Image.Image]], negative_image_embeds: Union[torch.Tensor, List[
    torch.Tensor]], hint: torch.Tensor, height: int=512, width: int=512,
    num_inference_steps: int=100, guidance_scale: float=4.0, strength:
    float=0.3, num_images_per_prompt: int=1, generator: Optional[Union[
    torch.Generator, List[torch.Generator]]]=None, output_type: Optional[
    str]='pil', callback: Optional[Callable[[int, int, torch.Tensor], None]
    ]=None, callback_steps: int=1, return_dict: bool=True):
    """
        Function invoked when calling the pipeline for generation.

        Args:
            image_embeds (`torch.Tensor` or `List[torch.Tensor]`):
                The clip image embeddings for text prompt, that will be used to condition the image generation.
            image (`torch.Tensor`, `PIL.Image.Image`, `np.ndarray`, `List[torch.Tensor]`, `List[PIL.Image.Image]`, or `List[np.ndarray]`):
                `Image`, or tensor representing an image batch, that will be used as the starting point for the
                process. Can also accept image latents as `image`, if passing latents directly, it will not be encoded
                again.
            strength (`float`, *optional*, defaults to 0.8):
                Conceptually, indicates how much to transform the reference `image`. Must be between 0 and 1. `image`
                will be used as a starting point, adding more noise to it the larger the `strength`. The number of
                denoising steps depends on the amount of noise initially added. When `strength` is 1, added noise will
                be maximum and the denoising process will run for the full number of iterations specified in
                `num_inference_steps`. A value of 1, therefore, essentially ignores `image`.
            hint (`torch.Tensor`):
                The controlnet condition.
            negative_image_embeds (`torch.Tensor` or `List[torch.Tensor]`):
                The clip image embeddings for negative text prompt, will be used to condition the image generation.
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
    device = self._execution_device
    do_classifier_free_guidance = guidance_scale > 1.0
    if isinstance(image_embeds, list):
        image_embeds = torch.cat(image_embeds, dim=0)
    if isinstance(negative_image_embeds, list):
        negative_image_embeds = torch.cat(negative_image_embeds, dim=0)
    if isinstance(hint, list):
        hint = torch.cat(hint, dim=0)
    batch_size = image_embeds.shape[0]
    if do_classifier_free_guidance:
        image_embeds = image_embeds.repeat_interleave(num_images_per_prompt,
            dim=0)
        negative_image_embeds = negative_image_embeds.repeat_interleave(
            num_images_per_prompt, dim=0)
        hint = hint.repeat_interleave(num_images_per_prompt, dim=0)
        image_embeds = torch.cat([negative_image_embeds, image_embeds], dim=0
            ).to(dtype=self.unet.dtype, device=device)
        hint = torch.cat([hint, hint], dim=0).to(dtype=self.unet.dtype,
            device=device)
    if not isinstance(image, list):
        image = [image]
    if not all(isinstance(i, (PIL.Image.Image, torch.Tensor)) for i in image):
        raise ValueError(
            f'Input is in incorrect format: {[type(i) for i in image]}. Currently, we only support  PIL image and pytorch tensor'
            )
    image = torch.cat([prepare_image(i, width, height) for i in image], dim=0)
    image = image.to(dtype=image_embeds.dtype, device=device)
    latents = self.movq.encode(image)['latents']
    latents = latents.repeat_interleave(num_images_per_prompt, dim=0)
    self.scheduler.set_timesteps(num_inference_steps, device=device)
    timesteps, num_inference_steps = self.get_timesteps(num_inference_steps,
        strength, device)
    latent_timestep = timesteps[:1].repeat(batch_size * num_images_per_prompt)
    height, width = downscale_height_and_width(height, width, self.
        movq_scale_factor)
    latents = self.prepare_latents(latents, latent_timestep, batch_size,
        num_images_per_prompt, image_embeds.dtype, device, generator)
    for i, t in enumerate(self.progress_bar(timesteps)):
        latent_model_input = torch.cat([latents] * 2
            ) if do_classifier_free_guidance else latents
        added_cond_kwargs = {'image_embeds': image_embeds, 'hint': hint}
        noise_pred = self.unet(sample=latent_model_input, timestep=t,
            encoder_hidden_states=None, added_cond_kwargs=added_cond_kwargs,
            return_dict=False)[0]
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
            generator)[0]
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

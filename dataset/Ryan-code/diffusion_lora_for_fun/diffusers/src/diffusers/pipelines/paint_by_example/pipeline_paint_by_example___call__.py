@torch.no_grad()
def __call__(self, example_image: Union[torch.Tensor, PIL.Image.Image],
    image: Union[torch.Tensor, PIL.Image.Image], mask_image: Union[torch.
    Tensor, PIL.Image.Image], height: Optional[int]=None, width: Optional[
    int]=None, num_inference_steps: int=50, guidance_scale: float=5.0,
    negative_prompt: Optional[Union[str, List[str]]]=None,
    num_images_per_prompt: Optional[int]=1, eta: float=0.0, generator:
    Optional[Union[torch.Generator, List[torch.Generator]]]=None, latents:
    Optional[torch.Tensor]=None, output_type: Optional[str]='pil',
    return_dict: bool=True, callback: Optional[Callable[[int, int, torch.
    Tensor], None]]=None, callback_steps: int=1):
    """
        The call function to the pipeline for generation.

        Args:
            example_image (`torch.Tensor` or `PIL.Image.Image` or `List[PIL.Image.Image]`):
                An example image to guide image generation.
            image (`torch.Tensor` or `PIL.Image.Image` or `List[PIL.Image.Image]`):
                `Image` or tensor representing an image batch to be inpainted (parts of the image are masked out with
                `mask_image` and repainted according to `prompt`).
            mask_image (`torch.Tensor` or `PIL.Image.Image` or `List[PIL.Image.Image]`):
                `Image` or tensor representing an image batch to mask `image`. White pixels in the mask are repainted,
                while black pixels are preserved. If `mask_image` is a PIL image, it is converted to a single channel
                (luminance) before use. If it's a tensor, it should contain one color channel (L) instead of 3, so the
                expected shape would be `(B, H, W, 1)`.
            height (`int`, *optional*, defaults to `self.unet.config.sample_size * self.vae_scale_factor`):
                The height in pixels of the generated image.
            width (`int`, *optional*, defaults to `self.unet.config.sample_size * self.vae_scale_factor`):
                The width in pixels of the generated image.
            num_inference_steps (`int`, *optional*, defaults to 50):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            guidance_scale (`float`, *optional*, defaults to 7.5):
                A higher guidance scale value encourages the model to generate images closely linked to the text
                `prompt` at the expense of lower image quality. Guidance scale is enabled when `guidance_scale > 1`.
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts to guide what to not include in image generation. If not defined, you need to
                pass `negative_prompt_embeds` instead. Ignored when not using guidance (`guidance_scale < 1`).
            num_images_per_prompt (`int`, *optional*, defaults to 1):
                The number of images to generate per prompt.
            eta (`float`, *optional*, defaults to 0.0):
                Corresponds to parameter eta (η) from the [DDIM](https://arxiv.org/abs/2010.02502) paper. Only applies
                to the [`~schedulers.DDIMScheduler`], and is ignored in other schedulers.
            generator (`torch.Generator` or `List[torch.Generator]`, *optional*):
                A [`torch.Generator`](https://pytorch.org/docs/stable/generated/torch.Generator.html) to make
                generation deterministic.
            latents (`torch.Tensor`, *optional*):
                Pre-generated noisy latents sampled from a Gaussian distribution, to be used as inputs for image
                generation. Can be used to tweak the same generation with different prompts. If not provided, a latents
                tensor is generated by sampling using the supplied random `generator`.
            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generated image. Choose between `PIL.Image` or `np.array`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] instead of a
                plain tuple.
            callback (`Callable`, *optional*):
                A function that calls every `callback_steps` steps during inference. The function is called with the
                following arguments: `callback(step: int, timestep: int, latents: torch.Tensor)`.
            callback_steps (`int`, *optional*, defaults to 1):
                The frequency at which the `callback` function is called. If not specified, the callback is called at
                every step.

        Example:

        ```py
        >>> import PIL
        >>> import requests
        >>> import torch
        >>> from io import BytesIO
        >>> from diffusers import PaintByExamplePipeline


        >>> def download_image(url):
        ...     response = requests.get(url)
        ...     return PIL.Image.open(BytesIO(response.content)).convert("RGB")


        >>> img_url = (
        ...     "https://raw.githubusercontent.com/Fantasy-Studio/Paint-by-Example/main/examples/image/example_1.png"
        ... )
        >>> mask_url = (
        ...     "https://raw.githubusercontent.com/Fantasy-Studio/Paint-by-Example/main/examples/mask/example_1.png"
        ... )
        >>> example_url = "https://raw.githubusercontent.com/Fantasy-Studio/Paint-by-Example/main/examples/reference/example_1.jpg"

        >>> init_image = download_image(img_url).resize((512, 512))
        >>> mask_image = download_image(mask_url).resize((512, 512))
        >>> example_image = download_image(example_url).resize((512, 512))

        >>> pipe = PaintByExamplePipeline.from_pretrained(
        ...     "Fantasy-Studio/Paint-by-Example",
        ...     torch_dtype=torch.float16,
        ... )
        >>> pipe = pipe.to("cuda")

        >>> image = pipe(image=init_image, mask_image=mask_image, example_image=example_image).images[0]
        >>> image
        ```

        Returns:
            [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] or `tuple`:
                If `return_dict` is `True`, [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] is returned,
                otherwise a `tuple` is returned where the first element is a list with the generated images and the
                second element is a list of `bool`s indicating whether the corresponding generated image contains
                "not-safe-for-work" (nsfw) content.
        """
    if isinstance(image, PIL.Image.Image):
        batch_size = 1
    elif isinstance(image, list):
        batch_size = len(image)
    else:
        batch_size = image.shape[0]
    device = self._execution_device
    do_classifier_free_guidance = guidance_scale > 1.0
    mask, masked_image = prepare_mask_and_masked_image(image, mask_image)
    height, width = masked_image.shape[-2:]
    self.check_inputs(example_image, height, width, callback_steps)
    image_embeddings = self._encode_image(example_image, device,
        num_images_per_prompt, do_classifier_free_guidance)
    self.scheduler.set_timesteps(num_inference_steps, device=device)
    timesteps = self.scheduler.timesteps
    num_channels_latents = self.vae.config.latent_channels
    latents = self.prepare_latents(batch_size * num_images_per_prompt,
        num_channels_latents, height, width, image_embeddings.dtype, device,
        generator, latents)
    mask, masked_image_latents = self.prepare_mask_latents(mask,
        masked_image, batch_size * num_images_per_prompt, height, width,
        image_embeddings.dtype, device, generator, do_classifier_free_guidance)
    num_channels_mask = mask.shape[1]
    num_channels_masked_image = masked_image_latents.shape[1]
    if (num_channels_latents + num_channels_mask +
        num_channels_masked_image != self.unet.config.in_channels):
        raise ValueError(
            f'Incorrect configuration settings! The config of `pipeline.unet`: {self.unet.config} expects {self.unet.config.in_channels} but received `num_channels_latents`: {num_channels_latents} + `num_channels_mask`: {num_channels_mask} + `num_channels_masked_image`: {num_channels_masked_image} = {num_channels_latents + num_channels_masked_image + num_channels_mask}. Please verify the config of `pipeline.unet` or your `mask_image` or `image` input.'
            )
    extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)
    num_warmup_steps = len(timesteps
        ) - num_inference_steps * self.scheduler.order
    with self.progress_bar(total=num_inference_steps) as progress_bar:
        for i, t in enumerate(timesteps):
            latent_model_input = torch.cat([latents] * 2
                ) if do_classifier_free_guidance else latents
            latent_model_input = self.scheduler.scale_model_input(
                latent_model_input, t)
            latent_model_input = torch.cat([latent_model_input,
                masked_image_latents, mask], dim=1)
            noise_pred = self.unet(latent_model_input, t,
                encoder_hidden_states=image_embeddings).sample
            if do_classifier_free_guidance:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (
                    noise_pred_text - noise_pred_uncond)
            latents = self.scheduler.step(noise_pred, t, latents, **
                extra_step_kwargs).prev_sample
            if i == len(timesteps) - 1 or i + 1 > num_warmup_steps and (i + 1
                ) % self.scheduler.order == 0:
                progress_bar.update()
                if callback is not None and i % callback_steps == 0:
                    step_idx = i // getattr(self.scheduler, 'order', 1)
                    callback(step_idx, t, latents)
    self.maybe_free_model_hooks()
    if not output_type == 'latent':
        image = self.vae.decode(latents / self.vae.config.scaling_factor,
            return_dict=False)[0]
        image, has_nsfw_concept = self.run_safety_checker(image, device,
            image_embeddings.dtype)
    else:
        image = latents
        has_nsfw_concept = None
    if has_nsfw_concept is None:
        do_denormalize = [True] * image.shape[0]
    else:
        do_denormalize = [(not has_nsfw) for has_nsfw in has_nsfw_concept]
    image = self.image_processor.postprocess(image, output_type=output_type,
        do_denormalize=do_denormalize)
    if not return_dict:
        return image, has_nsfw_concept
    return StableDiffusionPipelineOutput(images=image,
        nsfw_content_detected=has_nsfw_concept)

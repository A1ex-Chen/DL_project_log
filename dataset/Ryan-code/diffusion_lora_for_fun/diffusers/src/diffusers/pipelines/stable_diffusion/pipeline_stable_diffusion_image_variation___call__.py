@torch.no_grad()
def __call__(self, image: Union[PIL.Image.Image, List[PIL.Image.Image],
    torch.Tensor], height: Optional[int]=None, width: Optional[int]=None,
    num_inference_steps: int=50, guidance_scale: float=7.5,
    num_images_per_prompt: Optional[int]=1, eta: float=0.0, generator:
    Optional[Union[torch.Generator, List[torch.Generator]]]=None, latents:
    Optional[torch.Tensor]=None, output_type: Optional[str]='pil',
    return_dict: bool=True, callback: Optional[Callable[[int, int, torch.
    Tensor], None]]=None, callback_steps: int=1):
    """
        The call function to the pipeline for generation.

        Args:
            image (`PIL.Image.Image` or `List[PIL.Image.Image]` or `torch.Tensor`):
                Image or images to guide image generation. If you provide a tensor, it needs to be compatible with
                [`CLIPImageProcessor`](https://huggingface.co/lambdalabs/sd-image-variations-diffusers/blob/main/feature_extractor/preprocessor_config.json).
            height (`int`, *optional*, defaults to `self.unet.config.sample_size * self.vae_scale_factor`):
                The height in pixels of the generated image.
            width (`int`, *optional*, defaults to `self.unet.config.sample_size * self.vae_scale_factor`):
                The width in pixels of the generated image.
            num_inference_steps (`int`, *optional*, defaults to 50):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference. This parameter is modulated by `strength`.
            guidance_scale (`float`, *optional*, defaults to 7.5):
                A higher guidance scale value encourages the model to generate images closely linked to the text
                `prompt` at the expense of lower image quality. Guidance scale is enabled when `guidance_scale > 1`.
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

        Returns:
            [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] or `tuple`:
                If `return_dict` is `True`, [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] is returned,
                otherwise a `tuple` is returned where the first element is a list with the generated images and the
                second element is a list of `bool`s indicating whether the corresponding generated image contains
                "not-safe-for-work" (nsfw) content.

        Examples:

        ```py
        from diffusers import StableDiffusionImageVariationPipeline
        from PIL import Image
        from io import BytesIO
        import requests

        pipe = StableDiffusionImageVariationPipeline.from_pretrained(
            "lambdalabs/sd-image-variations-diffusers", revision="v2.0"
        )
        pipe = pipe.to("cuda")

        url = "https://lh3.googleusercontent.com/y-iFOHfLTwkuQSUegpwDdgKmOjRSTvPxat63dQLB25xkTs4lhIbRUFeNBWZzYf370g=s1200"

        response = requests.get(url)
        image = Image.open(BytesIO(response.content)).convert("RGB")

        out = pipe(image, num_images_per_prompt=3, guidance_scale=15)
        out["images"][0].save("result.jpg")
        ```
        """
    height = height or self.unet.config.sample_size * self.vae_scale_factor
    width = width or self.unet.config.sample_size * self.vae_scale_factor
    self.check_inputs(image, height, width, callback_steps)
    if isinstance(image, PIL.Image.Image):
        batch_size = 1
    elif isinstance(image, list):
        batch_size = len(image)
    else:
        batch_size = image.shape[0]
    device = self._execution_device
    do_classifier_free_guidance = guidance_scale > 1.0
    image_embeddings = self._encode_image(image, device,
        num_images_per_prompt, do_classifier_free_guidance)
    self.scheduler.set_timesteps(num_inference_steps, device=device)
    timesteps = self.scheduler.timesteps
    num_channels_latents = self.unet.config.in_channels
    latents = self.prepare_latents(batch_size * num_images_per_prompt,
        num_channels_latents, height, width, image_embeddings.dtype, device,
        generator, latents)
    extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)
    num_warmup_steps = len(timesteps
        ) - num_inference_steps * self.scheduler.order
    with self.progress_bar(total=num_inference_steps) as progress_bar:
        for i, t in enumerate(timesteps):
            latent_model_input = torch.cat([latents] * 2
                ) if do_classifier_free_guidance else latents
            latent_model_input = self.scheduler.scale_model_input(
                latent_model_input, t)
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
    self.maybe_free_model_hooks()
    if not return_dict:
        return image, has_nsfw_concept
    return StableDiffusionPipelineOutput(images=image,
        nsfw_content_detected=has_nsfw_concept)

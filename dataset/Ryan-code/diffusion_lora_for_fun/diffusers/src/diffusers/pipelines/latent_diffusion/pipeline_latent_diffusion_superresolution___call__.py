@torch.no_grad()
def __call__(self, image: Union[torch.Tensor, PIL.Image.Image]=None,
    batch_size: Optional[int]=1, num_inference_steps: Optional[int]=100,
    eta: Optional[float]=0.0, generator: Optional[Union[torch.Generator,
    List[torch.Generator]]]=None, output_type: Optional[str]='pil',
    return_dict: bool=True) ->Union[Tuple, ImagePipelineOutput]:
    """
        The call function to the pipeline for generation.

        Args:
            image (`torch.Tensor` or `PIL.Image.Image`):
                `Image` or tensor representing an image batch to be used as the starting point for the process.
            batch_size (`int`, *optional*, defaults to 1):
                Number of images to generate.
            num_inference_steps (`int`, *optional*, defaults to 100):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            eta (`float`, *optional*, defaults to 0.0):
                Corresponds to parameter eta (η) from the [DDIM](https://arxiv.org/abs/2010.02502) paper. Only applies
                to the [`~schedulers.DDIMScheduler`], and is ignored in other schedulers.
            generator (`torch.Generator` or `List[torch.Generator]`, *optional*):
                A [`torch.Generator`](https://pytorch.org/docs/stable/generated/torch.Generator.html) to make
                generation deterministic.
            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generated image. Choose between `PIL.Image` or `np.array`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`ImagePipelineOutput`] instead of a plain tuple.

        Example:

        ```py
        >>> import requests
        >>> from PIL import Image
        >>> from io import BytesIO
        >>> from diffusers import LDMSuperResolutionPipeline
        >>> import torch

        >>> # load model and scheduler
        >>> pipeline = LDMSuperResolutionPipeline.from_pretrained("CompVis/ldm-super-resolution-4x-openimages")
        >>> pipeline = pipeline.to("cuda")

        >>> # let's download an  image
        >>> url = (
        ...     "https://user-images.githubusercontent.com/38061659/199705896-b48e17b8-b231-47cd-a270-4ffa5a93fa3e.png"
        ... )
        >>> response = requests.get(url)
        >>> low_res_img = Image.open(BytesIO(response.content)).convert("RGB")
        >>> low_res_img = low_res_img.resize((128, 128))

        >>> # run pipeline in inference (sample random noise and denoise)
        >>> upscaled_image = pipeline(low_res_img, num_inference_steps=100, eta=1).images[0]
        >>> # save image
        >>> upscaled_image.save("ldm_generated_image.png")
        ```

        Returns:
            [`~pipelines.ImagePipelineOutput`] or `tuple`:
                If `return_dict` is `True`, [`~pipelines.ImagePipelineOutput`] is returned, otherwise a `tuple` is
                returned where the first element is a list with the generated images
        """
    if isinstance(image, PIL.Image.Image):
        batch_size = 1
    elif isinstance(image, torch.Tensor):
        batch_size = image.shape[0]
    else:
        raise ValueError(
            f'`image` has to be of type `PIL.Image.Image` or `torch.Tensor` but is {type(image)}'
            )
    if isinstance(image, PIL.Image.Image):
        image = preprocess(image)
    height, width = image.shape[-2:]
    latents_shape = (batch_size, self.unet.config.in_channels // 2, height,
        width)
    latents_dtype = next(self.unet.parameters()).dtype
    latents = randn_tensor(latents_shape, generator=generator, device=self.
        device, dtype=latents_dtype)
    image = image.to(device=self.device, dtype=latents_dtype)
    self.scheduler.set_timesteps(num_inference_steps, device=self.device)
    timesteps_tensor = self.scheduler.timesteps
    latents = latents * self.scheduler.init_noise_sigma
    accepts_eta = 'eta' in set(inspect.signature(self.scheduler.step).
        parameters.keys())
    extra_kwargs = {}
    if accepts_eta:
        extra_kwargs['eta'] = eta
    for t in self.progress_bar(timesteps_tensor):
        latents_input = torch.cat([latents, image], dim=1)
        latents_input = self.scheduler.scale_model_input(latents_input, t)
        noise_pred = self.unet(latents_input, t).sample
        latents = self.scheduler.step(noise_pred, t, latents, **extra_kwargs
            ).prev_sample
    image = self.vqvae.decode(latents).sample
    image = torch.clamp(image, -1.0, 1.0)
    image = image / 2 + 0.5
    image = image.cpu().permute(0, 2, 3, 1).numpy()
    if output_type == 'pil':
        image = self.numpy_to_pil(image)
    if not return_dict:
        return image,
    return ImagePipelineOutput(images=image)

@torch.no_grad()
def __call__(self, batch_size: int=1, generator: Optional[Union[torch.
    Generator, List[torch.Generator]]]=None, eta: float=0.0,
    num_inference_steps: int=50, output_type: Optional[str]='pil',
    return_dict: bool=True, **kwargs) ->Union[Tuple, ImagePipelineOutput]:
    """
        The call function to the pipeline for generation.

        Args:
            batch_size (`int`, *optional*, defaults to 1):
                Number of images to generate.
            generator (`torch.Generator`, *optional*):
                A [`torch.Generator`](https://pytorch.org/docs/stable/generated/torch.Generator.html) to make
                generation deterministic.
            num_inference_steps (`int`, *optional*, defaults to 50):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generated image. Choose between `PIL.Image` or `np.array`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~pipelines.ImagePipelineOutput`] instead of a plain tuple.

        Example:

        ```py
        >>> from diffusers import LDMPipeline

        >>> # load model and scheduler
        >>> pipe = LDMPipeline.from_pretrained("CompVis/ldm-celebahq-256")

        >>> # run pipeline in inference (sample random noise and denoise)
        >>> image = pipe().images[0]
        ```

        Returns:
            [`~pipelines.ImagePipelineOutput`] or `tuple`:
                If `return_dict` is `True`, [`~pipelines.ImagePipelineOutput`] is returned, otherwise a `tuple` is
                returned where the first element is a list with the generated images
        """
    latents = randn_tensor((batch_size, self.unet.config.in_channels, self.
        unet.config.sample_size, self.unet.config.sample_size), generator=
        generator)
    latents = latents.to(self.device)
    latents = latents * self.scheduler.init_noise_sigma
    self.scheduler.set_timesteps(num_inference_steps)
    accepts_eta = 'eta' in set(inspect.signature(self.scheduler.step).
        parameters.keys())
    extra_kwargs = {}
    if accepts_eta:
        extra_kwargs['eta'] = eta
    for t in self.progress_bar(self.scheduler.timesteps):
        latent_model_input = self.scheduler.scale_model_input(latents, t)
        noise_prediction = self.unet(latent_model_input, t).sample
        latents = self.scheduler.step(noise_prediction, t, latents, **
            extra_kwargs).prev_sample
    latents = latents / self.vqvae.config.scaling_factor
    image = self.vqvae.decode(latents).sample
    image = (image / 2 + 0.5).clamp(0, 1)
    image = image.cpu().permute(0, 2, 3, 1).numpy()
    if output_type == 'pil':
        image = self.numpy_to_pil(image)
    if not return_dict:
        return image,
    return ImagePipelineOutput(images=image)

@torch.no_grad()
def __call__(self, batch_size: int=1, generator: Optional[torch.Generator]=
    None, num_inference_steps: int=50, output_type: Optional[str]='pil',
    return_dict: bool=True, **kwargs) ->Union[ImagePipelineOutput, Tuple]:
    """
        Args:
            batch_size (`int`, *optional*, defaults to 1):
                The number of images to generate.
            generator (`torch.Generator`, *optional*):
                A [torch generator](https://pytorch.org/docs/stable/generated/torch.Generator.html) to make generation
                deterministic.
            eta (`float`, *optional*, defaults to 0.0):
                The eta parameter which controls the scale of the variance (0 is DDIM and 1 is one type of DDPM).
            num_inference_steps (`int`, *optional*, defaults to 50):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generate image. Choose between
                [PIL](https://pillow.readthedocs.io/en/stable/): `PIL.Image.Image` or `np.array`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~pipeline_utils.ImagePipelineOutput`] instead of a plain tuple.

        Returns:
            [`~pipeline_utils.ImagePipelineOutput`] or `tuple`: [`~pipelines.utils.ImagePipelineOutput`] if
            `return_dict` is True, otherwise a `tuple. When returning a tuple, the first element is a list with the
            generated images.
        """
    image = torch.randn((batch_size, self.unet.config.in_channels, self.
        unet.config.sample_size, self.unet.config.sample_size), generator=
        generator)
    image = image.to(self.device)
    self.scheduler.set_timesteps(num_inference_steps)
    for t in self.progress_bar(self.scheduler.timesteps):
        model_output = self.unet(image, t).sample
        image = self.scheduler.step(model_output, t, image).prev_sample
    image = (image / 2 + 0.5).clamp(0, 1)
    image = image.cpu().permute(0, 2, 3, 1).numpy()
    if output_type == 'pil':
        image = self.numpy_to_pil(image)
    if not return_dict:
        return (image,), 'This is a local test'
    return ImagePipelineOutput(images=image), 'This is a local test'

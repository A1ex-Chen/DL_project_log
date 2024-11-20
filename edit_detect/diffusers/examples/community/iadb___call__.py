@torch.no_grad()
def __call__(self, batch_size: int=1, generator: Optional[Union[torch.
    Generator, List[torch.Generator]]]=None, num_inference_steps: int=50,
    output_type: Optional[str]='pil', return_dict: bool=True) ->Union[
    ImagePipelineOutput, Tuple]:
    """
        Args:
            batch_size (`int`, *optional*, defaults to 1):
                The number of images to generate.
            num_inference_steps (`int`, *optional*, defaults to 50):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generate image. Choose between
                [PIL](https://pillow.readthedocs.io/en/stable/): `PIL.Image.Image` or `np.array`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~pipelines.ImagePipelineOutput`] instead of a plain tuple.

        Returns:
            [`~pipelines.ImagePipelineOutput`] or `tuple`: [`~pipelines.utils.ImagePipelineOutput`] if `return_dict` is
            True, otherwise a `tuple. When returning a tuple, the first element is a list with the generated images.
        """
    if isinstance(self.unet.config.sample_size, int):
        image_shape = (batch_size, self.unet.config.in_channels, self.unet.
            config.sample_size, self.unet.config.sample_size)
    else:
        image_shape = (batch_size, self.unet.config.in_channels, *self.unet
            .config.sample_size)
    if isinstance(generator, list) and len(generator) != batch_size:
        raise ValueError(
            f'You have passed a list of generators of length {len(generator)}, but requested an effective batch size of {batch_size}. Make sure the batch size matches the length of the generators.'
            )
    image = torch.randn(image_shape, generator=generator, device=self.
        device, dtype=self.unet.dtype)
    self.scheduler.set_timesteps(num_inference_steps)
    x_alpha = image.clone()
    for t in self.progress_bar(range(num_inference_steps)):
        alpha = t / num_inference_steps
        model_output = self.unet(x_alpha, torch.tensor(alpha, device=
            x_alpha.device)).sample
        x_alpha = self.scheduler.step(model_output, t, x_alpha)
    image = (x_alpha * 0.5 + 0.5).clamp(0, 1)
    image = image.cpu().permute(0, 2, 3, 1).numpy()
    if output_type == 'pil':
        image = self.numpy_to_pil(image)
    if not return_dict:
        return image,
    return ImagePipelineOutput(images=image)

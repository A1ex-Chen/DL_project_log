@torch.no_grad()
def __call__(self, batch_size: int=1, generator: Optional[Union[torch.
    Generator, List[torch.Generator]]]=None, num_inference_steps: int=1000,
    output_type: Optional[str]='pil', return_dict: bool=True) ->Union[
    ImagePipelineOutput, Tuple]:
    """
        Args:
            batch_size (`int`, *optional*, defaults to 1):
                The number of images to generate.
            generator (`torch.Generator`, *optional*):
                One or a list of [torch generator(s)](https://pytorch.org/docs/stable/generated/torch.Generator.html)
                to make generation deterministic.
            num_inference_steps (`int`, *optional*, defaults to 1000):
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
    if isinstance(self.unet.sample_size, int):
        image_shape = (batch_size, self.unet.in_channels, self.unet.
            sample_size, self.unet.sample_size)
    else:
        image_shape = batch_size, self.unet.in_channels, *self.unet.sample_size
    if self.device.type == 'mps':
        image = randn_tensor(image_shape, generator=generator)
        image = image.to(self.device)
    else:
        image = randn_tensor(image_shape, generator=generator, device=self.
            device)
    self.scheduler.set_timesteps(num_inference_steps)
    for t in self.progress_bar(self.scheduler.timesteps):
        model_output = self.unet(image, t).sample
        image = self.scheduler.step(model_output, t, image, generator=generator
            ).prev_sample
    image = (image / 2 + 0.5).clamp(0, 1)
    image = image.cpu().permute(0, 2, 3, 1).numpy()
    if output_type == 'pil':
        image = self.numpy_to_pil(image)
    if not return_dict:
        return image,
    return ImagePipelineOutput(images=image)

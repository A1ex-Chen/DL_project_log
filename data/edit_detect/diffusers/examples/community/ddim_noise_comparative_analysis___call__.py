@torch.no_grad()
def __call__(self, image: Union[torch.Tensor, PIL.Image.Image]=None,
    strength: float=0.8, batch_size: int=1, generator: Optional[Union[torch
    .Generator, List[torch.Generator]]]=None, eta: float=0.0,
    num_inference_steps: int=50, use_clipped_model_output: Optional[bool]=
    None, output_type: Optional[str]='pil', return_dict: bool=True) ->Union[
    ImagePipelineOutput, Tuple]:
    """
        Args:
            image (`torch.Tensor` or `PIL.Image.Image`):
                `Image`, or tensor representing an image batch, that will be used as the starting point for the
                process.
            strength (`float`, *optional*, defaults to 0.8):
                Conceptually, indicates how much to transform the reference `image`. Must be between 0 and 1. `image`
                will be used as a starting point, adding more noise to it the larger the `strength`. The number of
                denoising steps depends on the amount of noise initially added. When `strength` is 1, added noise will
                be maximum and the denoising process will run for the full number of iterations specified in
                `num_inference_steps`. A value of 1, therefore, essentially ignores `image`.
            batch_size (`int`, *optional*, defaults to 1):
                The number of images to generate.
            generator (`torch.Generator`, *optional*):
                One or a list of [torch generator(s)](https://pytorch.org/docs/stable/generated/torch.Generator.html)
                to make generation deterministic.
            eta (`float`, *optional*, defaults to 0.0):
                The eta parameter which controls the scale of the variance (0 is DDIM and 1 is one type of DDPM).
            num_inference_steps (`int`, *optional*, defaults to 50):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            use_clipped_model_output (`bool`, *optional*, defaults to `None`):
                if `True` or `False`, see documentation for `DDIMScheduler.step`. If `None`, nothing is passed
                downstream to the scheduler. So use `None` for schedulers which don't support this argument.
            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generate image. Choose between
                [PIL](https://pillow.readthedocs.io/en/stable/): `PIL.Image.Image` or `np.array`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~pipelines.ImagePipelineOutput`] instead of a plain tuple.

        Returns:
            [`~pipelines.ImagePipelineOutput`] or `tuple`: [`~pipelines.utils.ImagePipelineOutput`] if `return_dict` is
            True, otherwise a `tuple. When returning a tuple, the first element is a list with the generated images.
        """
    self.check_inputs(strength)
    image = preprocess(image)
    self.scheduler.set_timesteps(num_inference_steps, device=self.device)
    timesteps, num_inference_steps = self.get_timesteps(num_inference_steps,
        strength, self.device)
    latent_timestep = timesteps[:1].repeat(batch_size)
    latents = self.prepare_latents(image, latent_timestep, batch_size, self
        .unet.dtype, self.device, generator)
    image = latents
    for t in self.progress_bar(timesteps):
        model_output = self.unet(image, t).sample
        image = self.scheduler.step(model_output, t, image, eta=eta,
            use_clipped_model_output=use_clipped_model_output, generator=
            generator).prev_sample
    image = (image / 2 + 0.5).clamp(0, 1)
    image = image.cpu().permute(0, 2, 3, 1).numpy()
    if output_type == 'pil':
        image = self.numpy_to_pil(image)
    if not return_dict:
        return image, latent_timestep.item()
    return ImagePipelineOutput(images=image)

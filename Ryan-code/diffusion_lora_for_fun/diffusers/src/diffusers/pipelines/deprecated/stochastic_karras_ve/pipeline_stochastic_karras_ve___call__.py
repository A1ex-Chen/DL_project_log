@torch.no_grad()
def __call__(self, batch_size: int=1, num_inference_steps: int=50,
    generator: Optional[Union[torch.Generator, List[torch.Generator]]]=None,
    output_type: Optional[str]='pil', return_dict: bool=True, **kwargs
    ) ->Union[Tuple, ImagePipelineOutput]:
    """
        The call function to the pipeline for generation.

        Args:
            batch_size (`int`, *optional*, defaults to 1):
                The number of images to generate.
            generator (`torch.Generator`, *optional*):
                A [`torch.Generator`](https://pytorch.org/docs/stable/generated/torch.Generator.html) to make
                generation deterministic.
            num_inference_steps (`int`, *optional*, defaults to 50):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generated image. Choose between `PIL.Image` or `np.array`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`ImagePipelineOutput`] instead of a plain tuple.

        Example:

        Returns:
            [`~pipelines.ImagePipelineOutput`] or `tuple`:
                If `return_dict` is `True`, [`~pipelines.ImagePipelineOutput`] is returned, otherwise a `tuple` is
                returned where the first element is a list with the generated images.
        """
    img_size = self.unet.config.sample_size
    shape = batch_size, 3, img_size, img_size
    model = self.unet
    sample = randn_tensor(shape, generator=generator, device=self.device
        ) * self.scheduler.init_noise_sigma
    self.scheduler.set_timesteps(num_inference_steps)
    for t in self.progress_bar(self.scheduler.timesteps):
        sigma = self.scheduler.schedule[t]
        sigma_prev = self.scheduler.schedule[t - 1] if t > 0 else 0
        sample_hat, sigma_hat = self.scheduler.add_noise_to_input(sample,
            sigma, generator=generator)
        model_output = sigma_hat / 2 * model((sample_hat + 1) / 2, 
            sigma_hat / 2).sample
        step_output = self.scheduler.step(model_output, sigma_hat,
            sigma_prev, sample_hat)
        if sigma_prev != 0:
            model_output = sigma_prev / 2 * model((step_output.prev_sample +
                1) / 2, sigma_prev / 2).sample
            step_output = self.scheduler.step_correct(model_output,
                sigma_hat, sigma_prev, sample_hat, step_output.prev_sample,
                step_output['derivative'])
        sample = step_output.prev_sample
    sample = (sample / 2 + 0.5).clamp(0, 1)
    image = sample.cpu().permute(0, 2, 3, 1).numpy()
    if output_type == 'pil':
        image = self.numpy_to_pil(image)
    if not return_dict:
        return image,
    return ImagePipelineOutput(images=image)

@torch.no_grad()
def __call__(self, batch_size: int=1, num_inference_steps: int=2000,
    generator: Optional[Union[torch.Generator, List[torch.Generator]]]=None,
    output_type: Optional[str]='pil', return_dict: bool=True, **kwargs
    ) ->Union[ImagePipelineOutput, Tuple]:
    """
        The call function to the pipeline for generation.

        Args:
            batch_size (`int`, *optional*, defaults to 1):
                The number of images to generate.
            generator (`torch.Generator`, `optional`):
                A [`torch.Generator`](https://pytorch.org/docs/stable/generated/torch.Generator.html) to make
                generation deterministic.
            output_type (`str`, `optional`, defaults to `"pil"`):
                The output format of the generated image. Choose between `PIL.Image` or `np.array`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`ImagePipelineOutput`] instead of a plain tuple.

        Returns:
            [`~pipelines.ImagePipelineOutput`] or `tuple`:
                If `return_dict` is `True`, [`~pipelines.ImagePipelineOutput`] is returned, otherwise a `tuple` is
                returned where the first element is a list with the generated images.
        """
    img_size = self.unet.config.sample_size
    shape = batch_size, 3, img_size, img_size
    model = self.unet
    sample = randn_tensor(shape, generator=generator
        ) * self.scheduler.init_noise_sigma
    sample = sample.to(self.device)
    self.scheduler.set_timesteps(num_inference_steps)
    self.scheduler.set_sigmas(num_inference_steps)
    for i, t in enumerate(self.progress_bar(self.scheduler.timesteps)):
        sigma_t = self.scheduler.sigmas[i] * torch.ones(shape[0], device=
            self.device)
        for _ in range(self.scheduler.config.correct_steps):
            model_output = self.unet(sample, sigma_t).sample
            sample = self.scheduler.step_correct(model_output, sample,
                generator=generator).prev_sample
        model_output = model(sample, sigma_t).sample
        output = self.scheduler.step_pred(model_output, t, sample,
            generator=generator)
        sample, sample_mean = output.prev_sample, output.prev_sample_mean
    sample = sample_mean.clamp(0, 1)
    sample = sample.cpu().permute(0, 2, 3, 1).numpy()
    if output_type == 'pil':
        sample = self.numpy_to_pil(sample)
    if not return_dict:
        return sample,
    return ImagePipelineOutput(images=sample)

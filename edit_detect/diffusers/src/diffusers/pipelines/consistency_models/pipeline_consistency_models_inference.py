@replace_example_docstring(EXAMPLE_DOC_STRING)
def inference(self, batch_size: int=1, class_labels: Optional[Union[torch.
    Tensor, List[int], int]]=None, num_inference_steps: int=1, timesteps:
    List[int]=None, generator: Optional[Union[torch.Generator, List[torch.
    Generator]]]=None, latents: Optional[torch.Tensor]=None, output_type:
    Optional[str]='pil', return_dict: bool=True, callback: Optional[
    Callable[[int, int, torch.Tensor], None]]=None, callback_steps: int=1,
    is_pbar: bool=True):
    """
        Args:
            batch_size (`int`, *optional*, defaults to 1):
                The number of images to generate.
            class_labels (`torch.Tensor` or `List[int]` or `int`, *optional*):
                Optional class labels for conditioning class-conditional consistency models. Not used if the model is
                not class-conditional.
            num_inference_steps (`int`, *optional*, defaults to 1):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            timesteps (`List[int]`, *optional*):
                Custom timesteps to use for the denoising process. If not defined, equal spaced `num_inference_steps`
                timesteps are used. Must be in descending order.
            generator (`torch.Generator`, *optional*):
                A [`torch.Generator`](https://pytorch.org/docs/stable/generated/torch.Generator.html) to make
                generation deterministic.
            latents (`torch.Tensor`, *optional*):
                Pre-generated noisy latents sampled from a Gaussian distribution, to be used as inputs for image
                generation. Can be used to tweak the same generation with different prompts. If not provided, a latents
                tensor is generated by sampling using the supplied random `generator`.
            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generated image. Choose between `PIL.Image` or `np.array`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~pipelines.ImagePipelineOutput`] instead of a plain tuple.
            callback (`Callable`, *optional*):
                A function that calls every `callback_steps` steps during inference. The function is called with the
                following arguments: `callback(step: int, timestep: int, latents: torch.Tensor)`.
            callback_steps (`int`, *optional*, defaults to 1):
                The frequency at which the `callback` function is called. If not specified, the callback is called at
                every step.

        Examples:

        Returns:
            [`~pipelines.ImagePipelineOutput`] or `tuple`:
                If `return_dict` is `True`, [`~pipelines.ImagePipelineOutput`] is returned, otherwise a `tuple` is
                returned where the first element is a list with the generated images.
        """
    img_size = self.unet.config.sample_size
    device = self._execution_device
    self.check_inputs(num_inference_steps, timesteps, latents, batch_size,
        img_size, callback_steps)
    sample = self.prepare_latents(batch_size=batch_size, num_channels=self.
        unet.config.in_channels, height=img_size, width=img_size, dtype=
        self.unet.dtype, device=device, generator=generator, latents=latents)
    class_labels = self.prepare_class_labels(batch_size, device,
        class_labels=class_labels)
    if timesteps is not None:
        self.scheduler.set_timesteps(timesteps=timesteps, device=device)
        timesteps = self.scheduler.timesteps
        num_inference_steps = len(timesteps)
    else:
        self.scheduler.set_timesteps(num_inference_steps)
        timesteps = self.scheduler.timesteps
    if is_pbar:
        progress_bar = self.progress_bar(total=num_inference_steps)
    for i, t in enumerate(timesteps):
        scaled_sample = self.scheduler.scale_model_input(sample, t)
        model_output = self.unet(scaled_sample, t, class_labels=
            class_labels, return_dict=False)[0]
        sample = self.scheduler.step(model_output, t, sample, generator=
            generator)[0]
        if is_pbar:
            progress_bar.update()
        if callback is not None and i % callback_steps == 0:
            callback(i, t, sample)
    image = self.postprocess_image(sample, output_type=output_type)
    self.maybe_free_model_hooks()
    if not return_dict:
        return image,
    return ImagePipelineOutput(images=image)

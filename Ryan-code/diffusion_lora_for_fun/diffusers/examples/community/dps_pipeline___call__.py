@torch.no_grad()
def __call__(self, measurement: torch.Tensor, operator: torch.nn.Module,
    loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    batch_size: int=1, generator: Optional[Union[torch.Generator, List[
    torch.Generator]]]=None, num_inference_steps: int=1000, output_type:
    Optional[str]='pil', return_dict: bool=True, zeta: float=0.3) ->Union[
    ImagePipelineOutput, Tuple]:
    """
        The call function to the pipeline for generation.

        Args:
            measurement (`torch.Tensor`, *required*):
                A 'torch.Tensor', the corrupted image
            operator (`torch.nn.Module`, *required*):
                A 'torch.nn.Module', the operator generating the corrupted image
            loss_fn (`Callable[[torch.Tensor, torch.Tensor], torch.Tensor]`, *required*):
                A 'Callable[[torch.Tensor, torch.Tensor], torch.Tensor]', the loss function used
                between the measurements, for most of the cases using RMSE is fine.
            batch_size (`int`, *optional*, defaults to 1):
                The number of images to generate.
            generator (`torch.Generator`, *optional*):
                A [`torch.Generator`](https://pytorch.org/docs/stable/generated/torch.Generator.html) to make
                generation deterministic.
            num_inference_steps (`int`, *optional*, defaults to 1000):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generated image. Choose between `PIL.Image` or `np.array`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~pipelines.ImagePipelineOutput`] instead of a plain tuple.

        Example:

        ```py
        >>> from diffusers import DDPMPipeline

        >>> # load model and scheduler
        >>> pipe = DDPMPipeline.from_pretrained("google/ddpm-cat-256")

        >>> # run pipeline in inference (sample random noise and denoise)
        >>> image = pipe().images[0]

        >>> # save image
        >>> image.save("ddpm_generated_image.png")
        ```

        Returns:
            [`~pipelines.ImagePipelineOutput`] or `tuple`:
                If `return_dict` is `True`, [`~pipelines.ImagePipelineOutput`] is returned, otherwise a `tuple` is
                returned where the first element is a list with the generated images
        """
    if isinstance(self.unet.config.sample_size, int):
        image_shape = (batch_size, self.unet.config.in_channels, self.unet.
            config.sample_size, self.unet.config.sample_size)
    else:
        image_shape = (batch_size, self.unet.config.in_channels, *self.unet
            .config.sample_size)
    if self.device.type == 'mps':
        image = randn_tensor(image_shape, generator=generator)
        image = image.to(self.device)
    else:
        image = randn_tensor(image_shape, generator=generator, device=self.
            device)
    self.scheduler.set_timesteps(num_inference_steps)
    for t in self.progress_bar(self.scheduler.timesteps):
        with torch.enable_grad():
            image = image.requires_grad_()
            model_output = self.unet(image, t).sample
            scheduler_out = self.scheduler.step(model_output, t, image,
                generator=generator)
            image_pred, origi_pred = (scheduler_out.prev_sample,
                scheduler_out.pred_original_sample)
            measurement_pred = operator(origi_pred)
            loss = loss_fn(measurement, measurement_pred)
            loss.backward()
            print('distance: {0:.4f}'.format(loss.item()))
            with torch.no_grad():
                image_pred = image_pred - zeta * image.grad
                image = image_pred.detach()
    image = (image / 2 + 0.5).clamp(0, 1)
    image = image.cpu().permute(0, 2, 3, 1).numpy()
    if output_type == 'pil':
        image = self.numpy_to_pil(image)
    if not return_dict:
        return image,
    return ImagePipelineOutput(images=image)

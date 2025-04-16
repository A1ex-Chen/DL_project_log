@torch.no_grad()
def __call__(self, batch_size: int=1, generator: Optional[Union[torch.
    Generator, List[torch.Generator]]]=None, num_inference_steps: int=1000,
    start_ratio_inference_steps: float=0.0, output_type: Optional[str]=
    'pil', return_dict: bool=True, init: torch.Tensor=None) ->Union[
    ImagePipelineOutput, Tuple]:
    """
        The call function to the pipeline for generation.

        Args:
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
    if init is None:
        if self.device.type == 'mps':
            image = randn_tensor(image_shape, generator=generator)
            image = image.to(self.device)
        else:
            image = randn_tensor(image_shape, generator=generator, device=
                self.device)
    else:
        image = init.to(self.device)
    print(
        f'Image shape: {image.shape}, (min, max): ({torch.min(image)}, {torch.max(image)})'
        )
    self.scheduler.set_timesteps(num_inference_steps)
    pred_orig_images: list = []
    for t in self.progress_bar(self.scheduler.timesteps[-int(
        start_ratio_inference_steps * len(self.scheduler.timesteps)):]):
        model_output = self.unet(image, t).sample
        output = self.scheduler.step(model_output, t, image, generator=
            generator)
        image = output.prev_sample
        pred_orig_images.append(output.pred_original_sample.detach().cpu())
    pred_orig_images = torch.stack(pred_orig_images)
    latents = image.detach().cpu()
    image = (image / 2 + 0.5).clamp(0, 1)
    image = image.cpu().permute(0, 2, 3, 1).numpy()
    if output_type == 'pil':
        image = self.numpy_to_pil(image)
    if not return_dict:
        return image,
    return ImagePipelineOutputExt(images=image, latents=latents,
        pred_orig_samples=pred_orig_images)

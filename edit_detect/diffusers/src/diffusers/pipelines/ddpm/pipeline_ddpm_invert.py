@torch.no_grad()
def invert(self, init: Union[torch.Tensor, np.ndarray, PIL.Image.Image],
    batch_size: int=1, generator: Optional[Union[torch.Generator, List[
    torch.Generator]]]=None, num_inference_steps: int=1000,
    start_ratio_inference_steps: float=0.0, output_type: Optional[str]=
    'pil', return_dict: bool=True) ->Union[ImagePipelineOutput, Tuple]:
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
        >>> image = pipe.invert().images[0]

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
    if isinstance(init, np.ndarray):
        image = torch.from_numpy(init)
        if len(init.shape) < 4:
            image = torch.unsqueeze(image, 0)
    elif isinstance(init, torch.Tensor):
        image = init
        if len(init.shape) < 4:
            image = torch.unsqueeze(image, 0)
    elif isinstance(init, PIL.Image.Image):
        image = transforms.ToTensor()(init).unsqueeze(0)
    elif isinstance(init, list):
        if isinstance(init[0], np.ndarray):
            init = np.stack(init, axis=0)
            image = torch.from_numpy(init)
        elif isinstance(init[0], torch.Tensor):
            image = torch.stack(init, axis=0)
        elif isinstance(init[0], PIL.Image.Image):
            image = torch.stack([transforms.ToTensor()(e) for e in init],
                axis=0).permute(0, 3, 1, 2)
        else:
            raise TypeError(
                f'The elements of the arguement init should be numpy.ndarray, torch.Tensor, or PIL.Image.Image, not {type(init[0])}.'
                )
    else:
        raise TypeError(
            f'Arguement init should be numpy.ndarray, torch.Tensor, or PIL.Image.Image, not {type(init)}.'
            )
    print(
        f'Origianl image: {type(image)}, shape: {image.shape}, (min, max): ({torch.min(image)}, {torch.max(image)})'
        )
    image = (image.to(self.device) - 0.5) * 2
    print(
        f'image: {type(image)}, shape: {image.shape}, (min, max): ({torch.min(image)}, {torch.max(image)})'
        )
    self.inverse_scheduler.set_timesteps(num_inference_steps)
    pred_orig_images = []
    for t in self.progress_bar(self.inverse_scheduler.timesteps[-int(
        start_ratio_inference_steps * len(self.scheduler.timesteps)):]):
        model_output = self.unet(image, t).sample
        step_output: ImagePipelineOutputExt = self.inverse_scheduler.step(
            model_output, t, image)
        image = step_output.prev_sample
        pred_orig_images.append(step_output.pred_orig_samples.detach().cpu())
    pred_orig_images = torch.stack(pred_orig_images)
    latents = image.detach().cpu()
    print(
        f'Latents shape: {latents.shape}, (min, max): ({torch.min(latents)}, {torch.max(latents)})'
        )
    image = (image / 2 + 0.5).clamp(0, 1)
    image = image.detach().cpu().permute(0, 2, 3, 1).numpy()
    if output_type == 'pil':
        image = self.numpy_to_pil(image)
    print(f'pred_orig_images: {pred_orig_images.shape}')
    return ImagePipelineOutputExt(images=image, latents=latents,
        pred_orig_samples=pred_orig_images)

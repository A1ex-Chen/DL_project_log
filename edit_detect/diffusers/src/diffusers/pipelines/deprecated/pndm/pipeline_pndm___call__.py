@torch.no_grad()
def __call__(self, batch_size: int=1, num_inference_steps: int=50,
    generator: Optional[Union[torch.Generator, List[torch.Generator]]]=None,
    output_type: Optional[str]='pil', return_dict: bool=True, **kwargs
    ) ->Union[ImagePipelineOutput, Tuple]:
    """
        The call function to the pipeline for generation.

        Args:
            batch_size (`int`, `optional`, defaults to 1):
                The number of images to generate.
            num_inference_steps (`int`, `optional`, defaults to 50):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            generator (`torch.Generator`, `optional`):
                A [`torch.Generator`](https://pytorch.org/docs/stable/generated/torch.Generator.html) to make
                generation deterministic.
            output_type (`str`, `optional`, defaults to `"pil"`):
                The output format of the generated image. Choose between `PIL.Image` or `np.array`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`ImagePipelineOutput`] instead of a plain tuple.

        Example:

        ```py
        >>> from diffusers import PNDMPipeline

        >>> # load model and scheduler
        >>> pndm = PNDMPipeline.from_pretrained("google/ddpm-cifar10-32")

        >>> # run pipeline in inference (sample random noise and denoise)
        >>> image = pndm().images[0]

        >>> # save image
        >>> image.save("pndm_generated_image.png")
        ```

        Returns:
            [`~pipelines.ImagePipelineOutput`] or `tuple`:
                If `return_dict` is `True`, [`~pipelines.ImagePipelineOutput`] is returned, otherwise a `tuple` is
                returned where the first element is a list with the generated images.
        """
    image = randn_tensor((batch_size, self.unet.config.in_channels, self.
        unet.config.sample_size, self.unet.config.sample_size), generator=
        generator, device=self.device)
    self.scheduler.set_timesteps(num_inference_steps)
    for t in self.progress_bar(self.scheduler.timesteps):
        model_output = self.unet(image, t).sample
        image = self.scheduler.step(model_output, t, image).prev_sample
    image = (image / 2 + 0.5).clamp(0, 1)
    image = image.cpu().permute(0, 2, 3, 1).numpy()
    if output_type == 'pil':
        image = self.numpy_to_pil(image)
    if not return_dict:
        return image,
    return ImagePipelineOutput(images=image)

@torch.no_grad()
def __call__(self, image: Union[torch.Tensor, PIL.Image.Image], mask_image:
    Union[torch.Tensor, PIL.Image.Image], num_inference_steps: int=250, eta:
    float=0.0, jump_length: int=10, jump_n_sample: int=10, generator:
    Optional[Union[torch.Generator, List[torch.Generator]]]=None,
    output_type: Optional[str]='pil', return_dict: bool=True) ->Union[
    ImagePipelineOutput, Tuple]:
    """
        The call function to the pipeline for generation.

        Args:
            image (`torch.Tensor` or `PIL.Image.Image`):
                The original image to inpaint on.
            mask_image (`torch.Tensor` or `PIL.Image.Image`):
                The mask_image where 0.0 define which part of the original image to inpaint.
            num_inference_steps (`int`, *optional*, defaults to 1000):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            eta (`float`):
                The weight of the added noise in a diffusion step. Its value is between 0.0 and 1.0; 0.0 corresponds to
                DDIM and 1.0 is the DDPM scheduler.
            jump_length (`int`, *optional*, defaults to 10):
                The number of steps taken forward in time before going backward in time for a single jump ("j" in
                RePaint paper). Take a look at Figure 9 and 10 in the [paper](https://arxiv.org/pdf/2201.09865.pdf).
            jump_n_sample (`int`, *optional*, defaults to 10):
                The number of times to make a forward time jump for a given chosen time sample. Take a look at Figure 9
                and 10 in the [paper](https://arxiv.org/pdf/2201.09865.pdf).
            generator (`torch.Generator`, *optional*):
                A [`torch.Generator`](https://pytorch.org/docs/stable/generated/torch.Generator.html) to make
                generation deterministic.
            output_type (`str`, `optional`, defaults to `"pil"`):
                The output format of the generated image. Choose between `PIL.Image` or `np.array`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`ImagePipelineOutput`] instead of a plain tuple.

        Example:

        ```py
        >>> from io import BytesIO
        >>> import torch
        >>> import PIL
        >>> import requests
        >>> from diffusers import RePaintPipeline, RePaintScheduler


        >>> def download_image(url):
        ...     response = requests.get(url)
        ...     return PIL.Image.open(BytesIO(response.content)).convert("RGB")


        >>> img_url = "https://huggingface.co/datasets/hf-internal-testing/diffusers-images/resolve/main/repaint/celeba_hq_256.png"
        >>> mask_url = "https://huggingface.co/datasets/hf-internal-testing/diffusers-images/resolve/main/repaint/mask_256.png"

        >>> # Load the original image and the mask as PIL images
        >>> original_image = download_image(img_url).resize((256, 256))
        >>> mask_image = download_image(mask_url).resize((256, 256))

        >>> # Load the RePaint scheduler and pipeline based on a pretrained DDPM model
        >>> scheduler = RePaintScheduler.from_pretrained("google/ddpm-ema-celebahq-256")
        >>> pipe = RePaintPipeline.from_pretrained("google/ddpm-ema-celebahq-256", scheduler=scheduler)
        >>> pipe = pipe.to("cuda")

        >>> generator = torch.Generator(device="cuda").manual_seed(0)
        >>> output = pipe(
        ...     image=original_image,
        ...     mask_image=mask_image,
        ...     num_inference_steps=250,
        ...     eta=0.0,
        ...     jump_length=10,
        ...     jump_n_sample=10,
        ...     generator=generator,
        ... )
        >>> inpainted_image = output.images[0]
        ```

        Returns:
            [`~pipelines.ImagePipelineOutput`] or `tuple`:
                If `return_dict` is `True`, [`~pipelines.ImagePipelineOutput`] is returned, otherwise a `tuple` is
                returned where the first element is a list with the generated images.
        """
    original_image = image
    original_image = _preprocess_image(original_image)
    original_image = original_image.to(device=self._execution_device, dtype
        =self.unet.dtype)
    mask_image = _preprocess_mask(mask_image)
    mask_image = mask_image.to(device=self._execution_device, dtype=self.
        unet.dtype)
    batch_size = original_image.shape[0]
    if isinstance(generator, list) and len(generator) != batch_size:
        raise ValueError(
            f'You have passed a list of generators of length {len(generator)}, but requested an effective batch size of {batch_size}. Make sure the batch size matches the length of the generators.'
            )
    image_shape = original_image.shape
    image = randn_tensor(image_shape, generator=generator, device=self.
        _execution_device, dtype=self.unet.dtype)
    self.scheduler.set_timesteps(num_inference_steps, jump_length,
        jump_n_sample, self._execution_device)
    self.scheduler.eta = eta
    t_last = self.scheduler.timesteps[0] + 1
    generator = generator[0] if isinstance(generator, list) else generator
    for i, t in enumerate(self.progress_bar(self.scheduler.timesteps)):
        if t < t_last:
            model_output = self.unet(image, t).sample
            image = self.scheduler.step(model_output, t, image,
                original_image, mask_image, generator).prev_sample
        else:
            image = self.scheduler.undo_step(image, t_last, generator)
        t_last = t
    image = (image / 2 + 0.5).clamp(0, 1)
    image = image.cpu().permute(0, 2, 3, 1).numpy()
    if output_type == 'pil':
        image = self.numpy_to_pil(image)
    if not return_dict:
        return image,
    return ImagePipelineOutput(images=image)

@torch.no_grad()
def __call__(self, prompt: Union[str, List[str]]=None, image: Union[torch.
    Tensor, PIL.Image.Image]=None, strength: float=0.8, num_inference_steps:
    int=50, guidance_scale: float=7.5, negative_prompt: Optional[Union[str,
    List[str]]]=None, generator: Optional[Union[torch.Generator, List[torch
    .Generator]]]=None):
    """
        Function invoked when calling the pipeline for generation.

        Args:
            prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts to guide the image generation. If not defined, one has to pass `prompt_embeds`.
                instead.
            image (`PIL.Image.Image`):
                `Image`, or tensor representing an image batch which will be inpainted, *i.e.* parts of the image will
                be masked out with `mask_image` and repainted according to `prompt`.
            strength (`float`, *optional*, defaults to 0.8):
                Conceptually, indicates how much to transform the reference `image`. Must be between 0 and 1. `image`
                will be used as a starting point, adding more noise to it the larger the `strength`. The number of
                denoising steps depends on the amount of noise initially added. When `strength` is 1, added noise will
                be maximum and the denoising process will run for the full number of iterations specified in
                `num_inference_steps`. A value of 1, therefore, essentially ignores `image`.
            num_inference_steps (`int`, *optional*, defaults to 50):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            guidance_scale (`float`, *optional*, defaults to 7.5):
                Guidance scale as defined in [Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598).
                `guidance_scale` is defined as `w` of equation 2. of [Imagen
                Paper](https://arxiv.org/pdf/2205.11487.pdf). Guidance scale is enabled by setting `guidance_scale >
                1`. Higher guidance scale encourages to generate images that are closely linked to the text `prompt`,
                usually at the expense of lower image quality.
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the image generation. If not defined, one has to pass
                `negative_prompt_embeds`. instead. If not defined, one has to pass `negative_prompt_embeds`. instead.
                Ignored when not using guidance (i.e., ignored if `guidance_scale` is less than `1`).
            generator (`torch.Generator` or `List[torch.Generator]`, *optional*):
                One or a list of [torch generator(s)](https://pytorch.org/docs/stable/generated/torch.Generator.html)
                to make generation deterministic.

        """
    self.generator = generator
    self.denoising_steps = num_inference_steps
    self._guidance_scale = guidance_scale
    self.scheduler.set_timesteps(self.denoising_steps, device=self.torch_device
        )
    if prompt is not None and isinstance(prompt, str):
        batch_size = 1
        prompt = [prompt]
    elif prompt is not None and isinstance(prompt, list):
        batch_size = len(prompt)
    else:
        raise ValueError(
            f'Expected prompt to be of type list or str but got {type(prompt)}'
            )
    if negative_prompt is None:
        negative_prompt = [''] * batch_size
    if negative_prompt is not None and isinstance(negative_prompt, str):
        negative_prompt = [negative_prompt]
    assert len(prompt) == len(negative_prompt)
    if batch_size > self.max_batch_size:
        raise ValueError(
            f'Batch size {len(prompt)} is larger than allowed {self.max_batch_size}. If dynamic shape is used, then maximum batch size is 4'
            )
    self.__loadResources(self.image_height, self.image_width, batch_size)
    with torch.inference_mode(), torch.autocast('cuda'), trt.Runtime(TRT_LOGGER
        ):
        timesteps, t_start = self.__initialize_timesteps(self.
            denoising_steps, strength)
        latent_timestep = timesteps[:1].repeat(batch_size)
        if isinstance(image, PIL.Image.Image):
            image = preprocess_image(image)
        init_image = self.__preprocess_images(batch_size, (image,))[0]
        init_latents = self.__encode_image(init_image)
        noise = torch.randn(init_latents.shape, generator=self.generator,
            device=self.torch_device, dtype=torch.float32)
        latents = self.scheduler.add_noise(init_latents, noise, latent_timestep
            )
        text_embeddings = self.__encode_prompt(prompt, negative_prompt)
        latents = self.__denoise_latent(latents, text_embeddings, timesteps
            =timesteps, step_offset=t_start)
        images = self.__decode_latent(latents)
    images = self.numpy_to_pil(images)
    return StableDiffusionPipelineOutput(images=images,
        nsfw_content_detected=None)

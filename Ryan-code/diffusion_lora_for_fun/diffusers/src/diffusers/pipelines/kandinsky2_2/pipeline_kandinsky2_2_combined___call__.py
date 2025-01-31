@torch.no_grad()
@replace_example_docstring(INPAINT_EXAMPLE_DOC_STRING)
def __call__(self, prompt: Union[str, List[str]], image: Union[torch.Tensor,
    PIL.Image.Image, List[torch.Tensor], List[PIL.Image.Image]], mask_image:
    Union[torch.Tensor, PIL.Image.Image, List[torch.Tensor], List[PIL.Image
    .Image]], negative_prompt: Optional[Union[str, List[str]]]=None,
    num_inference_steps: int=100, guidance_scale: float=4.0,
    num_images_per_prompt: int=1, height: int=512, width: int=512,
    prior_guidance_scale: float=4.0, prior_num_inference_steps: int=25,
    generator: Optional[Union[torch.Generator, List[torch.Generator]]]=None,
    latents: Optional[torch.Tensor]=None, output_type: Optional[str]='pil',
    return_dict: bool=True, prior_callback_on_step_end: Optional[Callable[[
    int, int, Dict], None]]=None, prior_callback_on_step_end_tensor_inputs:
    List[str]=['latents'], callback_on_step_end: Optional[Callable[[int,
    int, Dict], None]]=None, callback_on_step_end_tensor_inputs: List[str]=
    ['latents'], **kwargs):
    """
        Function invoked when calling the pipeline for generation.

        Args:
            prompt (`str` or `List[str]`):
                The prompt or prompts to guide the image generation.
            image (`torch.Tensor`, `PIL.Image.Image`, `np.ndarray`, `List[torch.Tensor]`, `List[PIL.Image.Image]`, or `List[np.ndarray]`):
                `Image`, or tensor representing an image batch, that will be used as the starting point for the
                process. Can also accept image latents as `image`, if passing latents directly, it will not be encoded
                again.
            mask_image (`np.array`):
                Tensor representing an image batch, to mask `image`. White pixels in the mask will be repainted, while
                black pixels will be preserved. If `mask_image` is a PIL image, it will be converted to a single
                channel (luminance) before use. If it's a tensor, it should contain one color channel (L) instead of 3,
                so the expected shape would be `(B, H, W, 1)`.
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the image generation. Ignored when not using guidance (i.e., ignored
                if `guidance_scale` is less than `1`).
            num_images_per_prompt (`int`, *optional*, defaults to 1):
                The number of images to generate per prompt.
            guidance_scale (`float`, *optional*, defaults to 4.0):
                Guidance scale as defined in [Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598).
                `guidance_scale` is defined as `w` of equation 2. of [Imagen
                Paper](https://arxiv.org/pdf/2205.11487.pdf). Guidance scale is enabled by setting `guidance_scale >
                1`. Higher guidance scale encourages to generate images that are closely linked to the text `prompt`,
                usually at the expense of lower image quality.
            num_inference_steps (`int`, *optional*, defaults to 100):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            height (`int`, *optional*, defaults to 512):
                The height in pixels of the generated image.
            width (`int`, *optional*, defaults to 512):
                The width in pixels of the generated image.
            prior_guidance_scale (`float`, *optional*, defaults to 4.0):
                Guidance scale as defined in [Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598).
                `guidance_scale` is defined as `w` of equation 2. of [Imagen
                Paper](https://arxiv.org/pdf/2205.11487.pdf). Guidance scale is enabled by setting `guidance_scale >
                1`. Higher guidance scale encourages to generate images that are closely linked to the text `prompt`,
                usually at the expense of lower image quality.
            prior_num_inference_steps (`int`, *optional*, defaults to 100):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            generator (`torch.Generator` or `List[torch.Generator]`, *optional*):
                One or a list of [torch generator(s)](https://pytorch.org/docs/stable/generated/torch.Generator.html)
                to make generation deterministic.
            latents (`torch.Tensor`, *optional*):
                Pre-generated noisy latents, sampled from a Gaussian distribution, to be used as inputs for image
                generation. Can be used to tweak the same generation with different prompts. If not provided, a latents
                tensor will ge generated by sampling using the supplied random `generator`.
            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generate image. Choose between: `"pil"` (`PIL.Image.Image`), `"np"`
                (`np.array`) or `"pt"` (`torch.Tensor`).
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~pipelines.ImagePipelineOutput`] instead of a plain tuple.
            prior_callback_on_step_end (`Callable`, *optional*):
                A function that calls at the end of each denoising steps during the inference. The function is called
                with the following arguments: `prior_callback_on_step_end(self: DiffusionPipeline, step: int, timestep:
                int, callback_kwargs: Dict)`.
            prior_callback_on_step_end_tensor_inputs (`List`, *optional*):
                The list of tensor inputs for the `prior_callback_on_step_end` function. The tensors specified in the
                list will be passed as `callback_kwargs` argument. You will only be able to include variables listed in
                the `._callback_tensor_inputs` attribute of your pipeline class.
            callback_on_step_end (`Callable`, *optional*):
                A function that calls at the end of each denoising steps during the inference. The function is called
                with the following arguments: `callback_on_step_end(self: DiffusionPipeline, step: int, timestep: int,
                callback_kwargs: Dict)`. `callback_kwargs` will include a list of all tensors as specified by
                `callback_on_step_end_tensor_inputs`.
            callback_on_step_end_tensor_inputs (`List`, *optional*):
                The list of tensor inputs for the `callback_on_step_end` function. The tensors specified in the list
                will be passed as `callback_kwargs` argument. You will only be able to include variables listed in the
                `._callback_tensor_inputs` attribute of your pipeline class.


        Examples:

        Returns:
            [`~pipelines.ImagePipelineOutput`] or `tuple`
        """
    prior_kwargs = {}
    if kwargs.get('prior_callback', None) is not None:
        prior_kwargs['callback'] = kwargs.pop('prior_callback')
        deprecate('prior_callback', '1.0.0',
            'Passing `prior_callback` as an input argument to `__call__` is deprecated, consider use `prior_callback_on_step_end`'
            )
    if kwargs.get('prior_callback_steps', None) is not None:
        deprecate('prior_callback_steps', '1.0.0',
            'Passing `prior_callback_steps` as an input argument to `__call__` is deprecated, consider use `prior_callback_on_step_end`'
            )
        prior_kwargs['callback_steps'] = kwargs.pop('prior_callback_steps')
    prior_outputs = self.prior_pipe(prompt=prompt, negative_prompt=
        negative_prompt, num_images_per_prompt=num_images_per_prompt,
        num_inference_steps=prior_num_inference_steps, generator=generator,
        latents=latents, guidance_scale=prior_guidance_scale, output_type=
        'pt', return_dict=False, callback_on_step_end=
        prior_callback_on_step_end, callback_on_step_end_tensor_inputs=
        prior_callback_on_step_end_tensor_inputs, **prior_kwargs)
    image_embeds = prior_outputs[0]
    negative_image_embeds = prior_outputs[1]
    prompt = [prompt] if not isinstance(prompt, (list, tuple)) else prompt
    image = [image] if isinstance(prompt, PIL.Image.Image) else image
    mask_image = [mask_image] if isinstance(mask_image, PIL.Image.Image
        ) else mask_image
    if len(prompt) < image_embeds.shape[0] and image_embeds.shape[0] % len(
        prompt) == 0:
        prompt = image_embeds.shape[0] // len(prompt) * prompt
    if isinstance(image, (list, tuple)) and len(image) < image_embeds.shape[0
        ] and image_embeds.shape[0] % len(image) == 0:
        image = image_embeds.shape[0] // len(image) * image
    if isinstance(mask_image, (list, tuple)) and len(mask_image
        ) < image_embeds.shape[0] and image_embeds.shape[0] % len(mask_image
        ) == 0:
        mask_image = image_embeds.shape[0] // len(mask_image) * mask_image
    outputs = self.decoder_pipe(image=image, mask_image=mask_image,
        image_embeds=image_embeds, negative_image_embeds=
        negative_image_embeds, width=width, height=height,
        num_inference_steps=num_inference_steps, generator=generator,
        guidance_scale=guidance_scale, output_type=output_type, return_dict
        =return_dict, callback_on_step_end=callback_on_step_end,
        callback_on_step_end_tensor_inputs=
        callback_on_step_end_tensor_inputs, **kwargs)
    self.maybe_free_model_hooks()
    return outputs

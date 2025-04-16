@torch.no_grad()
@replace_example_docstring(TEXT2IMAGE_EXAMPLE_DOC_STRING)
def __call__(self, prompt: Optional[Union[str, List[str]]]=None, height:
    int=512, width: int=512, prior_num_inference_steps: int=60,
    prior_timesteps: Optional[List[float]]=None, prior_guidance_scale:
    float=4.0, num_inference_steps: int=12, decoder_timesteps: Optional[
    List[float]]=None, decoder_guidance_scale: float=0.0, negative_prompt:
    Optional[Union[str, List[str]]]=None, prompt_embeds: Optional[torch.
    Tensor]=None, negative_prompt_embeds: Optional[torch.Tensor]=None,
    num_images_per_prompt: int=1, generator: Optional[Union[torch.Generator,
    List[torch.Generator]]]=None, latents: Optional[torch.Tensor]=None,
    output_type: Optional[str]='pil', return_dict: bool=True,
    prior_callback_on_step_end: Optional[Callable[[int, int, Dict], None]]=
    None, prior_callback_on_step_end_tensor_inputs: List[str]=['latents'],
    callback_on_step_end: Optional[Callable[[int, int, Dict], None]]=None,
    callback_on_step_end_tensor_inputs: List[str]=['latents'], **kwargs):
    """
        Function invoked when calling the pipeline for generation.

        Args:
            prompt (`str` or `List[str]`):
                The prompt or prompts to guide the image generation for the prior and decoder.
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the image generation. Ignored when not using guidance (i.e., ignored
                if `guidance_scale` is less than `1`).
            prompt_embeds (`torch.Tensor`, *optional*):
                Pre-generated text embeddings for the prior. Can be used to easily tweak text inputs, *e.g.* prompt
                weighting. If not provided, text embeddings will be generated from `prompt` input argument.
            negative_prompt_embeds (`torch.Tensor`, *optional*):
                Pre-generated negative text embeddings for the prior. Can be used to easily tweak text inputs, *e.g.*
                prompt weighting. If not provided, negative_prompt_embeds will be generated from `negative_prompt`
                input argument.
            num_images_per_prompt (`int`, *optional*, defaults to 1):
                The number of images to generate per prompt.
            height (`int`, *optional*, defaults to 512):
                The height in pixels of the generated image.
            width (`int`, *optional*, defaults to 512):
                The width in pixels of the generated image.
            prior_guidance_scale (`float`, *optional*, defaults to 4.0):
                Guidance scale as defined in [Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598).
                `prior_guidance_scale` is defined as `w` of equation 2. of [Imagen
                Paper](https://arxiv.org/pdf/2205.11487.pdf). Guidance scale is enabled by setting
                `prior_guidance_scale > 1`. Higher guidance scale encourages to generate images that are closely linked
                to the text `prompt`, usually at the expense of lower image quality.
            prior_num_inference_steps (`Union[int, Dict[float, int]]`, *optional*, defaults to 60):
                The number of prior denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference. For more specific timestep spacing, you can pass customized
                `prior_timesteps`
            num_inference_steps (`int`, *optional*, defaults to 12):
                The number of decoder denoising steps. More denoising steps usually lead to a higher quality image at
                the expense of slower inference. For more specific timestep spacing, you can pass customized
                `timesteps`
            prior_timesteps (`List[float]`, *optional*):
                Custom timesteps to use for the denoising process for the prior. If not defined, equal spaced
                `prior_num_inference_steps` timesteps are used. Must be in descending order.
            decoder_timesteps (`List[float]`, *optional*):
                Custom timesteps to use for the denoising process for the decoder. If not defined, equal spaced
                `num_inference_steps` timesteps are used. Must be in descending order.
            decoder_guidance_scale (`float`, *optional*, defaults to 0.0):
                Guidance scale as defined in [Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598).
                `guidance_scale` is defined as `w` of equation 2. of [Imagen
                Paper](https://arxiv.org/pdf/2205.11487.pdf). Guidance scale is enabled by setting `guidance_scale >
                1`. Higher guidance scale encourages to generate images that are closely linked to the text `prompt`,
                usually at the expense of lower image quality.
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
            [`~pipelines.ImagePipelineOutput`] or `tuple` [`~pipelines.ImagePipelineOutput`] if `return_dict` is True,
            otherwise a `tuple`. When returning a tuple, the first element is a list with the generated images.
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
    prior_outputs = self.prior_pipe(prompt=prompt if prompt_embeds is None else
        None, height=height, width=width, num_inference_steps=
        prior_num_inference_steps, timesteps=prior_timesteps,
        guidance_scale=prior_guidance_scale, negative_prompt=
        negative_prompt if negative_prompt_embeds is None else None,
        prompt_embeds=prompt_embeds, negative_prompt_embeds=
        negative_prompt_embeds, num_images_per_prompt=num_images_per_prompt,
        generator=generator, latents=latents, output_type='pt', return_dict
        =False, callback_on_step_end=prior_callback_on_step_end,
        callback_on_step_end_tensor_inputs=
        prior_callback_on_step_end_tensor_inputs, **prior_kwargs)
    image_embeddings = prior_outputs[0]
    outputs = self.decoder_pipe(image_embeddings=image_embeddings, prompt=
        prompt if prompt is not None else '', num_inference_steps=
        num_inference_steps, timesteps=decoder_timesteps, guidance_scale=
        decoder_guidance_scale, negative_prompt=negative_prompt, generator=
        generator, output_type=output_type, return_dict=return_dict,
        callback_on_step_end=callback_on_step_end,
        callback_on_step_end_tensor_inputs=
        callback_on_step_end_tensor_inputs, **kwargs)
    return outputs

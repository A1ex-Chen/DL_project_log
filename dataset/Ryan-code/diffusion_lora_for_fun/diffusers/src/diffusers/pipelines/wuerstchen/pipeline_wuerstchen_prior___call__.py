@torch.no_grad()
@replace_example_docstring(EXAMPLE_DOC_STRING)
def __call__(self, prompt: Optional[Union[str, List[str]]]=None, height:
    int=1024, width: int=1024, num_inference_steps: int=60, timesteps: List
    [float]=None, guidance_scale: float=8.0, negative_prompt: Optional[
    Union[str, List[str]]]=None, prompt_embeds: Optional[torch.Tensor]=None,
    negative_prompt_embeds: Optional[torch.Tensor]=None,
    num_images_per_prompt: Optional[int]=1, generator: Optional[Union[torch
    .Generator, List[torch.Generator]]]=None, latents: Optional[torch.
    Tensor]=None, output_type: Optional[str]='pt', return_dict: bool=True,
    callback_on_step_end: Optional[Callable[[int, int, Dict], None]]=None,
    callback_on_step_end_tensor_inputs: List[str]=['latents'], **kwargs):
    """
        Function invoked when calling the pipeline for generation.

        Args:
            prompt (`str` or `List[str]`):
                The prompt or prompts to guide the image generation.
            height (`int`, *optional*, defaults to 1024):
                The height in pixels of the generated image.
            width (`int`, *optional*, defaults to 1024):
                The width in pixels of the generated image.
            num_inference_steps (`int`, *optional*, defaults to 60):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            timesteps (`List[int]`, *optional*):
                Custom timesteps to use for the denoising process. If not defined, equal spaced `num_inference_steps`
                timesteps are used. Must be in descending order.
            guidance_scale (`float`, *optional*, defaults to 8.0):
                Guidance scale as defined in [Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598).
                `decoder_guidance_scale` is defined as `w` of equation 2. of [Imagen
                Paper](https://arxiv.org/pdf/2205.11487.pdf). Guidance scale is enabled by setting
                `decoder_guidance_scale > 1`. Higher guidance scale encourages to generate images that are closely
                linked to the text `prompt`, usually at the expense of lower image quality.
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the image generation. Ignored when not using guidance (i.e., ignored
                if `decoder_guidance_scale` is less than `1`).
            prompt_embeds (`torch.Tensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not
                provided, text embeddings will be generated from `prompt` input argument.
            negative_prompt_embeds (`torch.Tensor`, *optional*):
                Pre-generated negative text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt
                weighting. If not provided, negative_prompt_embeds will be generated from `negative_prompt` input
                argument.
            num_images_per_prompt (`int`, *optional*, defaults to 1):
                The number of images to generate per prompt.
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
            [`~pipelines.WuerstchenPriorPipelineOutput`] or `tuple` [`~pipelines.WuerstchenPriorPipelineOutput`] if
            `return_dict` is True, otherwise a `tuple`. When returning a tuple, the first element is a list with the
            generated image embeddings.
        """
    callback = kwargs.pop('callback', None)
    callback_steps = kwargs.pop('callback_steps', None)
    if callback is not None:
        deprecate('callback', '1.0.0',
            'Passing `callback` as an input argument to `__call__` is deprecated, consider use `callback_on_step_end`'
            )
    if callback_steps is not None:
        deprecate('callback_steps', '1.0.0',
            'Passing `callback_steps` as an input argument to `__call__` is deprecated, consider use `callback_on_step_end`'
            )
    if callback_on_step_end_tensor_inputs is not None and not all(k in self
        ._callback_tensor_inputs for k in callback_on_step_end_tensor_inputs):
        raise ValueError(
            f'`callback_on_step_end_tensor_inputs` has to be in {self._callback_tensor_inputs}, but found {[k for k in callback_on_step_end_tensor_inputs if k not in self._callback_tensor_inputs]}'
            )
    device = self._execution_device
    self._guidance_scale = guidance_scale
    if prompt is not None and isinstance(prompt, str):
        batch_size = 1
    elif prompt is not None and isinstance(prompt, list):
        batch_size = len(prompt)
    else:
        batch_size = prompt_embeds.shape[0]
    if prompt is not None and not isinstance(prompt, list):
        if isinstance(prompt, str):
            prompt = [prompt]
        else:
            raise TypeError(
                f"'prompt' must be of type 'list' or 'str', but got {type(prompt)}."
                )
    if self.do_classifier_free_guidance:
        if negative_prompt is not None and not isinstance(negative_prompt, list
            ):
            if isinstance(negative_prompt, str):
                negative_prompt = [negative_prompt]
            else:
                raise TypeError(
                    f"'negative_prompt' must be of type 'list' or 'str', but got {type(negative_prompt)}."
                    )
    self.check_inputs(prompt, negative_prompt, num_inference_steps, self.
        do_classifier_free_guidance, prompt_embeds=prompt_embeds,
        negative_prompt_embeds=negative_prompt_embeds)
    prompt_embeds, negative_prompt_embeds = self.encode_prompt(prompt=
        prompt, device=device, num_images_per_prompt=num_images_per_prompt,
        do_classifier_free_guidance=self.do_classifier_free_guidance,
        negative_prompt=negative_prompt, prompt_embeds=prompt_embeds,
        negative_prompt_embeds=negative_prompt_embeds)
    text_encoder_hidden_states = torch.cat([prompt_embeds,
        negative_prompt_embeds]
        ) if negative_prompt_embeds is not None else prompt_embeds
    dtype = text_encoder_hidden_states.dtype
    latent_height = ceil(height / self.config.resolution_multiple)
    latent_width = ceil(width / self.config.resolution_multiple)
    num_channels = self.prior.config.c_in
    effnet_features_shape = (num_images_per_prompt * batch_size,
        num_channels, latent_height, latent_width)
    if timesteps is not None:
        self.scheduler.set_timesteps(timesteps=timesteps, device=device)
        timesteps = self.scheduler.timesteps
        num_inference_steps = len(timesteps)
    else:
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps
    latents = self.prepare_latents(effnet_features_shape, dtype, device,
        generator, latents, self.scheduler)
    self._num_timesteps = len(timesteps[:-1])
    for i, t in enumerate(self.progress_bar(timesteps[:-1])):
        ratio = t.expand(latents.size(0)).to(dtype)
        predicted_image_embedding = self.prior(torch.cat([latents] * 2) if
            self.do_classifier_free_guidance else latents, r=torch.cat([
            ratio] * 2) if self.do_classifier_free_guidance else ratio, c=
            text_encoder_hidden_states)
        if self.do_classifier_free_guidance:
            (predicted_image_embedding_text, predicted_image_embedding_uncond
                ) = predicted_image_embedding.chunk(2)
            predicted_image_embedding = torch.lerp(
                predicted_image_embedding_uncond,
                predicted_image_embedding_text, self.guidance_scale)
        latents = self.scheduler.step(model_output=
            predicted_image_embedding, timestep=ratio, sample=latents,
            generator=generator).prev_sample
        if callback_on_step_end is not None:
            callback_kwargs = {}
            for k in callback_on_step_end_tensor_inputs:
                callback_kwargs[k] = locals()[k]
            callback_outputs = callback_on_step_end(self, i, t, callback_kwargs
                )
            latents = callback_outputs.pop('latents', latents)
            text_encoder_hidden_states = callback_outputs.pop(
                'text_encoder_hidden_states', text_encoder_hidden_states)
            negative_prompt_embeds = callback_outputs.pop(
                'negative_prompt_embeds', negative_prompt_embeds)
        if callback is not None and i % callback_steps == 0:
            step_idx = i // getattr(self.scheduler, 'order', 1)
            callback(step_idx, t, latents)
    latents = latents * self.config.latent_mean - self.config.latent_std
    self.maybe_free_model_hooks()
    if output_type == 'np':
        latents = latents.cpu().float().numpy()
    if not return_dict:
        return latents,
    return WuerstchenPriorPipelineOutput(latents)

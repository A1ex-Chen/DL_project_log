@torch.no_grad()
@replace_example_docstring(EXAMPLE_DOC_STRING)
def __call__(self, prompt: Union[str, List[str]]=None, num_inference_steps:
    int=25, guidance_scale: float=3.0, negative_prompt: Optional[Union[str,
    List[str]]]=None, num_images_per_prompt: Optional[int]=1, height:
    Optional[int]=1024, width: Optional[int]=1024, generator: Optional[
    Union[torch.Generator, List[torch.Generator]]]=None, prompt_embeds:
    Optional[torch.Tensor]=None, negative_prompt_embeds: Optional[torch.
    Tensor]=None, attention_mask: Optional[torch.Tensor]=None,
    negative_attention_mask: Optional[torch.Tensor]=None, output_type:
    Optional[str]='pil', return_dict: bool=True, latents=None,
    callback_on_step_end: Optional[Callable[[int, int, Dict], None]]=None,
    callback_on_step_end_tensor_inputs: List[str]=['latents'], **kwargs):
    """
        Function invoked when calling the pipeline for generation.

        Args:
            prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts to guide the image generation. If not defined, one has to pass `prompt_embeds`.
                instead.
            num_inference_steps (`int`, *optional*, defaults to 25):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            timesteps (`List[int]`, *optional*):
                Custom timesteps to use for the denoising process. If not defined, equal spaced `num_inference_steps`
                timesteps are used. Must be in descending order.
            guidance_scale (`float`, *optional*, defaults to 3.0):
                Guidance scale as defined in [Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598).
                `guidance_scale` is defined as `w` of equation 2. of [Imagen
                Paper](https://arxiv.org/pdf/2205.11487.pdf). Guidance scale is enabled by setting `guidance_scale >
                1`. Higher guidance scale encourages to generate images that are closely linked to the text `prompt`,
                usually at the expense of lower image quality.
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the image generation. If not defined, one has to pass
                `negative_prompt_embeds` instead. Ignored when not using guidance (i.e., ignored if `guidance_scale` is
                less than `1`).
            num_images_per_prompt (`int`, *optional*, defaults to 1):
                The number of images to generate per prompt.
            height (`int`, *optional*, defaults to self.unet.config.sample_size):
                The height in pixels of the generated image.
            width (`int`, *optional*, defaults to self.unet.config.sample_size):
                The width in pixels of the generated image.
            eta (`float`, *optional*, defaults to 0.0):
                Corresponds to parameter eta (η) in the DDIM paper: https://arxiv.org/abs/2010.02502. Only applies to
                [`schedulers.DDIMScheduler`], will be ignored for others.
            generator (`torch.Generator` or `List[torch.Generator]`, *optional*):
                One or a list of [torch generator(s)](https://pytorch.org/docs/stable/generated/torch.Generator.html)
                to make generation deterministic.
            prompt_embeds (`torch.Tensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not
                provided, text embeddings will be generated from `prompt` input argument.
            negative_prompt_embeds (`torch.Tensor`, *optional*):
                Pre-generated negative text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt
                weighting. If not provided, negative_prompt_embeds will be generated from `negative_prompt` input
                argument.
            attention_mask (`torch.Tensor`, *optional*):
                Pre-generated attention mask. Must provide if passing `prompt_embeds` directly.
            negative_attention_mask (`torch.Tensor`, *optional*):
                Pre-generated negative attention mask. Must provide if passing `negative_prompt_embeds` directly.
            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generate image. Choose between
                [PIL](https://pillow.readthedocs.io/en/stable/): `PIL.Image.Image` or `np.array`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~pipelines.stable_diffusion.IFPipelineOutput`] instead of a plain tuple.
            callback (`Callable`, *optional*):
                A function that will be called every `callback_steps` steps during inference. The function will be
                called with the following arguments: `callback(step: int, timestep: int, latents: torch.Tensor)`.
            callback_steps (`int`, *optional*, defaults to 1):
                The frequency at which the `callback` function will be called. If not specified, the callback will be
                called at every step.
            clean_caption (`bool`, *optional*, defaults to `True`):
                Whether or not to clean the caption before creating embeddings. Requires `beautifulsoup4` and `ftfy` to
                be installed. If the dependencies are not installed, the embeddings will be created from the raw
                prompt.
            cross_attention_kwargs (`dict`, *optional*):
                A kwargs dictionary that if specified is passed along to the `AttentionProcessor` as defined under
                `self.processor` in
                [diffusers.models.attention_processor](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py).

        Examples:

        Returns:
            [`~pipelines.ImagePipelineOutput`] or `tuple`

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
    cut_context = True
    device = self._execution_device
    self.check_inputs(prompt, callback_steps, negative_prompt,
        prompt_embeds, negative_prompt_embeds,
        callback_on_step_end_tensor_inputs, attention_mask,
        negative_attention_mask)
    self._guidance_scale = guidance_scale
    if prompt is not None and isinstance(prompt, str):
        batch_size = 1
    elif prompt is not None and isinstance(prompt, list):
        batch_size = len(prompt)
    else:
        batch_size = prompt_embeds.shape[0]
    (prompt_embeds, negative_prompt_embeds, attention_mask,
        negative_attention_mask) = (self.encode_prompt(prompt, self.
        do_classifier_free_guidance, num_images_per_prompt=
        num_images_per_prompt, device=device, negative_prompt=
        negative_prompt, prompt_embeds=prompt_embeds,
        negative_prompt_embeds=negative_prompt_embeds, _cut_context=
        cut_context, attention_mask=attention_mask, negative_attention_mask
        =negative_attention_mask))
    if self.do_classifier_free_guidance:
        prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])
        attention_mask = torch.cat([negative_attention_mask, attention_mask]
            ).bool()
    self.scheduler.set_timesteps(num_inference_steps, device=device)
    timesteps = self.scheduler.timesteps
    height, width = downscale_height_and_width(height, width, 8)
    latents = self.prepare_latents((batch_size * num_images_per_prompt, 4,
        height, width), prompt_embeds.dtype, device, generator, latents,
        self.scheduler)
    if hasattr(self, 'text_encoder_offload_hook'
        ) and self.text_encoder_offload_hook is not None:
        self.text_encoder_offload_hook.offload()
    num_warmup_steps = len(timesteps
        ) - num_inference_steps * self.scheduler.order
    self._num_timesteps = len(timesteps)
    with self.progress_bar(total=num_inference_steps) as progress_bar:
        for i, t in enumerate(timesteps):
            latent_model_input = torch.cat([latents] * 2
                ) if self.do_classifier_free_guidance else latents
            noise_pred = self.unet(latent_model_input, t,
                encoder_hidden_states=prompt_embeds, encoder_attention_mask
                =attention_mask, return_dict=False)[0]
            if self.do_classifier_free_guidance:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = (guidance_scale + 1.0
                    ) * noise_pred_text - guidance_scale * noise_pred_uncond
            latents = self.scheduler.step(noise_pred, t, latents, generator
                =generator).prev_sample
            if callback_on_step_end is not None:
                callback_kwargs = {}
                for k in callback_on_step_end_tensor_inputs:
                    callback_kwargs[k] = locals()[k]
                callback_outputs = callback_on_step_end(self, i, t,
                    callback_kwargs)
                latents = callback_outputs.pop('latents', latents)
                prompt_embeds = callback_outputs.pop('prompt_embeds',
                    prompt_embeds)
                negative_prompt_embeds = callback_outputs.pop(
                    'negative_prompt_embeds', negative_prompt_embeds)
                attention_mask = callback_outputs.pop('attention_mask',
                    attention_mask)
                negative_attention_mask = callback_outputs.pop(
                    'negative_attention_mask', negative_attention_mask)
            if i == len(timesteps) - 1 or i + 1 > num_warmup_steps and (i + 1
                ) % self.scheduler.order == 0:
                progress_bar.update()
                if callback is not None and i % callback_steps == 0:
                    step_idx = i // getattr(self.scheduler, 'order', 1)
                    callback(step_idx, t, latents)
        if output_type not in ['pt', 'np', 'pil', 'latent']:
            raise ValueError(
                f'Only the output types `pt`, `pil`, `np` and `latent` are supported not output_type={output_type}'
                )
        if not output_type == 'latent':
            image = self.movq.decode(latents, force_not_quantize=True)['sample'
                ]
            if output_type in ['np', 'pil']:
                image = image * 0.5 + 0.5
                image = image.clamp(0, 1)
                image = image.cpu().permute(0, 2, 3, 1).float().numpy()
            if output_type == 'pil':
                image = self.numpy_to_pil(image)
        else:
            image = latents
        self.maybe_free_model_hooks()
        if not return_dict:
            return image,
        return ImagePipelineOutput(images=image)

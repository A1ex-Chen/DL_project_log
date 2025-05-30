@torch.no_grad()
@replace_example_docstring(EXAMPLE_DOC_STRING)
def __call__(self, prompt: Optional[Union[str, List[str]]]=None, images:
    Union[torch.Tensor, PIL.Image.Image, List[torch.Tensor], List[PIL.Image
    .Image]]=None, height: int=1024, width: int=1024, num_inference_steps:
    int=20, timesteps: List[float]=None, guidance_scale: float=4.0,
    negative_prompt: Optional[Union[str, List[str]]]=None, prompt_embeds:
    Optional[torch.Tensor]=None, prompt_embeds_pooled: Optional[torch.
    Tensor]=None, negative_prompt_embeds: Optional[torch.Tensor]=None,
    negative_prompt_embeds_pooled: Optional[torch.Tensor]=None,
    image_embeds: Optional[torch.Tensor]=None, num_images_per_prompt:
    Optional[int]=1, generator: Optional[Union[torch.Generator, List[torch.
    Generator]]]=None, latents: Optional[torch.Tensor]=None, output_type:
    Optional[str]='pt', return_dict: bool=True, callback_on_step_end:
    Optional[Callable[[int, int, Dict], None]]=None,
    callback_on_step_end_tensor_inputs: List[str]=['latents']):
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
            prompt_embeds_pooled (`torch.Tensor`, *optional*):
                Pre-generated pooled text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting.
                If not provided, pooled text embeddings will be generated from `prompt` input argument.
            negative_prompt_embeds (`torch.Tensor`, *optional*):
                Pre-generated negative text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt
                weighting. If not provided, negative_prompt_embeds will be generated from `negative_prompt` input
                argument.
            negative_prompt_embeds_pooled (`torch.Tensor`, *optional*):
                Pre-generated negative pooled text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt
                weighting. If not provided, negative_prompt_embeds_pooled will be generated from `negative_prompt`
                input argument.
            image_embeds (`torch.Tensor`, *optional*):
                Pre-generated image embeddings. Can be used to easily tweak image inputs, *e.g.* prompt weighting. If
                not provided, image embeddings will be generated from `image` input argument if existing.
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
            [`StableCascadePriorPipelineOutput`] or `tuple` [`StableCascadePriorPipelineOutput`] if `return_dict` is
            True, otherwise a `tuple`. When returning a tuple, the first element is a list with the generated image
            embeddings.
        """
    device = self._execution_device
    dtype = next(self.prior.parameters()).dtype
    self._guidance_scale = guidance_scale
    if prompt is not None and isinstance(prompt, str):
        batch_size = 1
    elif prompt is not None and isinstance(prompt, list):
        batch_size = len(prompt)
    else:
        batch_size = prompt_embeds.shape[0]
    self.check_inputs(prompt, images=images, image_embeds=image_embeds,
        negative_prompt=negative_prompt, prompt_embeds=prompt_embeds,
        prompt_embeds_pooled=prompt_embeds_pooled, negative_prompt_embeds=
        negative_prompt_embeds, negative_prompt_embeds_pooled=
        negative_prompt_embeds_pooled, callback_on_step_end_tensor_inputs=
        callback_on_step_end_tensor_inputs)
    (prompt_embeds, prompt_embeds_pooled, negative_prompt_embeds,
        negative_prompt_embeds_pooled) = (self.encode_prompt(prompt=prompt,
        device=device, batch_size=batch_size, num_images_per_prompt=
        num_images_per_prompt, do_classifier_free_guidance=self.
        do_classifier_free_guidance, negative_prompt=negative_prompt,
        prompt_embeds=prompt_embeds, prompt_embeds_pooled=
        prompt_embeds_pooled, negative_prompt_embeds=negative_prompt_embeds,
        negative_prompt_embeds_pooled=negative_prompt_embeds_pooled))
    if images is not None:
        image_embeds_pooled, uncond_image_embeds_pooled = self.encode_image(
            images=images, device=device, dtype=dtype, batch_size=
            batch_size, num_images_per_prompt=num_images_per_prompt)
    elif image_embeds is not None:
        image_embeds_pooled = image_embeds.repeat(batch_size *
            num_images_per_prompt, 1, 1)
        uncond_image_embeds_pooled = torch.zeros_like(image_embeds_pooled)
    else:
        image_embeds_pooled = torch.zeros(batch_size *
            num_images_per_prompt, 1, self.prior.config.
            clip_image_in_channels, device=device, dtype=dtype)
        uncond_image_embeds_pooled = torch.zeros(batch_size *
            num_images_per_prompt, 1, self.prior.config.
            clip_image_in_channels, device=device, dtype=dtype)
    if self.do_classifier_free_guidance:
        image_embeds = torch.cat([image_embeds_pooled,
            uncond_image_embeds_pooled], dim=0)
    else:
        image_embeds = image_embeds_pooled
    text_encoder_hidden_states = torch.cat([prompt_embeds,
        negative_prompt_embeds]
        ) if negative_prompt_embeds is not None else prompt_embeds
    text_encoder_pooled = torch.cat([prompt_embeds_pooled,
        negative_prompt_embeds_pooled]
        ) if negative_prompt_embeds is not None else prompt_embeds_pooled
    self.scheduler.set_timesteps(num_inference_steps, device=device)
    timesteps = self.scheduler.timesteps
    latents = self.prepare_latents(batch_size, height, width,
        num_images_per_prompt, dtype, device, generator, latents, self.
        scheduler)
    if isinstance(self.scheduler, DDPMWuerstchenScheduler):
        timesteps = timesteps[:-1]
    elif self.scheduler.config.clip_sample:
        self.scheduler.config.clip_sample = False
        logger.warning(' set `clip_sample` to be False')
    if hasattr(self.scheduler, 'betas'):
        alphas = 1.0 - self.scheduler.betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
    else:
        alphas_cumprod = []
    self._num_timesteps = len(timesteps)
    for i, t in enumerate(self.progress_bar(timesteps)):
        if not isinstance(self.scheduler, DDPMWuerstchenScheduler):
            if len(alphas_cumprod) > 0:
                timestep_ratio = self.get_timestep_ratio_conditioning(t.
                    long().cpu(), alphas_cumprod)
                timestep_ratio = timestep_ratio.expand(latents.size(0)).to(
                    dtype).to(device)
            else:
                timestep_ratio = t.float().div(self.scheduler.timesteps[-1]
                    ).expand(latents.size(0)).to(dtype)
        else:
            timestep_ratio = t.expand(latents.size(0)).to(dtype)
        predicted_image_embedding = self.prior(sample=torch.cat([latents] *
            2) if self.do_classifier_free_guidance else latents,
            timestep_ratio=torch.cat([timestep_ratio] * 2) if self.
            do_classifier_free_guidance else timestep_ratio,
            clip_text_pooled=text_encoder_pooled, clip_text=
            text_encoder_hidden_states, clip_img=image_embeds, return_dict=
            False)[0]
        if self.do_classifier_free_guidance:
            (predicted_image_embedding_text, predicted_image_embedding_uncond
                ) = predicted_image_embedding.chunk(2)
            predicted_image_embedding = torch.lerp(
                predicted_image_embedding_uncond,
                predicted_image_embedding_text, self.guidance_scale)
        if not isinstance(self.scheduler, DDPMWuerstchenScheduler):
            timestep_ratio = t
        latents = self.scheduler.step(model_output=
            predicted_image_embedding, timestep=timestep_ratio, sample=
            latents, generator=generator).prev_sample
        if callback_on_step_end is not None:
            callback_kwargs = {}
            for k in callback_on_step_end_tensor_inputs:
                callback_kwargs[k] = locals()[k]
            callback_outputs = callback_on_step_end(self, i, t, callback_kwargs
                )
            latents = callback_outputs.pop('latents', latents)
            prompt_embeds = callback_outputs.pop('prompt_embeds', prompt_embeds
                )
            negative_prompt_embeds = callback_outputs.pop(
                'negative_prompt_embeds', negative_prompt_embeds)
    self.maybe_free_model_hooks()
    if output_type == 'np':
        latents = latents.cpu().float().numpy()
        prompt_embeds = prompt_embeds.cpu().float().numpy()
        negative_prompt_embeds = negative_prompt_embeds.cpu().float().numpy(
            ) if negative_prompt_embeds is not None else None
    if not return_dict:
        return (latents, prompt_embeds, prompt_embeds_pooled,
            negative_prompt_embeds, negative_prompt_embeds_pooled)
    return StableCascadePriorPipelineOutput(image_embeddings=latents,
        prompt_embeds=prompt_embeds, prompt_embeds_pooled=
        prompt_embeds_pooled, negative_prompt_embeds=negative_prompt_embeds,
        negative_prompt_embeds_pooled=negative_prompt_embeds_pooled)

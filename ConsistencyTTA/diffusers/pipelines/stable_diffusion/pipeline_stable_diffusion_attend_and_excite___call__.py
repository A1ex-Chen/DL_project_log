@torch.no_grad()
@replace_example_docstring(EXAMPLE_DOC_STRING)
def __call__(self, prompt: Union[str, List[str]], token_indices: Union[List
    [int], List[List[int]]], height: Optional[int]=None, width: Optional[
    int]=None, num_inference_steps: int=50, guidance_scale: float=7.5,
    negative_prompt: Optional[Union[str, List[str]]]=None,
    num_images_per_prompt: int=1, eta: float=0.0, generator: Optional[Union
    [torch.Generator, List[torch.Generator]]]=None, latents: Optional[torch
    .FloatTensor]=None, prompt_embeds: Optional[torch.FloatTensor]=None,
    negative_prompt_embeds: Optional[torch.FloatTensor]=None, output_type:
    Optional[str]='pil', return_dict: bool=True, callback: Optional[
    Callable[[int, int, torch.FloatTensor], None]]=None, callback_steps:
    int=1, cross_attention_kwargs: Optional[Dict[str, Any]]=None,
    max_iter_to_alter: int=25, thresholds: dict={(0): 0.05, (10): 0.5, (20):
    0.8}, scale_factor: int=20, attn_res: int=16):
    """
        Function invoked when calling the pipeline for generation.

        Args:
            prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts to guide the image generation. If not defined, one has to pass `prompt_embeds`.
                instead.
            token_indices (`List[int]`):
                The token indices to alter with attend-and-excite.
            height (`int`, *optional*, defaults to self.unet.config.sample_size * self.vae_scale_factor):
                The height in pixels of the generated image.
            width (`int`, *optional*, defaults to self.unet.config.sample_size * self.vae_scale_factor):
                The width in pixels of the generated image.
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
                `negative_prompt_embeds` instead. Ignored when not using guidance (i.e., ignored if `guidance_scale` is
                less than `1`).
            num_images_per_prompt (`int`, *optional*, defaults to 1):
                The number of images to generate per prompt.
            eta (`float`, *optional*, defaults to 0.0):
                Corresponds to parameter eta (η) in the DDIM paper: https://arxiv.org/abs/2010.02502. Only applies to
                [`schedulers.DDIMScheduler`], will be ignored for others.
            generator (`torch.Generator` or `List[torch.Generator]`, *optional*):
                One or a list of [torch generator(s)](https://pytorch.org/docs/stable/generated/torch.Generator.html)
                to make generation deterministic.
            latents (`torch.FloatTensor`, *optional*):
                Pre-generated noisy latents, sampled from a Gaussian distribution, to be used as inputs for image
                generation. Can be used to tweak the same generation with different prompts. If not provided, a latents
                tensor will ge generated by sampling using the supplied random `generator`.
            prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not
                provided, text embeddings will be generated from `prompt` input argument.
            negative_prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated negative text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt
                weighting. If not provided, negative_prompt_embeds will be generated from `negative_prompt` input
                argument.
            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generate image. Choose between
                [PIL](https://pillow.readthedocs.io/en/stable/): `PIL.Image.Image` or `np.array`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] instead of a
                plain tuple.
            callback (`Callable`, *optional*):
                A function that will be called every `callback_steps` steps during inference. The function will be
                called with the following arguments: `callback(step: int, timestep: int, latents: torch.FloatTensor)`.
            callback_steps (`int`, *optional*, defaults to 1):
                The frequency at which the `callback` function will be called. If not specified, the callback will be
                called at every step.
            cross_attention_kwargs (`dict`, *optional*):
                A kwargs dictionary that if specified is passed along to the `AttentionProcessor` as defined under
                `self.processor` in
                [diffusers.cross_attention](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/cross_attention.py).
            max_iter_to_alter (`int`, *optional*, defaults to `25`):
                Number of denoising steps to apply attend-and-excite. The first <max_iter_to_alter> denoising steps are
                where the attend-and-excite is applied. I.e. if `max_iter_to_alter` is 25 and there are a total of `30`
                denoising steps, the first 25 denoising steps will apply attend-and-excite and the last 5 will not
                apply attend-and-excite.
            thresholds (`dict`, *optional*, defaults to `{0: 0.05, 10: 0.5, 20: 0.8}`):
                Dictionary defining the iterations and desired thresholds to apply iterative latent refinement in.
            scale_factor (`int`, *optional*, default to 20):
                Scale factor that controls the step size of each Attend and Excite update.
            attn_res (`int`, *optional*, default to 16):
                The resolution of most semantic attention map.

        Examples:

        Returns:
            [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] or `tuple`:
            [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] if `return_dict` is True, otherwise a `tuple.
            When returning a tuple, the first element is a list with the generated images, and the second element is a
            list of `bool`s denoting whether the corresponding generated image likely represents "not-safe-for-work"
            (nsfw) content, according to the `safety_checker`. :type attention_store: object
        """
    height = height or self.unet.config.sample_size * self.vae_scale_factor
    width = width or self.unet.config.sample_size * self.vae_scale_factor
    self.check_inputs(prompt, token_indices, height, width, callback_steps,
        negative_prompt, prompt_embeds, negative_prompt_embeds)
    if prompt is not None and isinstance(prompt, str):
        batch_size = 1
    elif prompt is not None and isinstance(prompt, list):
        batch_size = len(prompt)
    else:
        batch_size = prompt_embeds.shape[0]
    device = self._execution_device
    do_classifier_free_guidance = guidance_scale > 1.0
    prompt_embeds = self._encode_prompt(prompt, device,
        num_images_per_prompt, do_classifier_free_guidance, negative_prompt,
        prompt_embeds=prompt_embeds, negative_prompt_embeds=
        negative_prompt_embeds)
    self.scheduler.set_timesteps(num_inference_steps, device=device)
    timesteps = self.scheduler.timesteps
    num_channels_latents = self.unet.in_channels
    latents = self.prepare_latents(batch_size * num_images_per_prompt,
        num_channels_latents, height, width, prompt_embeds.dtype, device,
        generator, latents)
    extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)
    self.attention_store = AttentionStore(attn_res=attn_res)
    self.register_attention_control()
    scale_range = np.linspace(1.0, 0.5, len(self.scheduler.timesteps))
    step_size = scale_factor * np.sqrt(scale_range)
    text_embeddings = prompt_embeds[batch_size * num_images_per_prompt:
        ] if do_classifier_free_guidance else prompt_embeds
    if isinstance(token_indices[0], int):
        token_indices = [token_indices]
    indices = []
    for ind in token_indices:
        indices = indices + [ind] * num_images_per_prompt
    num_warmup_steps = len(timesteps
        ) - num_inference_steps * self.scheduler.order
    with self.progress_bar(total=num_inference_steps) as progress_bar:
        for i, t in enumerate(timesteps):
            with torch.enable_grad():
                latents = latents.clone().detach().requires_grad_(True)
                updated_latents = []
                for latent, index, text_embedding in zip(latents, indices,
                    text_embeddings):
                    latent = latent.unsqueeze(0)
                    text_embedding = text_embedding.unsqueeze(0)
                    self.unet(latent, t, encoder_hidden_states=
                        text_embedding, cross_attention_kwargs=
                        cross_attention_kwargs).sample
                    self.unet.zero_grad()
                    max_attention_per_index = (self.
                        _aggregate_and_get_max_attention_per_token(indices=
                        index))
                    loss = self._compute_loss(max_attention_per_index=
                        max_attention_per_index)
                    if i in thresholds.keys() and loss > 1.0 - thresholds[i]:
                        loss, latent, max_attention_per_index = (self.
                            _perform_iterative_refinement_step(latents=
                            latent, indices=index, loss=loss, threshold=
                            thresholds[i], text_embeddings=text_embedding,
                            step_size=step_size[i], t=t))
                    if i < max_iter_to_alter:
                        if loss != 0:
                            latent = self._update_latent(latents=latent,
                                loss=loss, step_size=step_size[i])
                        logger.info(f'Iteration {i} | Loss: {loss:0.4f}')
                    updated_latents.append(latent)
                latents = torch.cat(updated_latents, dim=0)
            latent_model_input = torch.cat([latents] * 2
                ) if do_classifier_free_guidance else latents
            latent_model_input = self.scheduler.scale_model_input(
                latent_model_input, t)
            noise_pred = self.unet(latent_model_input, t,
                encoder_hidden_states=prompt_embeds, cross_attention_kwargs
                =cross_attention_kwargs).sample
            if do_classifier_free_guidance:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (
                    noise_pred_text - noise_pred_uncond)
            latents = self.scheduler.step(noise_pred, t, latents, **
                extra_step_kwargs).prev_sample
            if i == len(timesteps) - 1 or i + 1 > num_warmup_steps and (i + 1
                ) % self.scheduler.order == 0:
                progress_bar.update()
                if callback is not None and i % callback_steps == 0:
                    callback(i, t, latents)
    image = self.decode_latents(latents)
    image, has_nsfw_concept = self.run_safety_checker(image, device,
        prompt_embeds.dtype)
    if output_type == 'pil':
        image = self.numpy_to_pil(image)
    if not return_dict:
        return image, has_nsfw_concept
    return StableDiffusionPipelineOutput(images=image,
        nsfw_content_detected=has_nsfw_concept)

@torch.no_grad()
def __call__(self, prompt: Union[str, List[str]]=None, rgb:
    PipelineImageInput=None, depth: PipelineDepthInput=None,
    num_inference_steps: int=75, guidance_scale: float=9.0, noise_level:
    int=20, negative_prompt: Optional[Union[str, List[str]]]=None,
    num_images_per_prompt: Optional[int]=1, eta: float=0.0, generator:
    Optional[Union[torch.Generator, List[torch.Generator]]]=None, latents:
    Optional[torch.Tensor]=None, prompt_embeds: Optional[torch.Tensor]=None,
    negative_prompt_embeds: Optional[torch.Tensor]=None, output_type:
    Optional[str]='pil', return_dict: bool=True, callback: Optional[
    Callable[[int, int, torch.Tensor], None]]=None, callback_steps: int=1,
    cross_attention_kwargs: Optional[Dict[str, Any]]=None, target_res:
    Optional[List[int]]=[1024, 1024]):
    """
        The call function to the pipeline for generation.

        Args:
            prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts to guide image generation. If not defined, you need to pass `prompt_embeds`.
            image (`torch.Tensor`, `PIL.Image.Image`, `np.ndarray`, `List[torch.Tensor]`, `List[PIL.Image.Image]`, or `List[np.ndarray]`):
                `Image` or tensor representing an image batch to be upscaled.
            num_inference_steps (`int`, *optional*, defaults to 50):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            guidance_scale (`float`, *optional*, defaults to 5.0):
                A higher guidance scale value encourages the model to generate images closely linked to the text
                `prompt` at the expense of lower image quality. Guidance scale is enabled when `guidance_scale > 1`.
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts to guide what to not include in image generation. If not defined, you need to
                pass `negative_prompt_embeds` instead. Ignored when not using guidance (`guidance_scale < 1`).
            num_images_per_prompt (`int`, *optional*, defaults to 1):
                The number of images to generate per prompt.
            eta (`float`, *optional*, defaults to 0.0):
                Corresponds to parameter eta (η) from the [DDIM](https://arxiv.org/abs/2010.02502) paper. Only applies
                to the [`~schedulers.DDIMScheduler`], and is ignored in other schedulers.
            generator (`torch.Generator` or `List[torch.Generator]`, *optional*):
                A [`torch.Generator`](https://pytorch.org/docs/stable/generated/torch.Generator.html) to make
                generation deterministic.
            latents (`torch.Tensor`, *optional*):
                Pre-generated noisy latents sampled from a Gaussian distribution, to be used as inputs for image
                generation. Can be used to tweak the same generation with different prompts. If not provided, a latents
                tensor is generated by sampling using the supplied random `generator`.
            prompt_embeds (`torch.Tensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs (prompt weighting). If not
                provided, text embeddings are generated from the `prompt` input argument.
            negative_prompt_embeds (`torch.Tensor`, *optional*):
                Pre-generated negative text embeddings. Can be used to easily tweak text inputs (prompt weighting). If
                not provided, `negative_prompt_embeds` are generated from the `negative_prompt` input argument.
            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generated image. Choose between `PIL.Image` or `np.array`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] instead of a
                plain tuple.
            callback (`Callable`, *optional*):
                A function that calls every `callback_steps` steps during inference. The function is called with the
                following arguments: `callback(step: int, timestep: int, latents: torch.Tensor)`.
            callback_steps (`int`, *optional*, defaults to 1):
                The frequency at which the `callback` function is called. If not specified, the callback is called at
                every step.
            cross_attention_kwargs (`dict`, *optional*):
                A kwargs dictionary that if specified is passed along to the [`AttentionProcessor`] as defined in
                [`self.processor`](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py).

        Examples:

        Returns:
            [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] or `tuple`:
                If `return_dict` is `True`, [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] is returned,
                otherwise a `tuple` is returned where the first element is a list with the generated images and the
                second element is a list of `bool`s indicating whether the corresponding generated image contains
                "not-safe-for-work" (nsfw) content.
        """
    self.check_inputs(prompt, rgb, noise_level, callback_steps,
        negative_prompt, prompt_embeds, negative_prompt_embeds)
    if prompt is not None and isinstance(prompt, str):
        batch_size = 1
    elif prompt is not None and isinstance(prompt, list):
        batch_size = len(prompt)
    else:
        batch_size = prompt_embeds.shape[0]
    device = self._execution_device
    do_classifier_free_guidance = guidance_scale > 1.0
    prompt_embeds, negative_prompt_embeds = self.encode_prompt(prompt,
        device, num_images_per_prompt, do_classifier_free_guidance,
        negative_prompt, prompt_embeds=prompt_embeds,
        negative_prompt_embeds=negative_prompt_embeds)
    if do_classifier_free_guidance:
        prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])
    rgb, depth = self.image_processor.preprocess(rgb, depth, target_res=
        target_res)
    rgb = rgb.to(dtype=prompt_embeds.dtype, device=device)
    depth = depth.to(dtype=prompt_embeds.dtype, device=device)
    self.scheduler.set_timesteps(num_inference_steps, device=device)
    timesteps = self.scheduler.timesteps
    image = torch.cat([rgb, depth], axis=1)
    latent_space_image = self.vae.encode(image).latent_dist.sample(generator)
    latent_space_image *= self.vae.scaling_factor
    noise_level = torch.tensor([noise_level], dtype=torch.long, device=device)
    batch_multiplier = 2 if do_classifier_free_guidance else 1
    latent_space_image = torch.cat([latent_space_image] * batch_multiplier *
        num_images_per_prompt)
    noise_level = torch.cat([noise_level] * latent_space_image.shape[0])
    height, width = latent_space_image.shape[2:]
    num_channels_latents = self.vae.config.latent_channels
    latents = self.prepare_latents(batch_size * num_images_per_prompt,
        num_channels_latents, height, width, prompt_embeds.dtype, device,
        generator, latents)
    num_channels_image = latent_space_image.shape[1]
    if (num_channels_latents + num_channels_image != self.unet.config.
        in_channels):
        raise ValueError(
            f'Incorrect configuration settings! The config of `pipeline.unet`: {self.unet.config} expects {self.unet.config.in_channels} but received `num_channels_latents`: {num_channels_latents} + `num_channels_image`: {num_channels_image}  = {num_channels_latents + num_channels_image}. Please verify the config of `pipeline.unet` or your `image` input.'
            )
    extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)
    num_warmup_steps = len(timesteps
        ) - num_inference_steps * self.scheduler.order
    with self.progress_bar(total=num_inference_steps) as progress_bar:
        for i, t in enumerate(timesteps):
            latent_model_input = torch.cat([latents] * 2
                ) if do_classifier_free_guidance else latents
            latent_model_input = self.scheduler.scale_model_input(
                latent_model_input, t)
            latent_model_input = torch.cat([latent_model_input,
                latent_space_image], dim=1)
            noise_pred = self.unet(latent_model_input, t,
                encoder_hidden_states=prompt_embeds, cross_attention_kwargs
                =cross_attention_kwargs, class_labels=noise_level,
                return_dict=False)[0]
            if do_classifier_free_guidance:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (
                    noise_pred_text - noise_pred_uncond)
            latents = self.scheduler.step(noise_pred, t, latents, **
                extra_step_kwargs, return_dict=False)[0]
            if i == len(timesteps) - 1 or i + 1 > num_warmup_steps and (i + 1
                ) % self.scheduler.order == 0:
                progress_bar.update()
                if callback is not None and i % callback_steps == 0:
                    callback(i, t, latents)
    if not output_type == 'latent':
        needs_upcasting = (self.vae.dtype == torch.float16 and self.vae.
            config.force_upcast)
        if needs_upcasting:
            self.upcast_vae()
            latents = latents.to(next(iter(self.vae.post_quant_conv.
                parameters())).dtype)
        image = self.vae.decode(latents / self.vae.scaling_factor,
            return_dict=False)[0]
        if needs_upcasting:
            self.vae.to(dtype=torch.float16)
        image, has_nsfw_concept = self.run_safety_checker(image, device,
            prompt_embeds.dtype)
    else:
        image = latents
        has_nsfw_concept = None
    if has_nsfw_concept is None:
        do_denormalize = [True] * image.shape[0]
    else:
        do_denormalize = [(not has_nsfw) for has_nsfw in has_nsfw_concept]
    rgb, depth = self.image_processor.postprocess(image, output_type=
        output_type, do_denormalize=do_denormalize)
    if output_type == 'pil' and self.watermarker is not None:
        rgb = self.watermarker.apply_watermark(rgb)
    if hasattr(self, 'final_offload_hook'
        ) and self.final_offload_hook is not None:
        self.final_offload_hook.offload()
    if not return_dict:
        return (rgb, depth), has_nsfw_concept
    return LDM3DPipelineOutput(rgb=rgb, depth=depth, nsfw_content_detected=
        has_nsfw_concept)
@torch.no_grad()
@replace_example_docstring(EXAMPLE_DOC_STRING)
def __call__(self, num_controlnet: int, fp16: bool=True, prompt: Union[str,
    List[str]]=None, image: Union[torch.Tensor, PIL.Image.Image, np.ndarray,
    List[torch.Tensor], List[PIL.Image.Image], List[np.ndarray]]=None,
    control_image: Union[torch.Tensor, PIL.Image.Image, np.ndarray, List[
    torch.Tensor], List[PIL.Image.Image], List[np.ndarray]]=None, height:
    Optional[int]=None, width: Optional[int]=None, strength: float=0.8,
    num_inference_steps: int=50, guidance_scale: float=7.5, negative_prompt:
    Optional[Union[str, List[str]]]=None, num_images_per_prompt: Optional[
    int]=1, eta: float=0.0, generator: Optional[Union[torch.Generator, List
    [torch.Generator]]]=None, latents: Optional[torch.Tensor]=None,
    prompt_embeds: Optional[torch.Tensor]=None, negative_prompt_embeds:
    Optional[torch.Tensor]=None, output_type: Optional[str]='pil',
    return_dict: bool=True, callback: Optional[Callable[[int, int, torch.
    Tensor], None]]=None, callback_steps: int=1, cross_attention_kwargs:
    Optional[Dict[str, Any]]=None, controlnet_conditioning_scale: Union[
    float, List[float]]=0.8, guess_mode: bool=False, control_guidance_start:
    Union[float, List[float]]=0.0, control_guidance_end: Union[float, List[
    float]]=1.0):
    """
        Function invoked when calling the pipeline for generation.

        Args:
            prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts to guide the image generation. If not defined, one has to pass `prompt_embeds`.
                instead.
            image (`torch.Tensor`, `PIL.Image.Image`, `np.ndarray`, `List[torch.Tensor]`, `List[PIL.Image.Image]`, `List[np.ndarray]`,:
                    `List[List[torch.Tensor]]`, `List[List[np.ndarray]]` or `List[List[PIL.Image.Image]]`):
                The initial image will be used as the starting point for the image generation process. Can also accept
                image latents as `image`, if passing latents directly, it will not be encoded again.
            control_image (`torch.Tensor`, `PIL.Image.Image`, `np.ndarray`, `List[torch.Tensor]`, `List[PIL.Image.Image]`, `List[np.ndarray]`,:
                    `List[List[torch.Tensor]]`, `List[List[np.ndarray]]` or `List[List[PIL.Image.Image]]`):
                The ControlNet input condition. ControlNet uses this input condition to generate guidance to Unet. If
                the type is specified as `torch.Tensor`, it is passed to ControlNet as is. `PIL.Image.Image` can
                also be accepted as an image. The dimensions of the output image defaults to `image`'s dimensions. If
                height and/or width are passed, `image` is resized according to them. If multiple ControlNets are
                specified in init, images must be passed as a list such that each element of the list can be correctly
                batched for input to a single controlnet.
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
            latents (`torch.Tensor`, *optional*):
                Pre-generated noisy latents, sampled from a Gaussian distribution, to be used as inputs for image
                generation. Can be used to tweak the same generation with different prompts. If not provided, a latents
                tensor will ge generated by sampling using the supplied random `generator`.
            prompt_embeds (`torch.Tensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not
                provided, text embeddings will be generated from `prompt` input argument.
            negative_prompt_embeds (`torch.Tensor`, *optional*):
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
                called with the following arguments: `callback(step: int, timestep: int, latents: torch.Tensor)`.
            callback_steps (`int`, *optional*, defaults to 1):
                The frequency at which the `callback` function will be called. If not specified, the callback will be
                called at every step.
            cross_attention_kwargs (`dict`, *optional*):
                A kwargs dictionary that if specified is passed along to the `AttentionProcessor` as defined under
                `self.processor` in
                [diffusers.models.attention_processor](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py).
            controlnet_conditioning_scale (`float` or `List[float]`, *optional*, defaults to 1.0):
                The outputs of the controlnet are multiplied by `controlnet_conditioning_scale` before they are added
                to the residual in the original unet. If multiple ControlNets are specified in init, you can set the
                corresponding scale as a list. Note that by default, we use a smaller conditioning scale for inpainting
                than for [`~StableDiffusionControlNetPipeline.__call__`].
            guess_mode (`bool`, *optional*, defaults to `False`):
                In this mode, the ControlNet encoder will try best to recognize the content of the input image even if
                you remove all prompts. The `guidance_scale` between 3.0 and 5.0 is recommended.
            control_guidance_start (`float` or `List[float]`, *optional*, defaults to 0.0):
                The percentage of total steps at which the controlnet starts applying.
            control_guidance_end (`float` or `List[float]`, *optional*, defaults to 1.0):
                The percentage of total steps at which the controlnet stops applying.

        Examples:

        Returns:
            [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] or `tuple`:
            [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] if `return_dict` is True, otherwise a `tuple.
            When returning a tuple, the first element is a list with the generated images, and the second element is a
            list of `bool`s denoting whether the corresponding generated image likely represents "not-safe-for-work"
            (nsfw) content, according to the `safety_checker`.
        """
    if fp16:
        torch_dtype = torch.float16
        np_dtype = np.float16
    else:
        torch_dtype = torch.float32
        np_dtype = np.float32
    if not isinstance(control_guidance_start, list) and isinstance(
        control_guidance_end, list):
        control_guidance_start = len(control_guidance_end) * [
            control_guidance_start]
    elif not isinstance(control_guidance_end, list) and isinstance(
        control_guidance_start, list):
        control_guidance_end = len(control_guidance_start) * [
            control_guidance_end]
    elif not isinstance(control_guidance_start, list) and not isinstance(
        control_guidance_end, list):
        mult = num_controlnet
        control_guidance_start, control_guidance_end = mult * [
            control_guidance_start], mult * [control_guidance_end]
    self.check_inputs(num_controlnet, prompt, control_image, callback_steps,
        negative_prompt, prompt_embeds, negative_prompt_embeds,
        controlnet_conditioning_scale, control_guidance_start,
        control_guidance_end)
    if prompt is not None and isinstance(prompt, str):
        batch_size = 1
    elif prompt is not None and isinstance(prompt, list):
        batch_size = len(prompt)
    else:
        batch_size = prompt_embeds.shape[0]
    device = self._execution_device
    do_classifier_free_guidance = guidance_scale > 1.0
    if num_controlnet > 1 and isinstance(controlnet_conditioning_scale, float):
        controlnet_conditioning_scale = [controlnet_conditioning_scale
            ] * num_controlnet
    prompt_embeds = self._encode_prompt(prompt, num_images_per_prompt,
        do_classifier_free_guidance, negative_prompt, prompt_embeds=
        prompt_embeds, negative_prompt_embeds=negative_prompt_embeds)
    image = self.image_processor.preprocess(image).to(dtype=torch.float32)
    if num_controlnet == 1:
        control_image = self.prepare_control_image(image=control_image,
            width=width, height=height, batch_size=batch_size *
            num_images_per_prompt, num_images_per_prompt=
            num_images_per_prompt, device=device, dtype=torch_dtype,
            do_classifier_free_guidance=do_classifier_free_guidance,
            guess_mode=guess_mode)
    elif num_controlnet > 1:
        control_images = []
        for control_image_ in control_image:
            control_image_ = self.prepare_control_image(image=
                control_image_, width=width, height=height, batch_size=
                batch_size * num_images_per_prompt, num_images_per_prompt=
                num_images_per_prompt, device=device, dtype=torch_dtype,
                do_classifier_free_guidance=do_classifier_free_guidance,
                guess_mode=guess_mode)
            control_images.append(control_image_)
        control_image = control_images
    else:
        assert False
    self.scheduler.set_timesteps(num_inference_steps, device=device)
    timesteps, num_inference_steps = self.get_timesteps(num_inference_steps,
        strength, device)
    latent_timestep = timesteps[:1].repeat(batch_size * num_images_per_prompt)
    latents = self.prepare_latents(image, latent_timestep, batch_size,
        num_images_per_prompt, torch_dtype, device, generator)
    extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)
    controlnet_keep = []
    for i in range(len(timesteps)):
        keeps = [(1.0 - float(i / len(timesteps) < s or (i + 1) / len(
            timesteps) > e)) for s, e in zip(control_guidance_start,
            control_guidance_end)]
        controlnet_keep.append(keeps[0] if num_controlnet == 1 else keeps)
    num_warmup_steps = len(timesteps
        ) - num_inference_steps * self.scheduler.order
    with self.progress_bar(total=num_inference_steps) as progress_bar:
        for i, t in enumerate(timesteps):
            latent_model_input = torch.cat([latents] * 2
                ) if do_classifier_free_guidance else latents
            latent_model_input = self.scheduler.scale_model_input(
                latent_model_input, t)
            if isinstance(controlnet_keep[i], list):
                cond_scale = [(c * s) for c, s in zip(
                    controlnet_conditioning_scale, controlnet_keep[i])]
            else:
                controlnet_cond_scale = controlnet_conditioning_scale
                if isinstance(controlnet_cond_scale, list):
                    controlnet_cond_scale = controlnet_cond_scale[0]
                cond_scale = controlnet_cond_scale * controlnet_keep[i]
            _latent_model_input = latent_model_input.cpu().detach().numpy()
            _prompt_embeds = np.array(prompt_embeds, dtype=np_dtype)
            _t = np.array([t.cpu().detach().numpy()], dtype=np_dtype)
            if num_controlnet == 1:
                control_images = np.array([control_image], dtype=np_dtype)
            else:
                control_images = []
                for _control_img in control_image:
                    _control_img = _control_img.cpu().detach().numpy()
                    control_images.append(_control_img)
                control_images = np.array(control_images, dtype=np_dtype)
            control_scales = np.array(cond_scale, dtype=np_dtype)
            control_scales = np.resize(control_scales, (num_controlnet, 1))
            noise_pred = self.unet(sample=_latent_model_input, timestep=_t,
                encoder_hidden_states=_prompt_embeds, controlnet_conds=
                control_images, conditioning_scales=control_scales)[0]
            noise_pred = torch.from_numpy(noise_pred).to(device)
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
                    step_idx = i // getattr(self.scheduler, 'order', 1)
                    callback(step_idx, t, latents)
    if not output_type == 'latent':
        _latents = latents.cpu().detach().numpy() / 0.18215
        _latents = np.array(_latents, dtype=np_dtype)
        image = self.vae_decoder(latent_sample=_latents)[0]
        image = torch.from_numpy(image).to(device, dtype=torch.float32)
        has_nsfw_concept = None
    else:
        image = latents
        has_nsfw_concept = None
    if has_nsfw_concept is None:
        do_denormalize = [True] * image.shape[0]
    else:
        do_denormalize = [(not has_nsfw) for has_nsfw in has_nsfw_concept]
    image = self.image_processor.postprocess(image, output_type=output_type,
        do_denormalize=do_denormalize)
    if not return_dict:
        return image, has_nsfw_concept
    return StableDiffusionPipelineOutput(images=image,
        nsfw_content_detected=has_nsfw_concept)

@torch.no_grad()
@replace_example_docstring(EXAMPLE_DOC_STRING)
def __call__(self, image: PipelineImageInput, prompt: Union[str, List[str]]
    =None, strength: float=1.0, num_frames: Optional[int]=16, height:
    Optional[int]=None, width: Optional[int]=None, num_inference_steps: int
    =50, guidance_scale: float=7.5, negative_prompt: Optional[Union[str,
    List[str]]]=None, num_videos_per_prompt: Optional[int]=1, eta: float=
    0.0, generator: Optional[Union[torch.Generator, List[torch.Generator]]]
    =None, latents: Optional[torch.Tensor]=None, prompt_embeds: Optional[
    torch.Tensor]=None, negative_prompt_embeds: Optional[torch.Tensor]=None,
    ip_adapter_image: Optional[PipelineImageInput]=None,
    ip_adapter_image_embeds: Optional[List[torch.Tensor]]=None,
    motion_scale: int=0, output_type: Optional[str]='pil', return_dict:
    bool=True, cross_attention_kwargs: Optional[Dict[str, Any]]=None,
    clip_skip: Optional[int]=None, callback_on_step_end: Optional[Callable[
    [int, int, Dict], None]]=None, callback_on_step_end_tensor_inputs: List
    [str]=['latents']):
    """
        The call function to the pipeline for generation.

        Args:
            image (`PipelineImageInput`):
                The input image to be used for video generation.
            prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts to guide image generation. If not defined, you need to pass `prompt_embeds`.
            strength (`float`, *optional*, defaults to 1.0):
                Indicates extent to transform the reference `image`. Must be between 0 and 1.
            height (`int`, *optional*, defaults to `self.unet.config.sample_size * self.vae_scale_factor`):
                The height in pixels of the generated video.
            width (`int`, *optional*, defaults to `self.unet.config.sample_size * self.vae_scale_factor`):
                The width in pixels of the generated video.
            num_frames (`int`, *optional*, defaults to 16):
                The number of video frames that are generated. Defaults to 16 frames which at 8 frames per seconds
                amounts to 2 seconds of video.
            num_inference_steps (`int`, *optional*, defaults to 50):
                The number of denoising steps. More denoising steps usually lead to a higher quality videos at the
                expense of slower inference.
            guidance_scale (`float`, *optional*, defaults to 7.5):
                A higher guidance scale value encourages the model to generate images closely linked to the text
                `prompt` at the expense of lower image quality. Guidance scale is enabled when `guidance_scale > 1`.
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts to guide what to not include in image generation. If not defined, you need to
                pass `negative_prompt_embeds` instead. Ignored when not using guidance (`guidance_scale < 1`).
            eta (`float`, *optional*, defaults to 0.0):
                Corresponds to parameter eta (η) from the [DDIM](https://arxiv.org/abs/2010.02502) paper. Only applies
                to the [`~schedulers.DDIMScheduler`], and is ignored in other schedulers.
            generator (`torch.Generator` or `List[torch.Generator]`, *optional*):
                A [`torch.Generator`](https://pytorch.org/docs/stable/generated/torch.Generator.html) to make
                generation deterministic.
            latents (`torch.Tensor`, *optional*):
                Pre-generated noisy latents sampled from a Gaussian distribution, to be used as inputs for video
                generation. Can be used to tweak the same generation with different prompts. If not provided, a latents
                tensor is generated by sampling using the supplied random `generator`. Latents should be of shape
                `(batch_size, num_channel, num_frames, height, width)`.
            prompt_embeds (`torch.Tensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs (prompt weighting). If not
                provided, text embeddings are generated from the `prompt` input argument.
            negative_prompt_embeds (`torch.Tensor`, *optional*):
                Pre-generated negative text embeddings. Can be used to easily tweak text inputs (prompt weighting). If
                not provided, `negative_prompt_embeds` are generated from the `negative_prompt` input argument.
            ip_adapter_image: (`PipelineImageInput`, *optional*):
                Optional image input to work with IP Adapters.
            ip_adapter_image_embeds (`List[torch.Tensor]`, *optional*):
                Pre-generated image embeddings for IP-Adapter. It should be a list of length same as number of
                IP-adapters. Each element should be a tensor of shape `(batch_size, num_images, emb_dim)`. It should
                contain the negative image embedding if `do_classifier_free_guidance` is set to `True`. If not
                provided, embeddings are computed from the `ip_adapter_image` input argument.
            motion_scale: (`int`, *optional*, defaults to 0):
                Parameter that controls the amount and type of motion that is added to the image. Increasing the value
                increases the amount of motion, while specific ranges of values control the type of motion that is
                added. Must be between 0 and 8. Set between 0-2 to only increase the amount of motion. Set between 3-5
                to create looping motion. Set between 6-8 to perform motion with image style transfer.
            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generated video. Choose between `torch.Tensor`, `PIL.Image` or `np.array`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~pipelines.text_to_video_synthesis.TextToVideoSDPipelineOutput`] instead
                of a plain tuple.
            cross_attention_kwargs (`dict`, *optional*):
                A kwargs dictionary that if specified is passed along to the [`AttentionProcessor`] as defined in
                [`self.processor`](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py).
            clip_skip (`int`, *optional*):
                Number of layers to be skipped from CLIP while computing the prompt embeddings. A value of 1 means that
                the output of the pre-final layer will be used for computing the prompt embeddings.
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
            [`~pipelines.pia.pipeline_pia.PIAPipelineOutput`] or `tuple`:
                If `return_dict` is `True`, [`~pipelines.pia.pipeline_pia.PIAPipelineOutput`] is returned, otherwise a
                `tuple` is returned where the first element is a list with the generated frames.
        """
    height = height or self.unet.config.sample_size * self.vae_scale_factor
    width = width or self.unet.config.sample_size * self.vae_scale_factor
    num_videos_per_prompt = 1
    self.check_inputs(prompt, height, width, negative_prompt, prompt_embeds,
        negative_prompt_embeds, ip_adapter_image, ip_adapter_image_embeds,
        callback_on_step_end_tensor_inputs)
    self._guidance_scale = guidance_scale
    self._clip_skip = clip_skip
    self._cross_attention_kwargs = cross_attention_kwargs
    if prompt is not None and isinstance(prompt, str):
        batch_size = 1
    elif prompt is not None and isinstance(prompt, list):
        batch_size = len(prompt)
    else:
        batch_size = prompt_embeds.shape[0]
    device = self._execution_device
    text_encoder_lora_scale = self.cross_attention_kwargs.get('scale', None
        ) if self.cross_attention_kwargs is not None else None
    prompt_embeds, negative_prompt_embeds = self.encode_prompt(prompt,
        device, num_videos_per_prompt, self.do_classifier_free_guidance,
        negative_prompt, prompt_embeds=prompt_embeds,
        negative_prompt_embeds=negative_prompt_embeds, lora_scale=
        text_encoder_lora_scale, clip_skip=self.clip_skip)
    if self.do_classifier_free_guidance:
        prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])
    if ip_adapter_image is not None or ip_adapter_image_embeds is not None:
        image_embeds = self.prepare_ip_adapter_image_embeds(ip_adapter_image,
            ip_adapter_image_embeds, device, batch_size *
            num_videos_per_prompt, self.do_classifier_free_guidance)
    self.scheduler.set_timesteps(num_inference_steps, device=device)
    timesteps, num_inference_steps = self.get_timesteps(num_inference_steps,
        strength, device)
    latent_timestep = timesteps[:1].repeat(batch_size * num_videos_per_prompt)
    self._num_timesteps = len(timesteps)
    latents = self.prepare_latents(batch_size * num_videos_per_prompt, 4,
        num_frames, height, width, prompt_embeds.dtype, device, generator,
        latents=latents)
    mask, masked_image = self.prepare_masked_condition(image, batch_size *
        num_videos_per_prompt, 4, num_frames=num_frames, height=height,
        width=width, dtype=self.unet.dtype, device=device, generator=
        generator, motion_scale=motion_scale)
    if strength < 1.0:
        noise = randn_tensor(latents.shape, generator=generator, device=
            device, dtype=latents.dtype)
        latents = self.scheduler.add_noise(masked_image[0], noise,
            latent_timestep)
    extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)
    added_cond_kwargs = ({'image_embeds': image_embeds} if ip_adapter_image
         is not None or ip_adapter_image_embeds is not None else None)
    num_free_init_iters = (self._free_init_num_iters if self.
        free_init_enabled else 1)
    for free_init_iter in range(num_free_init_iters):
        if self.free_init_enabled:
            latents, timesteps = self._apply_free_init(latents,
                free_init_iter, num_inference_steps, device, latents.dtype,
                generator)
        self._num_timesteps = len(timesteps)
        num_warmup_steps = len(timesteps
            ) - num_inference_steps * self.scheduler.order
        with self.progress_bar(total=self._num_timesteps) as progress_bar:
            for i, t in enumerate(timesteps):
                latent_model_input = torch.cat([latents] * 2
                    ) if self.do_classifier_free_guidance else latents
                latent_model_input = self.scheduler.scale_model_input(
                    latent_model_input, t)
                latent_model_input = torch.cat([latent_model_input, mask,
                    masked_image], dim=1)
                noise_pred = self.unet(latent_model_input, t,
                    encoder_hidden_states=prompt_embeds,
                    cross_attention_kwargs=cross_attention_kwargs,
                    added_cond_kwargs=added_cond_kwargs).sample
                if self.do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + guidance_scale * (
                        noise_pred_text - noise_pred_uncond)
                latents = self.scheduler.step(noise_pred, t, latents, **
                    extra_step_kwargs).prev_sample
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
                if i == len(timesteps) - 1 or i + 1 > num_warmup_steps and (i +
                    1) % self.scheduler.order == 0:
                    progress_bar.update()
    if output_type == 'latent':
        video = latents
    else:
        video_tensor = self.decode_latents(latents)
        video = self.video_processor.postprocess_video(video=video_tensor,
            output_type=output_type)
    self.maybe_free_model_hooks()
    if not return_dict:
        return video,
    return PIAPipelineOutput(frames=video)

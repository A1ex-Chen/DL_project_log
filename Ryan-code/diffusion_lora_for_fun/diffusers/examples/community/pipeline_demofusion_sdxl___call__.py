@torch.no_grad()
@replace_example_docstring(EXAMPLE_DOC_STRING)
def __call__(self, prompt: Union[str, List[str]]=None, prompt_2: Optional[
    Union[str, List[str]]]=None, height: Optional[int]=None, width:
    Optional[int]=None, num_inference_steps: int=50, denoising_end:
    Optional[float]=None, guidance_scale: float=5.0, negative_prompt:
    Optional[Union[str, List[str]]]=None, negative_prompt_2: Optional[Union
    [str, List[str]]]=None, num_images_per_prompt: Optional[int]=1, eta:
    float=0.0, generator: Optional[Union[torch.Generator, List[torch.
    Generator]]]=None, latents: Optional[torch.Tensor]=None, prompt_embeds:
    Optional[torch.Tensor]=None, negative_prompt_embeds: Optional[torch.
    Tensor]=None, pooled_prompt_embeds: Optional[torch.Tensor]=None,
    negative_pooled_prompt_embeds: Optional[torch.Tensor]=None, output_type:
    Optional[str]='pil', return_dict: bool=False, callback: Optional[
    Callable[[int, int, torch.Tensor], None]]=None, callback_steps: int=1,
    cross_attention_kwargs: Optional[Dict[str, Any]]=None, guidance_rescale:
    float=0.0, original_size: Optional[Tuple[int, int]]=None,
    crops_coords_top_left: Tuple[int, int]=(0, 0), target_size: Optional[
    Tuple[int, int]]=None, negative_original_size: Optional[Tuple[int, int]
    ]=None, negative_crops_coords_top_left: Tuple[int, int]=(0, 0),
    negative_target_size: Optional[Tuple[int, int]]=None, view_batch_size:
    int=16, multi_decoder: bool=True, stride: Optional[int]=64,
    cosine_scale_1: Optional[float]=3.0, cosine_scale_2: Optional[float]=
    1.0, cosine_scale_3: Optional[float]=1.0, sigma: Optional[float]=0.8,
    show_image: bool=False):
    """
        Function invoked when calling the pipeline for generation.

        Args:
            prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts to guide the image generation. If not defined, one has to pass `prompt_embeds`.
                instead.
            prompt_2 (`str` or `List[str]`, *optional*):
                The prompt or prompts to be sent to the `tokenizer_2` and `text_encoder_2`. If not defined, `prompt` is
                used in both text-encoders
            height (`int`, *optional*, defaults to self.unet.config.sample_size * self.vae_scale_factor):
                The height in pixels of the generated image. This is set to 1024 by default for the best results.
                Anything below 512 pixels won't work well for
                [stabilityai/stable-diffusion-xl-base-1.0](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0)
                and checkpoints that are not specifically fine-tuned on low resolutions.
            width (`int`, *optional*, defaults to self.unet.config.sample_size * self.vae_scale_factor):
                The width in pixels of the generated image. This is set to 1024 by default for the best results.
                Anything below 512 pixels won't work well for
                [stabilityai/stable-diffusion-xl-base-1.0](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0)
                and checkpoints that are not specifically fine-tuned on low resolutions.
            num_inference_steps (`int`, *optional*, defaults to 50):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            denoising_end (`float`, *optional*):
                When specified, determines the fraction (between 0.0 and 1.0) of the total denoising process to be
                completed before it is intentionally prematurely terminated. As a result, the returned sample will
                still retain a substantial amount of noise as determined by the discrete timesteps selected by the
                scheduler. The denoising_end parameter should ideally be utilized when this pipeline forms a part of a
                "Mixture of Denoisers" multi-pipeline setup, as elaborated in [**Refining the Image
                Output**](https://huggingface.co/docs/diffusers/api/pipelines/stable_diffusion/stable_diffusion_xl#refining-the-image-output)
            guidance_scale (`float`, *optional*, defaults to 5.0):
                Guidance scale as defined in [Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598).
                `guidance_scale` is defined as `w` of equation 2. of [Imagen
                Paper](https://arxiv.org/pdf/2205.11487.pdf). Guidance scale is enabled by setting `guidance_scale >
                1`. Higher guidance scale encourages to generate images that are closely linked to the text `prompt`,
                usually at the expense of lower image quality.
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the image generation. If not defined, one has to pass
                `negative_prompt_embeds` instead. Ignored when not using guidance (i.e., ignored if `guidance_scale` is
                less than `1`).
            negative_prompt_2 (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the image generation to be sent to `tokenizer_2` and
                `text_encoder_2`. If not defined, `negative_prompt` is used in both text-encoders
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
            pooled_prompt_embeds (`torch.Tensor`, *optional*):
                Pre-generated pooled text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting.
                If not provided, pooled text embeddings will be generated from `prompt` input argument.
            negative_pooled_prompt_embeds (`torch.Tensor`, *optional*):
                Pre-generated negative pooled text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt
                weighting. If not provided, pooled negative_prompt_embeds will be generated from `negative_prompt`
                input argument.
            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generate image. Choose between
                [PIL](https://pillow.readthedocs.io/en/stable/): `PIL.Image.Image` or `np.array`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~pipelines.stable_diffusion_xl.StableDiffusionXLPipelineOutput`] instead
                of a plain tuple.
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
            guidance_rescale (`float`, *optional*, defaults to 0.7):
                Guidance rescale factor proposed by [Common Diffusion Noise Schedules and Sample Steps are
                Flawed](https://arxiv.org/pdf/2305.08891.pdf) `guidance_scale` is defined as `φ` in equation 16. of
                [Common Diffusion Noise Schedules and Sample Steps are Flawed](https://arxiv.org/pdf/2305.08891.pdf).
                Guidance rescale factor should fix overexposure when using zero terminal SNR.
            original_size (`Tuple[int]`, *optional*, defaults to (1024, 1024)):
                If `original_size` is not the same as `target_size` the image will appear to be down- or upsampled.
                `original_size` defaults to `(width, height)` if not specified. Part of SDXL's micro-conditioning as
                explained in section 2.2 of
                [https://huggingface.co/papers/2307.01952](https://huggingface.co/papers/2307.01952).
            crops_coords_top_left (`Tuple[int]`, *optional*, defaults to (0, 0)):
                `crops_coords_top_left` can be used to generate an image that appears to be "cropped" from the position
                `crops_coords_top_left` downwards. Favorable, well-centered images are usually achieved by setting
                `crops_coords_top_left` to (0, 0). Part of SDXL's micro-conditioning as explained in section 2.2 of
                [https://huggingface.co/papers/2307.01952](https://huggingface.co/papers/2307.01952).
            target_size (`Tuple[int]`, *optional*, defaults to (1024, 1024)):
                For most cases, `target_size` should be set to the desired height and width of the generated image. If
                not specified it will default to `(width, height)`. Part of SDXL's micro-conditioning as explained in
                section 2.2 of [https://huggingface.co/papers/2307.01952](https://huggingface.co/papers/2307.01952).
            negative_original_size (`Tuple[int]`, *optional*, defaults to (1024, 1024)):
                To negatively condition the generation process based on a specific image resolution. Part of SDXL's
                micro-conditioning as explained in section 2.2 of
                [https://huggingface.co/papers/2307.01952](https://huggingface.co/papers/2307.01952). For more
                information, refer to this issue thread: https://github.com/huggingface/diffusers/issues/4208.
            negative_crops_coords_top_left (`Tuple[int]`, *optional*, defaults to (0, 0)):
                To negatively condition the generation process based on a specific crop coordinates. Part of SDXL's
                micro-conditioning as explained in section 2.2 of
                [https://huggingface.co/papers/2307.01952](https://huggingface.co/papers/2307.01952). For more
                information, refer to this issue thread: https://github.com/huggingface/diffusers/issues/4208.
            negative_target_size (`Tuple[int]`, *optional*, defaults to (1024, 1024)):
                To negatively condition the generation process based on a target image resolution. It should be as same
                as the `target_size` for most cases. Part of SDXL's micro-conditioning as explained in section 2.2 of
                [https://huggingface.co/papers/2307.01952](https://huggingface.co/papers/2307.01952). For more
                information, refer to this issue thread: https://github.com/huggingface/diffusers/issues/4208.
            ################### DemoFusion specific parameters ####################
            view_batch_size (`int`, defaults to 16):
                The batch size for multiple denoising paths. Typically, a larger batch size can result in higher
                efficiency but comes with increased GPU memory requirements.
            multi_decoder (`bool`, defaults to True):
                Determine whether to use a tiled decoder. Generally, when the resolution exceeds 3072x3072,
                a tiled decoder becomes necessary.
            stride (`int`, defaults to 64):
                The stride of moving local patches. A smaller stride is better for alleviating seam issues,
                but it also introduces additional computational overhead and inference time.
            cosine_scale_1 (`float`, defaults to 3):
                Control the strength of skip-residual. For specific impacts, please refer to Appendix C
                in the DemoFusion paper.
            cosine_scale_2 (`float`, defaults to 1):
                Control the strength of dilated sampling. For specific impacts, please refer to Appendix C
                in the DemoFusion paper.
            cosine_scale_3 (`float`, defaults to 1):
                Control the strength of the gaussion filter. For specific impacts, please refer to Appendix C
                in the DemoFusion paper.
            sigma (`float`, defaults to 1):
                The standerd value of the gaussian filter.
            show_image (`bool`, defaults to False):
                Determine whether to show intermediate results during generation.

        Examples:

        Returns:
            a `list` with the generated images at each phase.
        """
    height = height or self.default_sample_size * self.vae_scale_factor
    width = width or self.default_sample_size * self.vae_scale_factor
    x1_size = self.default_sample_size * self.vae_scale_factor
    height_scale = height / x1_size
    width_scale = width / x1_size
    scale_num = int(max(height_scale, width_scale))
    aspect_ratio = min(height_scale, width_scale) / max(height_scale,
        width_scale)
    original_size = original_size or (height, width)
    target_size = target_size or (height, width)
    self.check_inputs(prompt, prompt_2, height, width, callback_steps,
        negative_prompt, negative_prompt_2, prompt_embeds,
        negative_prompt_embeds, pooled_prompt_embeds,
        negative_pooled_prompt_embeds, num_images_per_prompt)
    if prompt is not None and isinstance(prompt, str):
        batch_size = 1
    elif prompt is not None and isinstance(prompt, list):
        batch_size = len(prompt)
    else:
        batch_size = prompt_embeds.shape[0]
    device = self._execution_device
    do_classifier_free_guidance = guidance_scale > 1.0
    text_encoder_lora_scale = cross_attention_kwargs.get('scale', None
        ) if cross_attention_kwargs is not None else None
    (prompt_embeds, negative_prompt_embeds, pooled_prompt_embeds,
        negative_pooled_prompt_embeds) = (self.encode_prompt(prompt=prompt,
        prompt_2=prompt_2, device=device, num_images_per_prompt=
        num_images_per_prompt, do_classifier_free_guidance=
        do_classifier_free_guidance, negative_prompt=negative_prompt,
        negative_prompt_2=negative_prompt_2, prompt_embeds=prompt_embeds,
        negative_prompt_embeds=negative_prompt_embeds, pooled_prompt_embeds
        =pooled_prompt_embeds, negative_pooled_prompt_embeds=
        negative_pooled_prompt_embeds, lora_scale=text_encoder_lora_scale))
    self.scheduler.set_timesteps(num_inference_steps, device=device)
    timesteps = self.scheduler.timesteps
    num_channels_latents = self.unet.config.in_channels
    latents = self.prepare_latents(batch_size * num_images_per_prompt,
        num_channels_latents, height // scale_num, width // scale_num,
        prompt_embeds.dtype, device, generator, latents)
    extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)
    add_text_embeds = pooled_prompt_embeds
    add_time_ids = self._get_add_time_ids(original_size,
        crops_coords_top_left, target_size, dtype=prompt_embeds.dtype)
    if negative_original_size is not None and negative_target_size is not None:
        negative_add_time_ids = self._get_add_time_ids(negative_original_size,
            negative_crops_coords_top_left, negative_target_size, dtype=
            prompt_embeds.dtype)
    else:
        negative_add_time_ids = add_time_ids
    if do_classifier_free_guidance:
        prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds],
            dim=0)
        add_text_embeds = torch.cat([negative_pooled_prompt_embeds,
            add_text_embeds], dim=0)
        add_time_ids = torch.cat([negative_add_time_ids, add_time_ids], dim=0)
    prompt_embeds = prompt_embeds.to(device)
    add_text_embeds = add_text_embeds.to(device)
    add_time_ids = add_time_ids.to(device).repeat(batch_size *
        num_images_per_prompt, 1)
    num_warmup_steps = max(len(timesteps) - num_inference_steps * self.
        scheduler.order, 0)
    if denoising_end is not None and isinstance(denoising_end, float
        ) and denoising_end > 0 and denoising_end < 1:
        discrete_timestep_cutoff = int(round(self.scheduler.config.
            num_train_timesteps - denoising_end * self.scheduler.config.
            num_train_timesteps))
        num_inference_steps = len(list(filter(lambda ts: ts >=
            discrete_timestep_cutoff, timesteps)))
        timesteps = timesteps[:num_inference_steps]
    output_images = []
    print('### Phase 1 Denoising ###')
    with self.progress_bar(total=num_inference_steps) as progress_bar:
        for i, t in enumerate(timesteps):
            latents_for_view = latents
            latent_model_input = latents.repeat_interleave(2, dim=0
                ) if do_classifier_free_guidance else latents
            latent_model_input = self.scheduler.scale_model_input(
                latent_model_input, t)
            added_cond_kwargs = {'text_embeds': add_text_embeds, 'time_ids':
                add_time_ids}
            noise_pred = self.unet(latent_model_input, t,
                encoder_hidden_states=prompt_embeds, cross_attention_kwargs
                =cross_attention_kwargs, added_cond_kwargs=
                added_cond_kwargs, return_dict=False)[0]
            if do_classifier_free_guidance:
                noise_pred_uncond, noise_pred_text = noise_pred[::2
                    ], noise_pred[1::2]
                noise_pred = noise_pred_uncond + guidance_scale * (
                    noise_pred_text - noise_pred_uncond)
            if do_classifier_free_guidance and guidance_rescale > 0.0:
                noise_pred = rescale_noise_cfg(noise_pred, noise_pred_text,
                    guidance_rescale=guidance_rescale)
            latents = self.scheduler.step(noise_pred, t, latents, **
                extra_step_kwargs, return_dict=False)[0]
            if i == len(timesteps) - 1 or i + 1 > num_warmup_steps and (i + 1
                ) % self.scheduler.order == 0:
                progress_bar.update()
                if callback is not None and i % callback_steps == 0:
                    step_idx = i // getattr(self.scheduler, 'order', 1)
                    callback(step_idx, t, latents)
        anchor_mean = latents.mean()
        anchor_std = latents.std()
        if not output_type == 'latent':
            needs_upcasting = (self.vae.dtype == torch.float16 and self.vae
                .config.force_upcast)
            if needs_upcasting:
                self.upcast_vae()
                latents = latents.to(next(iter(self.vae.post_quant_conv.
                    parameters())).dtype)
            print('### Phase 1 Decoding ###')
            image = self.vae.decode(latents / self.vae.config.
                scaling_factor, return_dict=False)[0]
            if needs_upcasting:
                self.vae.to(dtype=torch.float16)
        image = self.image_processor.postprocess(image, output_type=output_type
            )
        if show_image:
            plt.figure(figsize=(10, 10))
            plt.imshow(image[0])
            plt.axis('off')
            plt.show()
        output_images.append(image[0])
    for current_scale_num in range(2, scale_num + 1):
        print('### Phase {} Denoising ###'.format(current_scale_num))
        current_height = (self.unet.config.sample_size * self.
            vae_scale_factor * current_scale_num)
        current_width = (self.unet.config.sample_size * self.
            vae_scale_factor * current_scale_num)
        if height > width:
            current_width = int(current_width * aspect_ratio)
        else:
            current_height = int(current_height * aspect_ratio)
        latents = F.interpolate(latents, size=(int(current_height / self.
            vae_scale_factor), int(current_width / self.vae_scale_factor)),
            mode='bicubic')
        noise_latents = []
        noise = torch.randn_like(latents)
        for timestep in timesteps:
            noise_latent = self.scheduler.add_noise(latents, noise,
                timestep.unsqueeze(0))
            noise_latents.append(noise_latent)
        latents = noise_latents[0]
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                count = torch.zeros_like(latents)
                value = torch.zeros_like(latents)
                cosine_factor = 0.5 * (1 + torch.cos(torch.pi * (self.
                    scheduler.config.num_train_timesteps - t) / self.
                    scheduler.config.num_train_timesteps)).cpu()
                c1 = cosine_factor ** cosine_scale_1
                latents = latents * (1 - c1) + noise_latents[i] * c1
                views = self.get_views(current_height, current_width,
                    stride=stride, window_size=self.unet.config.sample_size,
                    random_jitter=True)
                views_batch = [views[i:i + view_batch_size] for i in range(
                    0, len(views), view_batch_size)]
                jitter_range = (self.unet.config.sample_size - stride) // 4
                latents_ = F.pad(latents, (jitter_range, jitter_range,
                    jitter_range, jitter_range), 'constant', 0)
                count_local = torch.zeros_like(latents_)
                value_local = torch.zeros_like(latents_)
                for j, batch_view in enumerate(views_batch):
                    vb_size = len(batch_view)
                    latents_for_view = torch.cat([latents_[:, :, h_start:
                        h_end, w_start:w_end] for h_start, h_end, w_start,
                        w_end in batch_view])
                    latent_model_input = latents_for_view
                    latent_model_input = latent_model_input.repeat_interleave(
                        2, dim=0
                        ) if do_classifier_free_guidance else latent_model_input
                    latent_model_input = self.scheduler.scale_model_input(
                        latent_model_input, t)
                    prompt_embeds_input = torch.cat([prompt_embeds] * vb_size)
                    add_text_embeds_input = torch.cat([add_text_embeds] *
                        vb_size)
                    add_time_ids_input = []
                    for h_start, h_end, w_start, w_end in batch_view:
                        add_time_ids_ = add_time_ids.clone()
                        add_time_ids_[:, 2] = h_start * self.vae_scale_factor
                        add_time_ids_[:, 3] = w_start * self.vae_scale_factor
                        add_time_ids_input.append(add_time_ids_)
                    add_time_ids_input = torch.cat(add_time_ids_input)
                    added_cond_kwargs = {'text_embeds':
                        add_text_embeds_input, 'time_ids': add_time_ids_input}
                    noise_pred = self.unet(latent_model_input, t,
                        encoder_hidden_states=prompt_embeds_input,
                        cross_attention_kwargs=cross_attention_kwargs,
                        added_cond_kwargs=added_cond_kwargs, return_dict=False
                        )[0]
                    if do_classifier_free_guidance:
                        noise_pred_uncond, noise_pred_text = noise_pred[::2
                            ], noise_pred[1::2]
                        noise_pred = noise_pred_uncond + guidance_scale * (
                            noise_pred_text - noise_pred_uncond)
                    if do_classifier_free_guidance and guidance_rescale > 0.0:
                        noise_pred = rescale_noise_cfg(noise_pred,
                            noise_pred_text, guidance_rescale=guidance_rescale)
                    self.scheduler._init_step_index(t)
                    latents_denoised_batch = self.scheduler.step(noise_pred,
                        t, latents_for_view, **extra_step_kwargs,
                        return_dict=False)[0]
                    for latents_view_denoised, (h_start, h_end, w_start, w_end
                        ) in zip(latents_denoised_batch.chunk(vb_size),
                        batch_view):
                        value_local[:, :, h_start:h_end, w_start:w_end
                            ] += latents_view_denoised
                        count_local[:, :, h_start:h_end, w_start:w_end] += 1
                value_local = value_local[:, :, jitter_range:jitter_range +
                    current_height // self.vae_scale_factor, jitter_range:
                    jitter_range + current_width // self.vae_scale_factor]
                count_local = count_local[:, :, jitter_range:jitter_range +
                    current_height // self.vae_scale_factor, jitter_range:
                    jitter_range + current_width // self.vae_scale_factor]
                c2 = cosine_factor ** cosine_scale_2
                value += value_local / count_local * (1 - c2)
                count += torch.ones_like(value_local) * (1 - c2)
                views = [[h, w] for h in range(current_scale_num) for w in
                    range(current_scale_num)]
                views_batch = [views[i:i + view_batch_size] for i in range(
                    0, len(views), view_batch_size)]
                h_pad = (current_scale_num - latents.size(2) %
                    current_scale_num) % current_scale_num
                w_pad = (current_scale_num - latents.size(3) %
                    current_scale_num) % current_scale_num
                latents_ = F.pad(latents, (w_pad, 0, h_pad, 0), 'constant', 0)
                count_global = torch.zeros_like(latents_)
                value_global = torch.zeros_like(latents_)
                c3 = 0.99 * cosine_factor ** cosine_scale_3 + 0.01
                std_, mean_ = latents_.std(), latents_.mean()
                latents_gaussian = gaussian_filter(latents_, kernel_size=2 *
                    current_scale_num - 1, sigma=sigma * c3)
                latents_gaussian = (latents_gaussian - latents_gaussian.mean()
                    ) / latents_gaussian.std() * std_ + mean_
                for j, batch_view in enumerate(views_batch):
                    latents_for_view = torch.cat([latents_[:, :, h::
                        current_scale_num, w::current_scale_num] for h, w in
                        batch_view])
                    latents_for_view_gaussian = torch.cat([latents_gaussian
                        [:, :, h::current_scale_num, w::current_scale_num] for
                        h, w in batch_view])
                    vb_size = latents_for_view.size(0)
                    latent_model_input = latents_for_view_gaussian
                    latent_model_input = latent_model_input.repeat_interleave(
                        2, dim=0
                        ) if do_classifier_free_guidance else latent_model_input
                    latent_model_input = self.scheduler.scale_model_input(
                        latent_model_input, t)
                    prompt_embeds_input = torch.cat([prompt_embeds] * vb_size)
                    add_text_embeds_input = torch.cat([add_text_embeds] *
                        vb_size)
                    add_time_ids_input = torch.cat([add_time_ids] * vb_size)
                    added_cond_kwargs = {'text_embeds':
                        add_text_embeds_input, 'time_ids': add_time_ids_input}
                    noise_pred = self.unet(latent_model_input, t,
                        encoder_hidden_states=prompt_embeds_input,
                        cross_attention_kwargs=cross_attention_kwargs,
                        added_cond_kwargs=added_cond_kwargs, return_dict=False
                        )[0]
                    if do_classifier_free_guidance:
                        noise_pred_uncond, noise_pred_text = noise_pred[::2
                            ], noise_pred[1::2]
                        noise_pred = noise_pred_uncond + guidance_scale * (
                            noise_pred_text - noise_pred_uncond)
                    if do_classifier_free_guidance and guidance_rescale > 0.0:
                        noise_pred = rescale_noise_cfg(noise_pred,
                            noise_pred_text, guidance_rescale=guidance_rescale)
                    self.scheduler._init_step_index(t)
                    latents_denoised_batch = self.scheduler.step(noise_pred,
                        t, latents_for_view, **extra_step_kwargs,
                        return_dict=False)[0]
                    for latents_view_denoised, (h, w) in zip(
                        latents_denoised_batch.chunk(vb_size), batch_view):
                        value_global[:, :, h::current_scale_num, w::
                            current_scale_num] += latents_view_denoised
                        count_global[:, :, h::current_scale_num, w::
                            current_scale_num] += 1
                c2 = cosine_factor ** cosine_scale_2
                value_global = value_global[:, :, h_pad:, w_pad:]
                value += value_global * c2
                count += torch.ones_like(value_global) * c2
                latents = torch.where(count > 0, value / count, value)
                if i == len(timesteps) - 1 or i + 1 > num_warmup_steps and (i +
                    1) % self.scheduler.order == 0:
                    progress_bar.update()
                    if callback is not None and i % callback_steps == 0:
                        step_idx = i // getattr(self.scheduler, 'order', 1)
                        callback(step_idx, t, latents)
            latents = (latents - latents.mean()) / latents.std(
                ) * anchor_std + anchor_mean
            if not output_type == 'latent':
                needs_upcasting = (self.vae.dtype == torch.float16 and self
                    .vae.config.force_upcast)
                if needs_upcasting:
                    self.upcast_vae()
                    latents = latents.to(next(iter(self.vae.post_quant_conv
                        .parameters())).dtype)
                print('### Phase {} Decoding ###'.format(current_scale_num))
                if multi_decoder:
                    image = self.tiled_decode(latents, current_height,
                        current_width)
                else:
                    image = self.vae.decode(latents / self.vae.config.
                        scaling_factor, return_dict=False)[0]
                if needs_upcasting:
                    self.vae.to(dtype=torch.float16)
            else:
                image = latents
            if not output_type == 'latent':
                image = self.image_processor.postprocess(image, output_type
                    =output_type)
                if show_image:
                    plt.figure(figsize=(10, 10))
                    plt.imshow(image[0])
                    plt.axis('off')
                    plt.show()
                output_images.append(image[0])
    self.maybe_free_model_hooks()
    return output_images

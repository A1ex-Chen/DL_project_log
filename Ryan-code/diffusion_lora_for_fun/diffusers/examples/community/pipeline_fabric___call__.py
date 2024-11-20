@torch.no_grad()
@replace_example_docstring(EXAMPLE_DOC_STRING)
def __call__(self, prompt: Optional[Union[str, List[str]]]='',
    negative_prompt: Optional[Union[str, List[str]]]=
    'lowres, bad anatomy, bad hands, cropped, worst quality', liked:
    Optional[Union[List[str], List[Image.Image]]]=[], disliked: Optional[
    Union[List[str], List[Image.Image]]]=[], generator: Optional[Union[
    torch.Generator, List[torch.Generator]]]=None, height: int=512, width:
    int=512, return_dict: bool=True, num_images: int=4, guidance_scale:
    float=7.0, num_inference_steps: int=20, output_type: Optional[str]=
    'pil', feedback_start_ratio: float=0.33, feedback_end_ratio: float=0.66,
    min_weight: float=0.05, max_weight: float=0.8, neg_scale: float=0.5,
    pos_bottleneck_scale: float=1.0, neg_bottleneck_scale: float=1.0,
    latents: Optional[torch.Tensor]=None):
    """
        The call function to the pipeline for generation. Generate a trajectory of images with binary feedback. The
        feedback can be given as a list of liked and disliked images.

        Args:
            prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts to guide image generation. If not defined, you need to pass `prompt_embeds`
                instead.
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts to guide what to not include in image generation. If not defined, you need to
                pass `negative_prompt_embeds` instead. Ignored when not using guidance (`guidance_scale < 1`).
            liked (`List[Image.Image]` or `List[str]`, *optional*):
                Encourages images with liked features.
            disliked (`List[Image.Image]` or `List[str]`, *optional*):
                Discourages images with disliked features.
            generator (`torch.Generator` or `List[torch.Generator]` or `int`, *optional*):
                A [`torch.Generator`](https://pytorch.org/docs/stable/generated/torch.Generator.html) or an `int` to
                make generation deterministic.
            height (`int`, *optional*, defaults to 512):
                Height of the generated image.
            width (`int`, *optional*, defaults to 512):
                Width of the generated image.
            num_images (`int`, *optional*, defaults to 4):
                The number of images to generate per prompt.
            guidance_scale (`float`, *optional*, defaults to 7.0):
                A higher guidance scale value encourages the model to generate images closely linked to the text
                `prompt` at the expense of lower image quality. Guidance scale is enabled when `guidance_scale > 1`.
            num_inference_steps (`int`, *optional*, defaults to 20):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generated image. Choose between `PIL.Image` or `np.array`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] instead of a
                plain tuple.
            feedback_start_ratio (`float`, *optional*, defaults to `.33`):
                Start point for providing feedback (between 0 and 1).
            feedback_end_ratio (`float`, *optional*, defaults to `.66`):
                End point for providing feedback (between 0 and 1).
            min_weight (`float`, *optional*, defaults to `.05`):
                Minimum weight for feedback.
            max_weight (`float`, *optional*, defults tp `1.0`):
                Maximum weight for feedback.
            neg_scale (`float`, *optional*, defaults to `.5`):
                Scale factor for negative feedback.

        Examples:

        Returns:
            [`~pipelines.fabric.FabricPipelineOutput`] or `tuple`:
                If `return_dict` is `True`, [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] is returned,
                otherwise a `tuple` is returned where the first element is a list with the generated images and the
                second element is a list of `bool`s indicating whether the corresponding generated image contains
                "not-safe-for-work" (nsfw) content.

        """
    self.check_inputs(prompt, negative_prompt, liked, disliked)
    device = self._execution_device
    dtype = self.unet.dtype
    if isinstance(prompt, str) and prompt is not None:
        batch_size = 1
    elif isinstance(prompt, list) and prompt is not None:
        batch_size = len(prompt)
    else:
        raise ValueError(
            f'`prompt` has to be of type `str` or `list` but is {type(prompt)}'
            )
    if isinstance(negative_prompt, str):
        negative_prompt = negative_prompt
    elif isinstance(negative_prompt, list):
        negative_prompt = negative_prompt
    else:
        assert len(negative_prompt) == batch_size
    shape = (batch_size * num_images, self.unet.config.in_channels, height //
        self.vae_scale_factor, width // self.vae_scale_factor)
    latent_noise = randn_tensor(shape, device=device, dtype=dtype,
        generator=generator)
    positive_latents = self.preprocess_feedback_images(liked, self.vae, (
        height, width), device, dtype, generator) if liked and len(liked
        ) > 0 else torch.tensor([], device=device, dtype=dtype)
    negative_latents = self.preprocess_feedback_images(disliked, self.vae,
        (height, width), device, dtype, generator) if disliked and len(disliked
        ) > 0 else torch.tensor([], device=device, dtype=dtype)
    do_classifier_free_guidance = guidance_scale > 0.1
    prompt_neg_embs, prompt_pos_embs = self._encode_prompt(prompt, device,
        num_images, do_classifier_free_guidance, negative_prompt).split([
        num_images * batch_size, num_images * batch_size])
    batched_prompt_embd = torch.cat([prompt_pos_embs, prompt_neg_embs], dim=0)
    null_tokens = self.tokenizer([''], return_tensors='pt', max_length=self
        .tokenizer.model_max_length, padding='max_length', truncation=True)
    if hasattr(self.text_encoder.config, 'use_attention_mask'
        ) and self.text_encoder.config.use_attention_mask:
        attention_mask = null_tokens.attention_mask.to(device)
    else:
        attention_mask = None
    null_prompt_emb = self.text_encoder(input_ids=null_tokens.input_ids.to(
        device), attention_mask=attention_mask).last_hidden_state
    null_prompt_emb = null_prompt_emb.to(device=device, dtype=dtype)
    self.scheduler.set_timesteps(num_inference_steps, device=device)
    timesteps = self.scheduler.timesteps
    latent_noise = latent_noise * self.scheduler.init_noise_sigma
    num_warmup_steps = len(timesteps
        ) - num_inference_steps * self.scheduler.order
    ref_start_idx = round(len(timesteps) * feedback_start_ratio)
    ref_end_idx = round(len(timesteps) * feedback_end_ratio)
    with self.progress_bar(total=num_inference_steps) as pbar:
        for i, t in enumerate(timesteps):
            sigma = self.scheduler.sigma_t[t] if hasattr(self.scheduler,
                'sigma_t') else 0
            if hasattr(self.scheduler, 'sigmas'):
                sigma = self.scheduler.sigmas[i]
            alpha_hat = 1 / (sigma ** 2 + 1)
            z_single = self.scheduler.scale_model_input(latent_noise, t)
            z_all = torch.cat([z_single] * 2, dim=0)
            z_ref = torch.cat([positive_latents, negative_latents], dim=0)
            if i >= ref_start_idx and i <= ref_end_idx:
                weight_factor = max_weight
            else:
                weight_factor = min_weight
            pos_ws = weight_factor, weight_factor * pos_bottleneck_scale
            neg_ws = (weight_factor * neg_scale, weight_factor * neg_scale *
                neg_bottleneck_scale)
            if z_ref.size(0) > 0 and weight_factor > 0:
                noise = torch.randn_like(z_ref)
                if isinstance(self.scheduler, EulerAncestralDiscreteScheduler):
                    z_ref_noised = (alpha_hat ** 0.5 * z_ref + (1 -
                        alpha_hat) ** 0.5 * noise).type(dtype)
                else:
                    z_ref_noised = self.scheduler.add_noise(z_ref, noise, t)
                ref_prompt_embd = torch.cat([null_prompt_emb] * (len(
                    positive_latents) + len(negative_latents)), dim=0)
                cached_hidden_states = self.get_unet_hidden_states(z_ref_noised
                    , t, ref_prompt_embd)
                n_pos, n_neg = positive_latents.shape[0
                    ], negative_latents.shape[0]
                cached_pos_hs, cached_neg_hs = [], []
                for hs in cached_hidden_states:
                    cached_pos, cached_neg = hs.split([n_pos, n_neg], dim=0)
                    cached_pos = cached_pos.view(1, -1, *cached_pos.shape[2:]
                        ).expand(num_images, -1, -1)
                    cached_neg = cached_neg.view(1, -1, *cached_neg.shape[2:]
                        ).expand(num_images, -1, -1)
                    cached_pos_hs.append(cached_pos)
                    cached_neg_hs.append(cached_neg)
                if n_pos == 0:
                    cached_pos_hs = None
                if n_neg == 0:
                    cached_neg_hs = None
            else:
                cached_pos_hs, cached_neg_hs = None, None
            unet_out = self.unet_forward_with_cached_hidden_states(z_all, t,
                prompt_embd=batched_prompt_embd, cached_pos_hiddens=
                cached_pos_hs, cached_neg_hiddens=cached_neg_hs,
                pos_weights=pos_ws, neg_weights=neg_ws)[0]
            noise_cond, noise_uncond = unet_out.chunk(2)
            guidance = noise_cond - noise_uncond
            noise_pred = noise_uncond + guidance_scale * guidance
            latent_noise = self.scheduler.step(noise_pred, t, latent_noise)[0]
            if i == len(timesteps) - 1 or i + 1 > num_warmup_steps and (i + 1
                ) % self.scheduler.order == 0:
                pbar.update()
    y = self.vae.decode(latent_noise / self.vae.config.scaling_factor,
        return_dict=False)[0]
    imgs = self.image_processor.postprocess(y, output_type=output_type)
    if not return_dict:
        return imgs
    return StableDiffusionPipelineOutput(imgs, False)

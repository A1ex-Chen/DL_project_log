@torch.no_grad()
def __call__(self, prompt: Union[str, List[str]], source_prompt: Union[str,
    List[str]], image: PipelineImageInput=None, strength: float=0.8,
    num_inference_steps: Optional[int]=50, guidance_scale: Optional[float]=
    7.5, source_guidance_scale: Optional[float]=1, num_images_per_prompt:
    Optional[int]=1, eta: Optional[float]=0.1, generator: Optional[Union[
    torch.Generator, List[torch.Generator]]]=None, prompt_embeds: Optional[
    torch.Tensor]=None, output_type: Optional[str]='pil', return_dict: bool
    =True, callback: Optional[Callable[[int, int, torch.Tensor], None]]=
    None, callback_steps: int=1, cross_attention_kwargs: Optional[Dict[str,
    Any]]=None, clip_skip: Optional[int]=None):
    """
        The call function to the pipeline for generation.

        Args:
            prompt (`str` or `List[str]`):
                The prompt or prompts to guide the image generation.
            image (`torch.Tensor` `np.ndarray`, `PIL.Image.Image`, `List[torch.Tensor]`, `List[PIL.Image.Image]`, or `List[np.ndarray]`):
                `Image` or tensor representing an image batch to be used as the starting point. Can also accept image
                latents as `image`, but if passing latents directly it is not encoded again.
            strength (`float`, *optional*, defaults to 0.8):
                Indicates extent to transform the reference `image`. Must be between 0 and 1. `image` is used as a
                starting point and more noise is added the higher the `strength`. The number of denoising steps depends
                on the amount of noise initially added. When `strength` is 1, added noise is maximum and the denoising
                process runs for the full number of iterations specified in `num_inference_steps`. A value of 1
                essentially ignores `image`.
            num_inference_steps (`int`, *optional*, defaults to 50):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference. This parameter is modulated by `strength`.
            guidance_scale (`float`, *optional*, defaults to 7.5):
                A higher guidance scale value encourages the model to generate images closely linked to the text
                `prompt` at the expense of lower image quality. Guidance scale is enabled when `guidance_scale > 1`.
            source_guidance_scale (`float`, *optional*, defaults to 1):
                Guidance scale for the source prompt. This is useful to control the amount of influence the source
                prompt has for encoding.
            num_images_per_prompt (`int`, *optional*, defaults to 1):
                The number of images to generate per prompt.
            eta (`float`, *optional*, defaults to 0.0):
                Corresponds to parameter eta (η) from the [DDIM](https://arxiv.org/abs/2010.02502) paper. Only applies
                to the [`~schedulers.DDIMScheduler`], and is ignored in other schedulers.
            generator (`torch.Generator` or `List[torch.Generator]`, *optional*):
                A [`torch.Generator`](https://pytorch.org/docs/stable/generated/torch.Generator.html) to make
                generation deterministic.
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
            clip_skip (`int`, *optional*):
                Number of layers to be skipped from CLIP while computing the prompt embeddings. A value of 1 means that
                the output of the pre-final layer will be used for computing the prompt embeddings.
        Example:

        ```py
        import requests
        import torch
        from PIL import Image
        from io import BytesIO

        from diffusers import CycleDiffusionPipeline, DDIMScheduler

        # load the pipeline
        # make sure you're logged in with `huggingface-cli login`
        model_id_or_path = "CompVis/stable-diffusion-v1-4"
        scheduler = DDIMScheduler.from_pretrained(model_id_or_path, subfolder="scheduler")
        pipe = CycleDiffusionPipeline.from_pretrained(model_id_or_path, scheduler=scheduler).to("cuda")

        # let's download an initial image
        url = "https://raw.githubusercontent.com/ChenWu98/cycle-diffusion/main/data/dalle2/An%20astronaut%20riding%20a%20horse.png"
        response = requests.get(url)
        init_image = Image.open(BytesIO(response.content)).convert("RGB")
        init_image = init_image.resize((512, 512))
        init_image.save("horse.png")

        # let's specify a prompt
        source_prompt = "An astronaut riding a horse"
        prompt = "An astronaut riding an elephant"

        # call the pipeline
        image = pipe(
            prompt=prompt,
            source_prompt=source_prompt,
            image=init_image,
            num_inference_steps=100,
            eta=0.1,
            strength=0.8,
            guidance_scale=2,
            source_guidance_scale=1,
        ).images[0]

        image.save("horse_to_elephant.png")

        # let's try another example
        # See more samples at the original repo: https://github.com/ChenWu98/cycle-diffusion
        url = (
            "https://raw.githubusercontent.com/ChenWu98/cycle-diffusion/main/data/dalle2/A%20black%20colored%20car.png"
        )
        response = requests.get(url)
        init_image = Image.open(BytesIO(response.content)).convert("RGB")
        init_image = init_image.resize((512, 512))
        init_image.save("black.png")

        source_prompt = "A black colored car"
        prompt = "A blue colored car"

        # call the pipeline
        torch.manual_seed(0)
        image = pipe(
            prompt=prompt,
            source_prompt=source_prompt,
            image=init_image,
            num_inference_steps=100,
            eta=0.1,
            strength=0.85,
            guidance_scale=3,
            source_guidance_scale=1,
        ).images[0]

        image.save("black_to_blue.png")
        ```

        Returns:
            [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] or `tuple`:
                If `return_dict` is `True`, [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] is returned,
                otherwise a `tuple` is returned where the first element is a list with the generated images and the
                second element is a list of `bool`s indicating whether the corresponding generated image contains
                "not-safe-for-work" (nsfw) content.
        """
    self.check_inputs(prompt, strength, callback_steps)
    batch_size = 1 if isinstance(prompt, str) else len(prompt)
    device = self._execution_device
    do_classifier_free_guidance = guidance_scale > 1.0
    text_encoder_lora_scale = cross_attention_kwargs.get('scale', None
        ) if cross_attention_kwargs is not None else None
    prompt_embeds_tuple = self.encode_prompt(prompt, device,
        num_images_per_prompt, do_classifier_free_guidance, prompt_embeds=
        prompt_embeds, lora_scale=text_encoder_lora_scale, clip_skip=clip_skip)
    source_prompt_embeds_tuple = self.encode_prompt(source_prompt, device,
        num_images_per_prompt, do_classifier_free_guidance, None, clip_skip
        =clip_skip)
    if prompt_embeds_tuple[1] is not None:
        prompt_embeds = torch.cat([prompt_embeds_tuple[1],
            prompt_embeds_tuple[0]])
    else:
        prompt_embeds = prompt_embeds_tuple[0]
    if source_prompt_embeds_tuple[1] is not None:
        source_prompt_embeds = torch.cat([source_prompt_embeds_tuple[1],
            source_prompt_embeds_tuple[0]])
    else:
        source_prompt_embeds = source_prompt_embeds_tuple[0]
    image = self.image_processor.preprocess(image)
    self.scheduler.set_timesteps(num_inference_steps, device=device)
    timesteps, num_inference_steps = self.get_timesteps(num_inference_steps,
        strength, device)
    latent_timestep = timesteps[:1].repeat(batch_size * num_images_per_prompt)
    latents, clean_latents = self.prepare_latents(image, latent_timestep,
        batch_size, num_images_per_prompt, prompt_embeds.dtype, device,
        generator)
    source_latents = latents
    extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)
    generator = extra_step_kwargs.pop('generator', None)
    num_warmup_steps = len(timesteps
        ) - num_inference_steps * self.scheduler.order
    with self.progress_bar(total=num_inference_steps) as progress_bar:
        for i, t in enumerate(timesteps):
            latent_model_input = torch.cat([latents] * 2
                ) if do_classifier_free_guidance else latents
            source_latent_model_input = torch.cat([source_latents] * 2
                ) if do_classifier_free_guidance else source_latents
            latent_model_input = self.scheduler.scale_model_input(
                latent_model_input, t)
            source_latent_model_input = self.scheduler.scale_model_input(
                source_latent_model_input, t)
            if do_classifier_free_guidance:
                concat_latent_model_input = torch.stack([
                    source_latent_model_input[0], latent_model_input[0],
                    source_latent_model_input[1], latent_model_input[1]], dim=0
                    )
                concat_prompt_embeds = torch.stack([source_prompt_embeds[0],
                    prompt_embeds[0], source_prompt_embeds[1],
                    prompt_embeds[1]], dim=0)
            else:
                concat_latent_model_input = torch.cat([
                    source_latent_model_input, latent_model_input], dim=0)
                concat_prompt_embeds = torch.cat([source_prompt_embeds,
                    prompt_embeds], dim=0)
            concat_noise_pred = self.unet(concat_latent_model_input, t,
                cross_attention_kwargs=cross_attention_kwargs,
                encoder_hidden_states=concat_prompt_embeds).sample
            if do_classifier_free_guidance:
                (source_noise_pred_uncond, noise_pred_uncond,
                    source_noise_pred_text, noise_pred_text
                    ) = concat_noise_pred.chunk(4, dim=0)
                noise_pred = noise_pred_uncond + guidance_scale * (
                    noise_pred_text - noise_pred_uncond)
                source_noise_pred = (source_noise_pred_uncond + 
                    source_guidance_scale * (source_noise_pred_text -
                    source_noise_pred_uncond))
            else:
                source_noise_pred, noise_pred = concat_noise_pred.chunk(2,
                    dim=0)
            prev_source_latents = posterior_sample(self.scheduler,
                source_latents, t, clean_latents, generator=generator, **
                extra_step_kwargs)
            noise = compute_noise(self.scheduler, prev_source_latents,
                source_latents, t, source_noise_pred, **extra_step_kwargs)
            source_latents = prev_source_latents
            latents = self.scheduler.step(noise_pred, t, latents,
                variance_noise=noise, **extra_step_kwargs).prev_sample
            if i == len(timesteps) - 1 or i + 1 > num_warmup_steps and (i + 1
                ) % self.scheduler.order == 0:
                progress_bar.update()
                if callback is not None and i % callback_steps == 0:
                    step_idx = i // getattr(self.scheduler, 'order', 1)
                    callback(step_idx, t, latents)
    if not output_type == 'latent':
        image = self.vae.decode(latents / self.vae.config.scaling_factor,
            return_dict=False)[0]
        image, has_nsfw_concept = self.run_safety_checker(image, device,
            prompt_embeds.dtype)
    else:
        image = latents
        has_nsfw_concept = None
    if has_nsfw_concept is None:
        do_denormalize = [True] * image.shape[0]
    else:
        do_denormalize = [(not has_nsfw) for has_nsfw in has_nsfw_concept]
    image = self.image_processor.postprocess(image, output_type=output_type,
        do_denormalize=do_denormalize)
    self.maybe_free_model_hooks()
    if not return_dict:
        return image, has_nsfw_concept
    return StableDiffusionPipelineOutput(images=image,
        nsfw_content_detected=has_nsfw_concept)

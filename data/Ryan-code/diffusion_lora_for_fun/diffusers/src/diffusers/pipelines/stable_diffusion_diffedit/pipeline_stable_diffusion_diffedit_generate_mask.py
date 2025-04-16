@torch.no_grad()
@replace_example_docstring(EXAMPLE_DOC_STRING)
def generate_mask(self, image: Union[torch.Tensor, PIL.Image.Image]=None,
    target_prompt: Optional[Union[str, List[str]]]=None,
    target_negative_prompt: Optional[Union[str, List[str]]]=None,
    target_prompt_embeds: Optional[torch.Tensor]=None,
    target_negative_prompt_embeds: Optional[torch.Tensor]=None,
    source_prompt: Optional[Union[str, List[str]]]=None,
    source_negative_prompt: Optional[Union[str, List[str]]]=None,
    source_prompt_embeds: Optional[torch.Tensor]=None,
    source_negative_prompt_embeds: Optional[torch.Tensor]=None,
    num_maps_per_mask: Optional[int]=10, mask_encode_strength: Optional[
    float]=0.5, mask_thresholding_ratio: Optional[float]=3.0,
    num_inference_steps: int=50, guidance_scale: float=7.5, generator:
    Optional[Union[torch.Generator, List[torch.Generator]]]=None,
    output_type: Optional[str]='np', cross_attention_kwargs: Optional[Dict[
    str, Any]]=None):
    """
        Generate a latent mask given a mask prompt, a target prompt, and an image.

        Args:
            image (`PIL.Image.Image`):
                `Image` or tensor representing an image batch to be used for computing the mask.
            target_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts to guide semantic mask generation. If not defined, you need to pass
                `prompt_embeds`.
            target_negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts to guide what to not include in image generation. If not defined, you need to
                pass `negative_prompt_embeds` instead. Ignored when not using guidance (`guidance_scale < 1`).
            target_prompt_embeds (`torch.Tensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs (prompt weighting). If not
                provided, text embeddings are generated from the `prompt` input argument.
            target_negative_prompt_embeds (`torch.Tensor`, *optional*):
                Pre-generated negative text embeddings. Can be used to easily tweak text inputs (prompt weighting). If
                not provided, `negative_prompt_embeds` are generated from the `negative_prompt` input argument.
            source_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts to guide semantic mask generation using DiffEdit. If not defined, you need to
                pass `source_prompt_embeds` or `source_image` instead.
            source_negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts to guide semantic mask generation away from using DiffEdit. If not defined, you
                need to pass `source_negative_prompt_embeds` or `source_image` instead.
            source_prompt_embeds (`torch.Tensor`, *optional*):
                Pre-generated text embeddings to guide the semantic mask generation. Can be used to easily tweak text
                inputs (prompt weighting). If not provided, text embeddings are generated from `source_prompt` input
                argument.
            source_negative_prompt_embeds (`torch.Tensor`, *optional*):
                Pre-generated text embeddings to negatively guide the semantic mask generation. Can be used to easily
                tweak text inputs (prompt weighting). If not provided, text embeddings are generated from
                `source_negative_prompt` input argument.
            num_maps_per_mask (`int`, *optional*, defaults to 10):
                The number of noise maps sampled to generate the semantic mask using DiffEdit.
            mask_encode_strength (`float`, *optional*, defaults to 0.5):
                The strength of the noise maps sampled to generate the semantic mask using DiffEdit. Must be between 0
                and 1.
            mask_thresholding_ratio (`float`, *optional*, defaults to 3.0):
                The maximum multiple of the mean absolute difference used to clamp the semantic guidance map before
                mask binarization.
            num_inference_steps (`int`, *optional*, defaults to 50):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            guidance_scale (`float`, *optional*, defaults to 7.5):
                A higher guidance scale value encourages the model to generate images closely linked to the text
                `prompt` at the expense of lower image quality. Guidance scale is enabled when `guidance_scale > 1`.
            generator (`torch.Generator` or `List[torch.Generator]`, *optional*):
                A [`torch.Generator`](https://pytorch.org/docs/stable/generated/torch.Generator.html) to make
                generation deterministic.
            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generated image. Choose between `PIL.Image` or `np.array`.
            cross_attention_kwargs (`dict`, *optional*):
                A kwargs dictionary that if specified is passed along to the
                [`~models.attention_processor.AttnProcessor`] as defined in
                [`self.processor`](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py).

        Examples:

        Returns:
            `List[PIL.Image.Image]` or `np.array`:
                When returning a `List[PIL.Image.Image]`, the list consists of a batch of single-channel binary images
                with dimensions `(height // self.vae_scale_factor, width // self.vae_scale_factor)`. If it's
                `np.array`, the shape is `(batch_size, height // self.vae_scale_factor, width //
                self.vae_scale_factor)`.
        """
    self.check_inputs(target_prompt, mask_encode_strength, 1,
        target_negative_prompt, target_prompt_embeds,
        target_negative_prompt_embeds)
    self.check_source_inputs(source_prompt, source_negative_prompt,
        source_prompt_embeds, source_negative_prompt_embeds)
    if num_maps_per_mask is None or num_maps_per_mask is not None and (not
        isinstance(num_maps_per_mask, int) or num_maps_per_mask <= 0):
        raise ValueError(
            f'`num_maps_per_mask` has to be a positive integer but is {num_maps_per_mask} of type {type(num_maps_per_mask)}.'
            )
    if mask_thresholding_ratio is None or mask_thresholding_ratio <= 0:
        raise ValueError(
            f'`mask_thresholding_ratio` has to be positive but is {mask_thresholding_ratio} of type {type(mask_thresholding_ratio)}.'
            )
    if target_prompt is not None and isinstance(target_prompt, str):
        batch_size = 1
    elif target_prompt is not None and isinstance(target_prompt, list):
        batch_size = len(target_prompt)
    else:
        batch_size = target_prompt_embeds.shape[0]
    if cross_attention_kwargs is None:
        cross_attention_kwargs = {}
    device = self._execution_device
    do_classifier_free_guidance = guidance_scale > 1.0
    cross_attention_kwargs.get('scale', None
        ) if cross_attention_kwargs is not None else None
    target_negative_prompt_embeds, target_prompt_embeds = self.encode_prompt(
        target_prompt, device, num_maps_per_mask,
        do_classifier_free_guidance, target_negative_prompt, prompt_embeds=
        target_prompt_embeds, negative_prompt_embeds=
        target_negative_prompt_embeds)
    if do_classifier_free_guidance:
        target_prompt_embeds = torch.cat([target_negative_prompt_embeds,
            target_prompt_embeds])
    source_negative_prompt_embeds, source_prompt_embeds = self.encode_prompt(
        source_prompt, device, num_maps_per_mask,
        do_classifier_free_guidance, source_negative_prompt, prompt_embeds=
        source_prompt_embeds, negative_prompt_embeds=
        source_negative_prompt_embeds)
    if do_classifier_free_guidance:
        source_prompt_embeds = torch.cat([source_negative_prompt_embeds,
            source_prompt_embeds])
    image = self.image_processor.preprocess(image).repeat_interleave(
        num_maps_per_mask, dim=0)
    self.scheduler.set_timesteps(num_inference_steps, device=device)
    timesteps, _ = self.get_timesteps(num_inference_steps,
        mask_encode_strength, device)
    encode_timestep = timesteps[0]
    image_latents = self.prepare_image_latents(image, batch_size *
        num_maps_per_mask, self.vae.dtype, device, generator)
    noise = randn_tensor(image_latents.shape, generator=generator, device=
        device, dtype=self.vae.dtype)
    image_latents = self.scheduler.add_noise(image_latents, noise,
        encode_timestep)
    latent_model_input = torch.cat([image_latents] * (4 if
        do_classifier_free_guidance else 2))
    latent_model_input = self.scheduler.scale_model_input(latent_model_input,
        encode_timestep)
    prompt_embeds = torch.cat([source_prompt_embeds, target_prompt_embeds])
    noise_pred = self.unet(latent_model_input, encode_timestep,
        encoder_hidden_states=prompt_embeds, cross_attention_kwargs=
        cross_attention_kwargs).sample
    if do_classifier_free_guidance:
        (noise_pred_neg_src, noise_pred_source, noise_pred_uncond,
            noise_pred_target) = noise_pred.chunk(4)
        noise_pred_source = noise_pred_neg_src + guidance_scale * (
            noise_pred_source - noise_pred_neg_src)
        noise_pred_target = noise_pred_uncond + guidance_scale * (
            noise_pred_target - noise_pred_uncond)
    else:
        noise_pred_source, noise_pred_target = noise_pred.chunk(2)
    mask_guidance_map = torch.abs(noise_pred_target - noise_pred_source
        ).reshape(batch_size, num_maps_per_mask, *noise_pred_target.shape[-3:]
        ).mean([1, 2])
    clamp_magnitude = mask_guidance_map.mean() * mask_thresholding_ratio
    semantic_mask_image = mask_guidance_map.clamp(0, clamp_magnitude
        ) / clamp_magnitude
    semantic_mask_image = torch.where(semantic_mask_image <= 0.5, 0, 1)
    mask_image = semantic_mask_image.cpu().numpy()
    if output_type == 'pil':
        mask_image = self.image_processor.numpy_to_pil(mask_image)
    self.maybe_free_model_hooks()
    return mask_image
